#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# The shard-scale puzzle, decisive test (26-09-2026): long-context NLL of Qwen3.8-27B OUTSIDE vLLM, in Hugging Face
# transformers, on EXACTLY the token sequences ~/logprob_eval.py sends to vLLM (prose + code, 1k/4k/16k/32k).
#
# Arms (one results file each, same format as ~/logprob_eval/<tag>.json: {doc: {"lp": [...], "top1": [...]}}):
#   NVFP4 model (--model nvfp4), decompressed to BF16 on the GPU, activation QDQ off (W4A16 like vLLM's Marlin path):
#     hfT      true per-shard NVFP4 global scales (what transformers does natively: in_proj_qkv / in_proj_z are
#              separate nn.Linear modules, each decompressed with its own weight_global_scale); GDN via fla 0.5.2
#     hfTref   same weights, GDN + causal conv via transformers' pure-PyTorch FP32 reference (no fla, no
#              causal_conv1d): rules out a GDN-kernel bug shared by fla and vLLM (vLLM's GDN kernels derive from fla)
#     hfTchk   same weights, prefill fed in 8192-token chunks through the cache (vLLM's max_num_batched_tokens), to
#              exercise the state hand-off between chunks (only if the one-shot hfT is stable)
#     hfC      vLLM's collapse emulated: every vLLM-fused family (linear_attn [in_proj_qkv, in_proj_z], self_attn
#              [q, k, v], mlp [gate, up]) gets weight *= gs_own / max(gs_family) (compressed-tensors global scales
#              are divisors; vLLM keeps 1/max, compressed_tensors_w4a4_nvfp4.py:111), GDN via fla
#   official FP8 model (--model fp8), block-FP8 dequantised to BF16 on load:
#     hfFP8    GDN via fla
#
# Every per-token log-prob is computed teacher-forced from one full forward (use_cache=False) per document, logits in
# 2048-row slices, log-softmax in fp32, exactly the quantity vLLM returns as prompt_logprobs (raw logprobs).
# Run under $HK_WORK/gpu_lock.sh with vLLM stopped (run_hf_shardscale.sh does all of that).
import argparse
import gc
import glob
import hashlib
import json
import math
import os
import sys
import threading
import time

import torch
import torch.nn.functional as F

NVFP4 = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
FP8_GLOB = "$HK_DATA/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*"
OUT = os.path.expanduser("$HK_WORK/shardscale/results")
LENGTHS = (1024, 4096, 16384, 32768)
CHUNK = 2048
GIB = 1024 ** 3
FUSED = {"linear_attn": ["in_proj_qkv", "in_proj_z"], "self_attn": ["q_proj", "k_proj", "v_proj"],
         "mlp": ["gate_proj", "up_proj"]}
CHECK_LAYERS = (0, 21, 40)


def log(*a):
    print(time.strftime("%d-%m-%Y %H:%M:%S"), *a, flush=True)


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) / 2 ** 20
    rss = 0.0
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            rss = int(line.split()[1]) / 2 ** 20
    return d["MemAvailable"], rss


def watchdog(min_avail=3.5):
    def run():
        while True:
            a, r = meminfo()
            if a < min_avail:
                log(f"RAM WATCHDOG: MemAvailable {a:.1f} GiB < {min_avail} GiB (RSS {r:.1f}) - hard exit")
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=run, daemon=True).start()


def mem():
    a, r = meminfo()
    return (f"GPU alloc {torch.cuda.memory_allocated() / GIB:.1f} GiB (peak {torch.cuda.max_memory_allocated() / GIB:.1f})"
            f" | host avail {a:.1f} GiB, RSS {r:.1f} GiB")


# ---------------------------------------------------------------------------------------------------------------------
# documents: identical to ~/logprob_eval.py corpus() (prose.md + the vLLM source files it concatenates), wiki skipped
# (the HF wikitext id changed, so the vLLM runs never had it either)
def corpus(tok):
    texts = {"prose": open(os.path.expanduser("$HK_WORK/corpus/prose.md"), encoding="utf-8", errors="ignore").read()}
    V = "$HK_VENV/lib/python3.12/site-packages/vllm"
    files = sorted(glob.glob(f"{V}/v1/spec_decode/*.py") + glob.glob(f"{V}/v1/worker/gpu/*.py")
                   + glob.glob(f"{V}/model_executor/layers/*.py"))
    texts["code"] = "\n\n".join(open(f).read() for f in files)
    docs = {}
    for name, t in texts.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        for L in LENGTHS:
            if len(ids) >= L:
                docs[f"{name}_{L}"] = ids[:L]
    return docs


# ---------------------------------------------------------------------------------------------------------------------
# NVFP4 manual dequantisation (to verify what transformers loaded)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def dequant_nvfp4(packed, scale, gs, lo_first=True):
    O = packed.shape[0]
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    idx = torch.stack([lo, hi] if lo_first else [hi, lo], -1).reshape(O, -1)
    w = E2M1[idx]
    return (w.view(O, -1, 16) * scale.float().unsqueeze(-1)).reshape(O, -1) / gs.float().reshape(())


def read_global_scales(model_dir):
    from safetensors import safe_open
    gs = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight_global_scale") and ".language_model.layers." in k:
                gs[k[:-len(".weight_global_scale")]] = f.get_tensor(k).float().reshape(-1)[0].item()
    return gs


def verify_nvfp4_weights(lm, model_dir, report):
    """HF in_proj_qkv / in_proj_z weights vs manual per-shard and max-collapsed dequantisation."""
    from safetensors import safe_open
    out = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt", device="cpu") as f:
        for li in CHECK_LAYERS:
            gsl = {}
            for p in ("in_proj_qkv", "in_proj_z"):
                pre = f"model.language_model.layers.{li}.linear_attn.{p}"
                gsl[p] = f.get_tensor(pre + ".weight_global_scale").float()
            gmax = max(v.item() for v in gsl.values())
            for p in ("in_proj_qkv", "in_proj_z"):
                pre = f"model.language_model.layers.{li}.linear_attn.{p}"
                packed, sc = f.get_tensor(pre + ".weight_packed"), f.get_tensor(pre + ".weight_scale")
                hf = getattr(lm.layers[li].linear_attn, p).weight.detach().float().cpu()
                best = None
                for lo_first in (True, False):
                    man = dequant_nvfp4(packed, sc, gsl[p], lo_first)
                    err = ((hf - man).abs().max() / man.abs().max()).item()
                    if best is None or err < best[0]:
                        best = (err, lo_first, man)
                err, lo_first, man = best
                ratio = ((hf * man).sum() / (man * man).sum()).item()
                out[f"L{li}.{p}"] = {"gs": gsl[p].item(), "gs_max_family": gmax,
                                     "vllm_collapse_factor": gsl[p].item() / gmax,
                                     "hf_over_true_per_shard": round(ratio, 6), "max_rel_err": err,
                                     "nibble_lo_first": lo_first,
                                     "row_norms": man.norm(dim=1).tolist()}
                log(f"  weight check L{li} {p}: gs {gsl[p].item():.1f} (family max {gmax:.1f}, vLLM factor "
                    f"{gsl[p].item() / gmax:.4f}) | HF / true-per-shard = {ratio:.6f}, max rel err {err:.2e}")
    report["weight_check"] = {k: {kk: vv for kk, vv in v.items() if kk != "row_norms"} for k, v in out.items()}
    torch.save({k: torch.tensor(v["row_norms"]) for k, v in out.items()}, f"{OUT}/nvfp4_true_row_norms.pt")
    return out


def apply_vllm_collapse(lm, gs, report):
    """weight *= gs_own / max(gs_family) for every vLLM-fused family: vLLM's stock behaviour on this checkpoint."""
    n, facs = 0, {}
    for li, layer in enumerate(lm.layers):
        for fam, parts in FUSED.items():
            mod = getattr(layer, fam, None)
            if mod is None:
                continue
            keys = [f"model.language_model.layers.{li}.{fam}.{p}" for p in parts]
            if not all(k in gs for k in keys):
                continue
            g = [gs[k] for k in keys]
            gm = max(g)
            for p, gi in zip(parts, g):
                fac = gi / gm
                if fac != 1.0:
                    lin = getattr(mod, p)
                    lin.weight.data.copy_((lin.weight.data.float() * fac).to(lin.weight.dtype))
                    n += 1
                    facs.setdefault(f"{fam}.{p}", []).append(fac)
    summ = {k: {"layers": len(v), "min": min(v), "max": max(v), "mean": sum(v) / len(v)} for k, v in facs.items()}
    log(f"vLLM collapse applied to {n} modules: {json.dumps(summ)}")
    report["collapse"] = {"modules_rescaled": n, "factors": summ}


# ---------------------------------------------------------------------------------------------------------------------
# GDN implementation switch
def gdn_impl(kind):
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    if not hasattr(M, "_orig_impls"):
        M._orig_impls = {n: getattr(M, n) for n in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")}
    if kind == "kernel":
        for n, f in M._orig_impls.items():
            setattr(M, n, f)
        return
    # the pure-PyTorch functions sit in the wrapper's closure (hub_kernels.use_kernel_func_from_hub_with_fallback)
    for n, f in M._orig_impls.items():
        ref = None
        for c in f.__closure__ or ():
            v = c.cell_contents
            if callable(v) and getattr(v, "__module__", "") == M.__name__ and getattr(v, "__name__", "") == n:
                ref = v
        if ref is None:
            ref = getattr(f, "__wrapped__", None)
        assert ref is not None and ref.__module__ == M.__name__, f"no torch reference for {n}"

        def call(*a, _ref=ref, **k):
            k.pop("cu_seqlens", None)
            return _ref(*a, **k)
        setattr(M, n, call)
    log(f"GDN switched to the pure-PyTorch FP32 reference (chunk delta rule + conv1d)")


def describe_impls():
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    out = {}
    for n in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn"):
        f = getattr(M, n)
        mods = [getattr(c.cell_contents, "__module__", None) for c in (f.__closure__ or ())
                if callable(c.cell_contents)]
        out[n] = mods or [getattr(f, "__module__", "?")]
    return out


# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def score(lm, W, ids, chunked=0):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    x = torch.tensor([ids], device="cuda")
    T = x.shape[1]
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):         # never the O(T^2) math fallback
        if not chunked:
            h = lm(input_ids=x, use_cache=False).last_hidden_state[0]
        else:
            from transformers import DynamicCache
            cache, hs = DynamicCache(config=lm.config), []
            for s in range(0, T, chunked):
                e = min(T, s + chunked)
                o = lm(input_ids=x[:, s:e], past_key_values=cache, use_cache=True,
                       position_ids=torch.arange(s, e, device="cuda").view(1, 1, -1).expand(3, 1, -1))
                cache = o.past_key_values
                hs.append(o.last_hidden_state[0])
            h = torch.cat(hs)
            del cache
    lp, top1 = [], []
    for c in range(0, T - 1, CHUNK):
        e = min(T - 1, c + CHUNK)
        logits = F.linear(h[c:e], W).float()
        tgt = x[0, c + 1:e + 1]
        lp.append((logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, -1)))
        top1.append(logits.argmax(-1))
        del logits
    lp, top1 = torch.cat(lp).cpu(), torch.cat(top1).cpu()
    assert torch.isfinite(lp).all(), "non-finite log-prob"
    return lp.tolist(), [str(t) for t in top1.tolist()]


def windows(lp, w=4096):
    return [round(-sum(lp[i:i + w]) / len(lp[i:i + w]), 4) for i in range(0, len(lp), w)]


def run_arm(tag, lm, W, docs, names, report, chunked=0):
    path = f"{OUT}/{tag}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in names:
        if name in res:
            log(f"{tag} {name}: already done, skipping")
            continue
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        lp, top1 = score(lm, W, docs[name], chunked)
        dt = time.time() - t0
        res[name] = {"lp": lp, "top1": top1}
        nll = -sum(lp) / len(lp)
        win = windows(lp) if len(lp) > 4096 else []
        report.setdefault("arms", {}).setdefault(tag, {})[name] = {"n": len(lp), "nll": round(nll, 4),
                                                                  "nll_4k_windows": win, "sec": round(dt, 1)}
        log(f"{tag} {name}: {len(lp)} tokens, mean NLL {nll:.4f} ({dt:.1f}s) windows {win} | {mem()}")
        json.dump(res, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------------
def load_nvfp4():
    from transformers import CompressedTensorsConfig, Qwen3_5ForConditionalGeneration
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        NVFP4, dtype=torch.bfloat16, device_map="cuda", quantization_config=CompressedTensorsConfig(dequantize=True),
        attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    n = 0
    for m in model.modules():      # W4A16 like vLLM's Marlin: no NVFP4 activation fake-quant; free unused scales
        if getattr(m, "quantization_scheme", None) is not None:
            m.quantization_enabled = False
            n += 1
            m._parameters.pop("weight_scale", None)
    return model, n


def load_fp8():
    from transformers import FineGrainedFP8Config, Qwen3_5ForConditionalGeneration
    d = sorted(glob.glob(FP8_GLOB))[-1]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        d, dtype=torch.bfloat16, device_map="cuda", quantization_config=FineGrainedFP8Config(dequantize=True),
        attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    return model, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["nvfp4", "fp8"], required=True)
    ap.add_argument("--arms", default="")
    ap.add_argument("--ref-docs", default="prose_1024,prose_4096,prose_16384,prose_32768,code_32768")
    ap.add_argument("--chk-docs", default="prose_32768")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    watchdog()
    torch.manual_seed(0)
    from transformers import AutoTokenizer
    import transformers
    tok = AutoTokenizer.from_pretrained(NVFP4)
    docs = corpus(tok)
    names = sorted(docs, key=lambda n: (n.split("_")[0], int(n.split("_")[1])))
    report = {"transformers": transformers.__version__, "torch": torch.__version__,
              "docs": {n: {"tokens": len(docs[n]), "sha1": hashlib.sha1(json.dumps(docs[n]).encode()).hexdigest()[:12]}
                       for n in names}}
    log("docs", json.dumps(report["docs"]))
    t0 = time.time()
    if args.model == "nvfp4":
        model, nq = load_nvfp4()
        log(f"NVFP4 loaded in {(time.time() - t0) / 60:.1f} min, activation QDQ off on {nq} modules | {mem()}")
    else:
        model, d = load_fp8()
        log(f"FP8 loaded from {d} in {(time.time() - t0) / 60:.1f} min | {mem()}")
    try:
        model.model.visual = None
    except Exception:  # noqa: BLE001
        pass
    gc.collect(); torch.cuda.empty_cache()
    lm = model.model.language_model
    W = model.lm_head.weight
    bad = {str(p.dtype) for p in lm.parameters() if p.dtype not in (torch.bfloat16, torch.float32)}
    assert not bad, f"language model still has non-float params {bad}"
    report["impls"] = describe_impls()
    log("GDN implementations:", report["impls"], "|", mem())

    arms = [a for a in args.arms.split(",") if a]
    if args.model == "nvfp4":
        gs = read_global_scales(NVFP4)
        verify_nvfp4_weights(lm, NVFP4, report)
        for a in arms:
            if a == "hfT":
                gdn_impl("kernel"); run_arm("hfT", lm, W, docs, names, report)
            elif a == "hfTref":
                gdn_impl("ref"); run_arm("hfTref", lm, W, docs, [n for n in args.ref_docs.split(",") if n], report)
                gdn_impl("kernel")
            elif a == "hfTchk":
                gdn_impl("kernel")
                run_arm("hfTchk", lm, W, docs, [n for n in args.chk_docs.split(",") if n], report, chunked=8192)
            elif a == "hfC":
                gdn_impl("kernel"); apply_vllm_collapse(lm, gs, report)
                run_arm("hfC", lm, W, docs, names, report)
            elif a == "hfCref":
                assert "collapse" in report, "hfCref must come after hfC"
                gdn_impl("ref"); run_arm("hfCref", lm, W, docs, [n for n in args.ref_docs.split(",") if n], report)
                gdn_impl("kernel")
            else:
                raise SystemExit(f"unknown arm {a}")
    else:
        # cross-check: FP8 in_proj row norms vs the NVFP4 true-per-shard dequantisation (should be ~1.0)
        p = f"{OUT}/nvfp4_true_row_norms.pt"
        if os.path.exists(p):
            ref = torch.load(p)
            xc = {}
            for k, v in ref.items():
                li, proj = k[1:].split(".")
                w = getattr(lm.layers[int(li)].linear_attn, proj).weight.detach().float().norm(dim=1).cpu()
                xc[k] = round((w.sum() / v.sum()).item(), 4)
            report["fp8_over_nvfp4_true_row_norm_ratio"] = xc
            log("FP8 / NVFP4-true row-norm ratio:", xc)
        for a in arms:
            assert a == "hfFP8", a
            gdn_impl("kernel"); run_arm("hfFP8", lm, W, docs, names, report)
    rp = f"{OUT}/report_{args.model}.json"
    old = json.load(open(rp)) if os.path.exists(rp) else {}
    for k, v in report.items():
        if k == "arms":
            old.setdefault("arms", {})
            for t, d in v.items():
                old["arms"].setdefault(t, {}).update(d)
        else:
            old[k] = v
    json.dump(old, open(rp, "w"), indent=1)
    log(f"done in {(time.time() - t0) / 60:.1f} min | {mem()}")


if __name__ == "__main__":
    main()
