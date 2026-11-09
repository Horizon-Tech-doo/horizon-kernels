#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""REAP expert saliency for Qwen3.8-Flash-Next (Qwen4Exp) inside vLLM (03-10-2026).

  reap_calibrate.py run    --model DIR --calib calib.jsonl --out DIR [--mtp 3] [--score-mode kernel|routing]
  reap_calibrate.py verify --model DIR --out DIR [--layers 4]            (CPU, numpy: offline cross-check)
  reap_calibrate.py report --out DIR                                       (coverage summary of scores.npz)

`run` loads the checkpoint in vLLM with enforce_eager and prefix caching off (every calibration token goes through
every MoE block exactly once), wraps every MoE runner (horizon_flashnext.hooks), prefills the calibration set and
accumulates per block and expert:
    count     tokens routed to the expert (it was in the token's top-10)
    gsum      sum of its renormalized gate values
    reap_sum  sum of gate * ||expert output||_2        REAP score = reap_sum / count   (arXiv:2510.13999)
    ean_sum   sum of ||expert output||_2
The expert outputs come from vLLM's own expert kernel (one slot, weight 1), i.e. the served numerics; force
`--moe-backend marlin` (the default here) so they are the numerics of our SM120 box. With --mtp N the MTP drafter runs
too and its MoE block (mtp.layers.0.mlp) is scored; without it, prune with --mtp drop.

Output DIR: scores.npz ('<checkpoint prefix>|count|gsum|reap_sum|ean_sum|tokens' + 'meta'), meta.json, samples/
(<=256 tokens per block: hidden, routing, norms, logits) for `verify`, which recomputes routing from the checkpoint's
router weights and expert norms from the dequantized NVFP4/FP8/BF16 expert weights (W4A16 view: activations not
quantized) and reports the agreement.

Calibration data: calib_set.py (public sources only, no customer data, decontaminated against our eval sets).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fn_common as fc  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(tok, rec, max_tokens):
    if "messages" in rec:
        kw = {"tools": rec["tools"]} if rec.get("tools") else {}
        try:
            text = tok.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=False, **kw)
        except Exception:  # noqa: BLE001  (a template that rejects e.g. a tool role: fall back to plain turns)
            text = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in rec["messages"])
    else:
        text = rec["text"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


def cmd_run(a):
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")   # collective_rpc with our functions
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415
    import vllm  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from horizon_flashnext import hooks  # noqa: PLC0415

    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = fc.text_config(cfg)
    nhl, E, K = int(tc["num_hidden_layers"]), int(tc["num_experts"]), int(tc["num_experts_per_tok"])
    tok = AutoTokenizer.from_pretrained(a.model)
    recs = [json.loads(l) for l in open(a.calib) if l.strip()]
    if a.limit:
        recs = recs[: a.limit]
    prompts, src_tokens = [], {}
    for r in recs:
        ids = render(tok, r, a.max_sample_tokens)
        if len(ids) < 16:
            continue
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        src_tokens[r.get("source", "?")] = src_tokens.get(r.get("source", "?"), 0) + len(ids)
    total = sum(src_tokens.values())
    print(f"{len(prompts)} calibration prompts, {total:,} tokens: "
          + ", ".join(f"{k} {v:,}" for k, v in sorted(src_tokens.items())), flush=True)

    engine = dict(model=a.model, enforce_eager=True, enable_prefix_caching=False, max_model_len=a.max_model_len,
                  gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs,
                  max_num_batched_tokens=a.max_num_batched_tokens, language_model_only=True,
                  kv_cache_dtype="auto")
    if a.moe_backend:
        engine["moe_backend"] = a.moe_backend
    if a.mtp:
        engine["speculative_config"] = {"method": "mtp", "num_speculative_tokens": a.mtp}
    for kv in a.engine_arg:
        k, v = kv.split("=", 1)
        try:
            engine[k] = json.loads(v)
        except json.JSONDecodeError:
            engine[k] = v
    t0 = time.time()
    llm = LLM(**engine)
    print(f"engine up in {time.time() - t0:.0f} s", flush=True)
    names = llm.collective_rpc(hooks.rpc_install_calibration,
                               kwargs={"score_mode": a.score_mode, "sample_tokens": a.sample_tokens,
                                       "sample_dir": os.path.join(a.out, "samples")})[0]
    print(f"{len(names)} MoE blocks hooked: {names[0][0]} .. {names[-1][0]}", flush=True)
    if a.mtp and not any(p.startswith("mtp.") for p, _ in names):
        print("WARNING: --mtp given but no drafter MoE block was found; the MTP block will have no scores", flush=True)
    sp = SamplingParams(max_tokens=max(1, a.gen_tokens if not a.mtp else max(2, a.gen_tokens)), temperature=0.0)

    llm.collective_rpc(hooks.rpc_set_active, kwargs={"active": True})
    llm.generate(prompts[:2], sp, use_tqdm=False)                      # fail fast: kernel path works?
    st = llm.collective_rpc(hooks.rpc_status)[0]
    if st["errors"] or not st["calls"]:
        raise SystemExit(f"calibration hook failed on the first prompts: {st}. "
                         "If forward_modular is the problem, rerun with --score-mode routing (frequency/gate mass "
                         "only) and report it.")
    t1 = time.time()
    for i in range(2, len(prompts), a.chunk):
        llm.generate(prompts[i:i + a.chunk], sp, use_tqdm=False)
        st = llm.collective_rpc(hooks.rpc_status)[0]
        if st["errors"]:
            raise SystemExit(f"calibration hook error: {st['errors'][:3]}")
        done = min(len(prompts), i + a.chunk)
        print(f"  {done}/{len(prompts)} prompts, {time.time() - t1:.0f} s", flush=True)
    llm.collective_rpc(hooks.rpc_set_active, kwargs={"active": False})
    res = llm.collective_rpc(hooks.rpc_results)[0]

    arrays, per_layer = {}, {}
    for prefix, r in sorted(res.items()):
        for s in ("count", "gsum", "reap_sum", "ean_sum"):
            arrays[f"{prefix}|{s}"] = np.asarray(r[s], dtype=np.float64)
        arrays[f"{prefix}|tokens"] = np.array([r["tokens"]], dtype=np.int64)
        c = np.asarray(r["count"])
        per_layer[prefix] = {"tokens": int(r["tokens"]), "unrouted_experts": int((c == 0).sum()),
                             "min_count": int(c.min()), "median_count": float(np.median(c))}
    meta = {"model": os.path.abspath(a.model), "index_sha256": sha256_file(os.path.join(a.model,
            "model.safetensors.index.json")) if os.path.exists(os.path.join(a.model, "model.safetensors.index.json"))
            else None, "calib": os.path.abspath(a.calib), "calib_sha256": sha256_file(a.calib),
            "prompts": len(prompts), "tokens": total, "tokens_by_source": src_tokens, "num_experts": E, "top_k": K,
            "num_hidden_layers": nhl, "score_mode": a.score_mode, "moe_backend": a.moe_backend, "mtp": a.mtp,
            "vllm": getattr(vllm, "__version__", "?"), "engine": {k: v for k, v in engine.items() if k != "model"},
            "date": datetime.datetime.now().strftime("%d-%m-%Y %H:%M"), "seconds": round(time.time() - t1),
            "per_layer": per_layer}
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(os.path.join(a.out, "scores.npz"), **arrays)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"scores -> {a.out}/scores.npz ({len(res)} MoE blocks, {meta['seconds']} s)", flush=True)
    cmd_report(argparse.Namespace(out=a.out))


def cmd_report(a):
    import numpy as np  # noqa: PLC0415
    sys.path.insert(0, HERE)
    from prune_checkpoint import load_scores  # noqa: PLC0415
    scores, meta = load_scores(os.path.join(a.out, "scores.npz"))
    rows = []
    for p, st in sorted(scores.items()):
        c = st["count"]
        reap = np.where(c > 0, st["reap_sum"] / np.where(c > 0, c, 1), 0)
        top = np.sort(reap)[::-1]
        mass = np.sort(st["reap_sum"])[::-1]
        cum = np.cumsum(mass) / max(mass.sum(), 1e-30)
        rows.append({"block": p, "unrouted": int((c == 0).sum()), "min_count": int(c.min()),
                     "reap_mass_top65pct": round(float(cum[int(len(c) * 0.65) - 1]), 4),
                     "reap_mass_top50pct": round(float(cum[int(len(c) * 0.50) - 1]), 4),
                     "reap_max_over_median": round(float(top[0] / max(np.median(top), 1e-30)), 2)})
    for r in rows:
        print(json.dumps(r))
    print(json.dumps({"blocks": len(rows), "tokens": meta.get("tokens"), "score_mode": meta.get("score_mode")}))


# ---------------------------------------------------------------------------------------------------------------------
# offline verification (numpy, CPU)
# ---------------------------------------------------------------------------------------------------------------------

class CkptReader:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.wmap, _ = fc.load_index(model_dir)
        self._hdr = {}

    def info(self, name):
        sh = self.wmap[name]
        if sh not in self._hdr:
            self._hdr[sh] = fc.read_header(os.path.join(self.dir, sh))[0]
        return self._hdr[sh][name]

    def has(self, name):
        return name in self.wmap

    def array(self, name):
        import numpy as np  # noqa: PLC0415
        ti = self.info(name)
        raw = fc.read_bytes(ti)
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8,
              "I8": np.int8, "F8_E8M0": np.uint8}[ti.dtype]
        a = np.frombuffer(raw, dtype=dt).reshape(ti.shape)
        if ti.dtype == "BF16":
            return fc.bf16_to_f32(a)
        return a

    def linear(self, base):
        """float32 [out, in] for '<base>.weight' in BF16, NVFP4 (weight_scale + weight_scale_2) or FP8 block."""
        w = self.info(base + ".weight")
        if w.dtype == "BF16":
            return self.array(base + ".weight")
        if self.has(base + ".weight_scale_2"):
            g = float(self.array(base + ".weight_scale_2").reshape(-1)[0])
            return fc.dequant_nvfp4(self.array(base + ".weight"), self.array(base + ".weight_scale"), g)
        if self.has(base + ".weight_scale_inv"):
            return fc.dequant_fp8_block(self.array(base + ".weight"), self.array(base + ".weight_scale_inv"))
        raise ValueError(f"{base}: unsupported weight format {w.dtype}")


def expert_out(rd, prefix, e, x):
    import numpy as np  # noqa: PLC0415
    b = f"{prefix}.experts.{e}"
    g = x @ rd.linear(b + ".gate_proj").T
    u = x @ rd.linear(b + ".up_proj").T
    h = (g / (1.0 + np.exp(-g))) * u
    return h @ rd.linear(b + ".down_proj").T


def verify_block(rd, prefix, sample, top_k):
    import numpy as np  # noqa: PLC0415
    x = fc.bf16_to_f32(sample["hidden"].view(np.uint16))
    ids, w, norms = sample["ids"], sample["weights"], sample["norms"]
    E = sample["logits"].shape[1]
    wg = rd.array(prefix + ".gate.weight")[:E]
    logits = x @ wg.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    top = np.argsort(-p, axis=1)[:, :top_k]
    agree = np.mean([set(top[i]) == set(int(v) for v in ids[i] if v < E) for i in range(len(x))])
    rel = []
    cache = {}
    for i in range(len(x)):
        for s in range(ids.shape[1]):
            e = int(ids[i, s])
            if e >= E or norms[i, s] == 0:
                continue
            if e not in cache:
                cache[e] = None
            ref = float(np.linalg.norm(expert_out(rd, prefix, e, x[i:i + 1])))
            rel.append(abs(ref - float(norms[i, s])) / max(ref, 1e-12))
    rel = np.array(rel) if rel else np.array([np.nan])
    return {"block": prefix, "tokens": int(len(x)), "routing_set_agreement": round(float(agree), 4),
            "norm_rel_err_median": round(float(np.median(rel)), 5), "norm_rel_err_p99": round(float(np.quantile(rel, .99)), 5),
            "pairs": int(np.isfinite(rel).sum())}


def cmd_verify(a):
    import glob  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    rd = CkptReader(a.model)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    top_k = int(fc.text_config(cfg)["num_experts_per_tok"])
    files = sorted(glob.glob(os.path.join(a.out, "samples", "*.npz")))
    if a.layers:
        step = max(1, len(files) // a.layers)
        files = files[::step][: a.layers]
    out = []
    for f in files:
        prefix = os.path.basename(f)[:-4]
        s = dict(np.load(f))
        if a.tokens:
            s = {k: v[: a.tokens] for k, v in s.items()}
        r = verify_block(rd, prefix, s, top_k)
        print(json.dumps(r), flush=True)
        out.append(r)
    ok = all(r["routing_set_agreement"] >= a.min_routing_agreement and
             (np.isnan(r["norm_rel_err_median"]) or r["norm_rel_err_median"] <= a.max_norm_err) for r in out)
    print(json.dumps({"verify": "pass" if ok else "FAIL", "blocks": len(out),
                      "thresholds": {"routing": a.min_routing_agreement, "norm_median": a.max_norm_err}}))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--calib", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mtp", type=int, default=0, help="speculative MTP drafts (also scores the MTP MoE block)")
    r.add_argument("--score-mode", choices=("kernel", "routing"), default="kernel")
    r.add_argument("--moe-backend", default="marlin")
    r.add_argument("--max-sample-tokens", type=int, default=16384)
    r.add_argument("--max-model-len", type=int, default=20480)
    r.add_argument("--max-num-seqs", type=int, default=32)
    r.add_argument("--max-num-batched-tokens", type=int, default=16384)
    r.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    r.add_argument("--gen-tokens", type=int, default=1)
    r.add_argument("--chunk", type=int, default=256, help="prompts per generate() call (progress granularity)")
    r.add_argument("--sample-tokens", type=int, default=256)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--engine-arg", action="append", default=[], help="extra LLM(...) kwarg KEY=JSON (repeatable)")
    v = sub.add_parser("verify")
    v.add_argument("--model", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--layers", type=int, default=4)
    v.add_argument("--tokens", type=int, default=64)
    v.add_argument("--min-routing-agreement", type=float, default=0.97)
    v.add_argument("--max-norm-err", type=float, default=0.03)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "verify": cmd_verify, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
