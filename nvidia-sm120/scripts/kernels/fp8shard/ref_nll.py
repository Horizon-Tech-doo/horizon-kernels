# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# ref_nll.py TAG SHARD_SCALES NAMES (fp8fix investigation, 27-09-2026): the key test. Per-token log-probs of the long
# documents through the training harness's HF-style forward (scripts/training/trainctx.py: our NVFP4 checkpoint,
# packed r2 base, BF16 compute, the whole document in ONE pass, FLA GDN + flash attention), with vLLM STOPPED.
#   SHARD_SCALES: served (vLLM's collapsed max scale = production) | exact (true per-shard scales)
# Output ~/logprob_eval/inv-TAG.json (same format as invest_eval.py). With served: also activation statistics of the
# FP8 projections' inputs against NVIDIA's static input_scale x 448 (e4m3 limit), per layer and 2048-token bucket ->
# $HK_WORK/fp8fix/clip-TAG.json (our model's activations stand in for NVIDIA's: same architecture, same base weights).
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("$HK_WORK/trainctx"))
import torch
import trainctx as T

tag, ss, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
F = os.path.expanduser("$HK_WORK/fp8fix")
docs = json.load(open(f"{F}/invest_docs_q38.json"))
pr = json.load(open(os.path.expanduser("$HK_WORK/trainctx/probe.json")))
import torch.nn.attention as tna
T.Impl.sdpa_backends = [getattr(tna.SDPBackend, pr["sdpa_choice"])]
T.Impl.conv = "cuda" if pr.get("causal_conv1d", {}).get("ok") else "torch"
T._TRITON["ok"] = bool(pr.get("triton_dequant", {}).get("ok"))
t0 = time.time()
model, t_load = T.build_model(T.MODEL, "r2", "cuda", shard_scales=ss)
print(f"ref {tag}: model built ({ss}) in {time.time() - t0:.0f} s", flush=True)

clip, hooks = {}, []
if ss == "served":
    from safetensors import safe_open
    NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4"
    idx = json.load(open(f"{NV}/model.safetensors.index.json"))["weight_map"]
    fh = {}

    def nv_scale(key):
        f = idx.get(key)
        if f is None:
            return None
        if f not in fh:
            fh[f] = safe_open(f"{NV}/{f}", "pt")
        return float(fh[f].get_tensor(key))

    for i, layer in enumerate(model.layers):
        blk, mods = (("linear_attn", ("in_proj_qkv", "out_proj")) if hasattr(layer, "linear_attn")
                     else ("self_attn", ("q_proj", "o_proj")))
        for mn in mods:
            s = nv_scale(f"model.language_model.layers.{i}.{blk}.{mn}.input_scale")
            if s is None:
                continue
            key = f"{i}.{mn}"
            clip[key] = {"input_scale": s, "limit": 448.0 * s, "buckets": {}}

            def hook(mod, args, _k=key, _lim=448.0 * s):
                x = args[0].detach()
                x = x.reshape(-1, x.shape[-1])
                doc = clip["_doc"]
                for b in range(0, x.shape[0], 2048):
                    xb = x[b:b + 2048].float().abs()
                    d = clip[_k]["buckets"].setdefault(doc, [])
                    d.append({"b": b, "absmax": round(xb.max().item(), 3), "clipped": int((xb > _lim).sum().item()),
                              "n": xb.numel()})
            hooks.append(getattr(getattr(layer, blk), mn).register_forward_pre_hook(hook))

out = {"_meta": {"tag": tag, "shard_scales": ss, "harness": "trainctx r2, one pass", "time": time.strftime("%d-%m-%Y %H:%M")}}
for n in names:
    ids = torch.tensor(docs[n]["ids"], device="cuda")
    clip["_doc"] = n
    t1 = time.time()
    with torch.no_grad():
        h = model.hidden(ids)[0]
        lps = []
        for s in range(0, ids.shape[0] - 1, 2048):
            e = min(s + 2048, ids.shape[0] - 1)
            lp = torch.log_softmax(model.logits(h[s:e]).float(), dim=-1)
            lps.append(lp.gather(1, ids[s + 1:e + 1, None]).squeeze(1).cpu())
            del lp
    lp = torch.cat(lps).tolist()
    del h
    torch.cuda.empty_cache()
    out[n] = {"lp": lp}
    print(f"inv-{tag} {n}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f} ({time.time() - t1:.0f} s)", flush=True)
    json.dump(out, open(os.path.expanduser(f"~/logprob_eval/inv-{tag}.json"), "w"))
if clip:
    clip.pop("_doc", None)
    json.dump(clip, open(f"{F}/clip-{tag}.json", "w"))
    tot = sum(x["clipped"] for v in clip.values() for d in v["buckets"].values() for x in d)
    worst = sorted(((max(x["absmax"] / v["limit"] for d in v["buckets"].values() for x in d), k) for k, v in clip.items()), reverse=True)[:5]
    print(f"clip-{tag}: elements above NVIDIA's 448 x input_scale: {tot}; worst absmax/limit: {worst}", flush=True)
