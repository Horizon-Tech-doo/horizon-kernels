# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Short diagnostic only; derived from unfinished ref_nll.py. No clipping hooks.
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

out = {"_meta": {"tag": tag, "shard_scales": ss, "harness": "trainctx r2, one pass", "time": time.strftime("%d-%m-%Y %H:%M")}}
model.eval()
for n in names:
    ids = torch.tensor(docs[n]["ids"], device="cuda")
    t1 = time.time()
    with torch.no_grad():
        h = model.hidden(ids)[0]
        lps = []
        for start in range(0, ids.shape[0] - 1, 512):
            end = min(start + 512, ids.shape[0] - 1)
            lp = torch.log_softmax(model.logits(h[start:end]).float(), dim=-1)
            lps.append(lp.gather(1, ids[start + 1:end + 1, None]).squeeze(1).cpu())
            del lp
    lp = torch.cat(lps).tolist()
    assert len(lp) == len(ids) - 1
    del h
    torch.cuda.empty_cache()
    out[n] = {"lp": lp}
    print(f"inv-{tag} {n}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f} ({time.time() - t1:.0f} s)", flush=True)
    json.dump(out, open(os.path.expanduser(f"~/logprob_eval/inv-{tag}.json"), "w"))
