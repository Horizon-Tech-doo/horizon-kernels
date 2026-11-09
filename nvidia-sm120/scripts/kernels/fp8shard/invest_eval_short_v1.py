# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# invest_eval.py TAG DOCSET NAMES [--ruler] (fp8fix, 27-09-2026): one vLLM arm of the long-context investigation.
#   DOCSET: q38 | moe ($HK_WORK/fp8fix/invest_docs_<set>.json from invest_docs.py); NAMES: comma list or "all16"
#   --ruler: RULER-style synthetic recall at 16k and 32k (multi-key needle, variable tracking, 2-hop QA; 4 each)
# Per-token prompt log-probs (same request as logprob_eval.py) -> ~/logprob_eval/inv-TAG.json
import json, os, random, re, sys, time, urllib.request
F = os.path.expanduser("$HK_WORK/fp8fix")
OUT = os.path.expanduser("~/logprob_eval")
BASE = "http://localhost:8001"
tag, dset, names = sys.argv[1], sys.argv[2], sys.argv[3]
docs = json.load(open(f"{F}/invest_docs_{dset}.json"))
if names == "all16":
    names = ",".join(sorted(k for k in docs if k.endswith("_16384")))


def post(path, body, timeout=1800):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=timeout))


def prompt_lp(ids):
    r = post("/v1/completions", {"model": "qwen3.8", "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
    lp = []
    for pos, d in enumerate(r["choices"][0]["prompt_logprobs"]):
        if d is not None:
            lp.append(d[str(ids[pos])]["logprob"])
    return lp


out = {"_meta": {"tag": tag, "docset": dset, "time": time.strftime("%d-%m-%Y %H:%M")}}
for n in names.split(","):
    if not n:
        continue
    t0 = time.time()
    lp = prompt_lp(docs[n]["ids"])
    assert len(lp) == len(docs[n]["ids"]) - 1
    out[n] = {"lp": lp}
    print(f"inv-{tag} {n}: {len(lp)} tokens, mean NLL {-sum(lp) / len(lp):.4f} ({time.time() - t0:.0f} s)", flush=True)
    json.dump(out, open(f"{OUT}/inv-{tag}.json", "w"))
