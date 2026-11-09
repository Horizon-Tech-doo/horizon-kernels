#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache determinism + drift probe, 16 prompts (27-09-2026, experiment 4: why --prefix-match-unit changes
outputs). For each prompt (3.5k-23k tokens, lengths chosen so the uncached tails vary):
    uncached_1 (salt X), cached (salt X again), uncached_2 (salt Y)
greedy (temperature 0), 128 tokens, thinking off, with per-token logprobs (top 5).

Drift is measured on the SHARED greedy prefix: up to the first token where the cached answer departs from uncached_1,
both runs scored the same token sequence, so comparing their per-token logprobs there is teacher forcing on identical
context. At the first divergent position the two runs' top-1 tokens differ (a top-1 disagreement) and the uncached
token's logprob in the cached run is read from its top-5 when present. Compare with logprob_eval.py's kernel-swap
drift (mean |dlogp| ~0.022-0.025). uncached_2 vs uncached_1 gives the run-to-run floor (0 if deterministic).
(Why not prompt_logprobs: vLLM 0.29 sets skip_reading_prefix_cache for any request with prompt_logprobs, so the
cached path cannot be scored that way.)
    pmu_det16.py TAG [--url http://localhost:8001] [--n 16]   -> $HK_WORK/pmudiag/det16.jsonl + summary line"""
import argparse, json, os, random, statistics, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--url", default="http://localhost:8001")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--corpus", default=os.path.expanduser("$HK_WORK/prefix/corpus.json"))
ap.add_argument("--tokenizer", default="$HK_DATA/models/Qwen3.8-27B-NVFP4")
ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/pmudiag/det16.jsonl"))
A = ap.parse_args()
SIZES = [3500, 4200, 5100, 6300, 7000, 8100, 8800, 9700, 10500, 11800, 12900, 14200, 15500, 17100, 19800, 23300]
SYSTEM = ("You are Code Assistant, an autonomous coding agent working inside a software repository. Read before you "
          "write, keep changes minimal, and refer to code with file paths and line numbers.")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.tokenizer)


def cut(s, n):
    ids = tok.encode(s, add_special_tokens=False)
    return s if len(ids) <= n else tok.decode(ids[:n])


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))


FILES = json.load(open(A.corpus))["files"]


def prompt(k, n_tokens):
    rng = random.Random(f"det16-{k}")
    parts, have = [], 0
    for i in rng.sample(range(len(FILES)), 80):
        t = cut(FILES[i]["text"], n_tokens - have)
        parts.append(f"### {FILES[i]['path']}\n{t}")
        have += ntok(t) + 10
        if have >= n_tokens:
            break
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts) + "\n\nList the five most important functions above and "
             "say in one sentence each what they do."}]


def chat(msgs, salt):
    body = {"model": "qwen3.8", "messages": msgs, "max_tokens": 128, "temperature": 0, "seed": 1,
            "logprobs": True, "top_logprobs": 5, "cache_salt": salt}
    req = urllib.request.Request(A.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    ch = r["choices"][0]
    lp = [(c["token"], c["logprob"], {t["token"]: t["logprob"] for t in c.get("top_logprobs") or []})
          for c in (ch.get("logprobs") or {}).get("content") or []]
    return {"text": ch["message"].get("content") or "", "lp": lp, "e2e": time.time() - t0,
            "prompt_tokens": r["usage"]["prompt_tokens"],
            "cached_tokens": ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)}


def compare(a, b):
    """Positions of the shared prefix (identical tokens): |dlogp| of the chosen token; then the first divergence."""
    d, n = [], 0
    for (ta, la, _), (tb, lb, topb) in zip(a["lp"], b["lp"]):
        if ta != tb:
            x = topb.get(ta)
            return {"shared": n, "absd": d, "diverged": True,
                    "div_absd": None if x is None else abs(la - x)}
        d.append(abs(la - lb)); n += 1
    return {"shared": n, "absd": d, "diverged": len(a["lp"]) != len(b["lp"]), "div_absd": None}


rows = []
for k, n in enumerate(SIZES[:A.n]):
    msgs = prompt(k, n)
    sx, sy = "d16-" + uuid.uuid4().hex, "d16-" + uuid.uuid4().hex
    u1 = chat(msgs, sx); time.sleep(0.5)
    c = chat(msgs, sx); time.sleep(0.5)
    u2 = chat(msgs, sy)
    cu, uu = compare(u1, c), compare(u1, u2)
    row = {"k": k, "prompt_tokens": c["prompt_tokens"], "cached": c["cached_tokens"],
           "tail": c["prompt_tokens"] - c["cached_tokens"],
           "identical": u1["text"] == c["text"], "uncached_repeatable": u1["text"] == u2["text"],
           "shared_tokens": cu["shared"], "gen_tokens": len(u1["lp"]), "diverged": cu["diverged"],
           "mean_absd_shared": round(statistics.mean(cu["absd"]), 5) if cu["absd"] else None,
           "max_absd_shared": round(max(cu["absd"]), 5) if cu["absd"] else None,
           "div_absd": cu["div_absd"], "noise_mean_absd": round(statistics.mean(uu["absd"]), 5) if uu["absd"] else None,
           "absd": cu["absd"]}
    rows.append(row)
    print("det16 " + json.dumps({x: y for x, y in row.items() if x != "absd"}), flush=True)

allabs = [v for r in rows for v in r["absd"]]
npos = len(allabs) + sum(1 for r in rows if r["diverged"])          # compared positions incl. divergence points
res = {"tag": A.tag, "time": time.strftime("%d-%m-%Y %H:%M"), "n": len(rows),
       "identical": sum(r["identical"] for r in rows), "uncached_repeatable": sum(r["uncached_repeatable"] for r in rows),
       "mean_absd": round(statistics.mean(allabs), 5) if allabs else None,
       "max_absd": round(max(allabs), 5) if allabs else None,
       "top1_agree": round(1 - sum(1 for r in rows if r["diverged"]) / max(1, npos), 5),
       "positions": npos, "div_absd_mean": round(statistics.mean([r["div_absd"] for r in rows
                                                                    if r["div_absd"] is not None]), 4)
       if any(r["div_absd"] is not None for r in rows) else None,
       "tails": [r["tail"] for r in rows]}
print("DET16 " + json.dumps(res), flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "a") as f:
    f.write(json.dumps(dict(res, rows=rows)) + "\n")
