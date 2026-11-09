#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Agent-style follow-up probe for the prefix-match-unit + MTP fix (27-09-2026, HORIZON_PMU_MTP_FIX).

For each of N prompts (the det16 prompt builder, 5k-17k tokens):
    turn 1 (salt X)                 greedy, 64 tokens
    turn 2 = turn 1 + its answer + a new ~1.5k-token "tool result" + a question
    turn 2 cached (salt X)  vs  turn 2 cold (fresh salt Y)      greedy, 64 tokens, logprobs
Reports turn 2's cached tokens (how much of turn 1 is reused), turn 1's prompt length, and whether the cached
answer equals the cold one (same comparison as pmu_det16.py: shared-prefix |dlogp|, identical text).
    pmu_followup.py TAG [--url http://localhost:8001] [--n 6]  -> $HK_WORK/pmufix/followup.jsonl + FOLLOWUP line"""
import argparse, json, os, random, statistics, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--url", default="http://localhost:8001")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--corpus", default=os.path.expanduser("$HK_WORK/prefix/corpus.json"))
ap.add_argument("--tokenizer", default="$HK_DATA/models/Qwen3.8-27B-NVFP4")
ap.add_argument("--out", default=os.path.expanduser("$HK_WORK/pmufix/followup.jsonl"))
A = ap.parse_args()
SIZES = [5100, 8800, 12900, 17100, 7000, 10500, 3500, 15500]
SYSTEM = ("You are Code Assistant, an autonomous coding agent working inside a software repository. Read before you "
          "write, keep changes minimal, and refer to code with file paths and line numbers.")

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(A.tokenizer)
FILES = json.load(open(A.corpus))["files"]


def cut(s, n):
    ids = tok.encode(s, add_special_tokens=False)
    return s if len(ids) <= n else tok.decode(ids[:n])


def ntok(s):
    return len(tok.encode(s, add_special_tokens=False))


def files_text(seed, n_tokens):
    rng = random.Random(seed)
    parts, have = [], 0
    for i in rng.sample(range(len(FILES)), 80):
        t = cut(FILES[i]["text"], n_tokens - have)
        parts.append(f"### {FILES[i]['path']}\n{t}")
        have += ntok(t) + 10
        if have >= n_tokens:
            break
    return "\n\n".join(parts)


def chat(msgs, salt):
    body = {"model": "qwen3.8", "messages": msgs, "max_tokens": 64, "temperature": 0, "seed": 1,
            "logprobs": True, "top_logprobs": 5, "cache_salt": salt}
    req = urllib.request.Request(A.url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    ch = r["choices"][0]
    lp = [(c["token"], c["logprob"]) for c in (ch.get("logprobs") or {}).get("content") or []]
    return {"text": ch["message"].get("content") or "", "lp": lp, "prompt_tokens": r["usage"]["prompt_tokens"],
            "cached_tokens": ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)}


rows = []
for k, n in enumerate(SIZES[:A.n]):
    m1 = [{"role": "system", "content": SYSTEM},
          {"role": "user", "content": files_text(f"fu-{k}", n) + "\n\nWhich function above is the most complex? "
           "Answer in two sentences."}]
    sx, sy = "fu-" + uuid.uuid4().hex, "fu-" + uuid.uuid4().hex
    t1 = chat(m1, sx); time.sleep(0.3)
    m2 = m1 + [{"role": "assistant", "content": t1["text"]},
               {"role": "user", "content": "Tool result (read_file):\n" + files_text(f"fu2-{k}", 1500) +
                "\n\nNow list three functions from this new file and say what each does."}]
    warm = chat(m2, sx); time.sleep(0.3)
    cold = chat(m2, sy)
    shared, d = 0, []
    for (ta, la), (tb, lb) in zip(cold["lp"], warm["lp"]):
        if ta != tb:
            break
        d.append(abs(la - lb)); shared += 1
    row = {"k": k, "turn1_prompt": t1["prompt_tokens"], "turn2_prompt": warm["prompt_tokens"],
           "turn2_cached": warm["cached_tokens"], "turn2_recomputed": warm["prompt_tokens"] - warm["cached_tokens"],
           "turn1_reused_frac": round(warm["cached_tokens"] / t1["prompt_tokens"], 4),
           "cold_cached": cold["cached_tokens"], "identical": warm["text"] == cold["text"], "shared_tokens": shared,
           "mean_absd_shared": round(statistics.mean(d), 5) if d else None}
    rows.append(row)
    print("followup " + json.dumps(row), flush=True)

res = {"tag": A.tag, "time": time.strftime("%d-%m-%Y %H:%M"), "n": len(rows),
       "identical": sum(r["identical"] for r in rows),
       "turn2_cached_share": round(sum(r["turn2_cached"] for r in rows) / sum(r["turn2_prompt"] for r in rows), 4),
       "turn2_recomputed_mean": round(statistics.mean(r["turn2_recomputed"] for r in rows), 1),
       "turn1_unreused_mean": round(statistics.mean(r["turn1_prompt"] - min(r["turn2_cached"], r["turn1_prompt"])
                                                   for r in rows), 1)}
print("FOLLOWUP " + json.dumps(res), flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "a") as f:
    f.write(json.dumps(dict(res, rows=rows)) + "\n")
