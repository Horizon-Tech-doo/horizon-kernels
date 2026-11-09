# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Long-generation quality probe for a 16-bit GDN recurrent state (29-09-2026). ONE command, one server, JSON out.

usage: gdn_longgen_check.py URL OUT.json [--budget-s 870] [--concurrency 64] [--model qwen3.8]
  URL: the server base (http://localhost:8001) or its /v1 (both accepted).

Why this probe: a 16-bit state is rounded after every DECODE step (1-4 tokens), so its risk shows in long generations
(thinking), not in prompt log-probs (prefill rounds once per chunk). Thinking-on maths is the most sensitive cheap
signal we have: long outputs, a checkable final answer, and drift shows up as wrong answers, longer outputs and more
runs hitting the token limit.
Workload (fixed, identical on every arm): AIME 2025 (30 problems, max 12,288 tokens) first, then MATH-500 problems
0-99 (max 6,144 tokens); thinking on (reasoning_effort high), temperature 0.6, top_p 0.95, seed = problem index;
eval_math_chat.py's prompt and grading (last \\boxed{}, math_verify). All 130 are submitted at once through a pool of
--concurrency workers. A hard wall-clock budget (default 870 s < 15 min) aborts what is
still running: those items are recorded as incomplete and the result is marked complete=false (the compare step then
fails, never passes on partial data).
Output: {"complete", "budget_s", "wall_s", "summary": {set: {n, correct, hit_limit, gen_tokens}, "all": ...},
         "items": [{"set", "i", "ok", "finish", "gen_tokens", "secs"}]}; also one summary JSON line on stdout.
Pass rule vs the P arm: gdn_longgen_compare.py.
"""
import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("out")
ap.add_argument("--budget-s", type=float, default=870.0)
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--model", default="qwen3.8")
a = ap.parse_args()
BASE = a.url.rstrip("/")
if not BASE.endswith("/v1"):
    BASE += "/v1"
T0 = time.time()
DEADLINE = T0 + a.budget_s
sys.path.insert(0, "$HK_DATA/eval-pkgs")
sys.path.insert(1, "$HK_DATA/llmc-pkgs")
from datasets import load_dataset  # noqa: E402
from math_verify import parse, verify  # noqa: E402

ITEMS = [("aime25", i, r["problem"], str(r["answer"]), 12288)
         for i, r in enumerate(load_dataset("math-ai/aime25", split="test"))]
ITEMS += [("math500", i, r["problem"], r["answer"], 6144)
          for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test")) if i < 100]


def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(t) and d:
        d += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return t[i + 7:j - 1]


def ask(it):
    s, i, q, gold, mx = it
    t = time.time()
    left = DEADLINE - t
    rec = {"set": s, "i": i, "ok": False, "finish": None, "gen_tokens": 0, "secs": 0.0}
    if left <= 5:
        rec["finish"] = "not_started"
        return rec
    body = {"model": a.model, "messages": [{"role": "user", "content": q + "\n\nPlease reason step by step, and put "
            "your final answer within \\boxed{}."}], "temperature": 0.6, "top_p": 0.95, "max_tokens": mx,
            "reasoning_effort": "high", "seed": i}
    try:
        req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        m = json.load(urllib.request.urlopen(req, timeout=left))
        c = m["choices"][0]
        ans = c["message"].get("content") or ""
        rec.update(finish=c.get("finish_reason"), gen_tokens=m["usage"]["completion_tokens"], pred=boxed(ans))
    except Exception as e:  # noqa: BLE001  (timeout at the budget, or a server error)
        rec["finish"] = "incomplete"
        rec["error"] = str(e)[:160]
    rec["secs"] = round(time.time() - t, 1)
    return rec


with ThreadPoolExecutor(a.concurrency) as pool:
    items = list(pool.map(ask, ITEMS))
# grading in the main thread: math_verify's parse() uses signal.alarm and refuses to run in worker threads
for it, r in zip(ITEMS, items):
    b = r.pop("pred", None)
    r["ok"] = bool(b is not None and verify(parse("$" + it[3] + "$"), parse("$" + b + "$")))
summ = {}
for r in items:
    for k in (r["set"], "all"):
        d = summ.setdefault(k, {"n": 0, "correct": 0, "hit_limit": 0, "gen_tokens": 0, "incomplete": 0})
        d["n"] += 1
        d["correct"] += r["ok"]
        d["hit_limit"] += r["finish"] == "length"
        d["gen_tokens"] += r["gen_tokens"]
        d["incomplete"] += r["finish"] in ("incomplete", "not_started")
out = {"complete": summ["all"]["incomplete"] == 0, "budget_s": a.budget_s, "wall_s": round(time.time() - T0, 1),
       "url": BASE, "summary": summ, "items": items}
json.dump(out, open(a.out, "w"))
print(json.dumps({k: v for k, v in out.items() if k != "items"}), flush=True)
sys.exit(0 if out["complete"] else 3)
