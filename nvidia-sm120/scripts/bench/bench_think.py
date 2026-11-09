# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Thinking-mode speed benchmark (26-09-2026): every earlier speed test ran with thinking OFF. Same kinds of prompts
# with thinking ON (reasoning_effort high for code), 1 user per category and 8 users together; tok/s over all
# completion tokens (reasoning + answer) and speculative acceptance from vLLM's own counters (delta around each run).
#   bench_think.py TAG            -> ~/think_results.jsonl
import json, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"
TAG = sys.argv[1]
P = {"prose": ["Write a 300-word essay on why railways shaped modern Europe."],
     "maths": ["A train leaves at 08:15 at 84 km/h; a second leaves the same station at 09:00 at 112 km/h on the same "
               "track. When and where does the second catch the first?",
               "Find all real x with 2x^2 - 7x + 3 = 0, then compute the sum of their squares."],
     "code": ["Write a Python function that parses an ISO-8601 date string and returns the weekday name, with type "
              "hints and 5 pytest tests.",
              "Here is a buggy function:\n```python\ndef avg(xs):\n    return sum(xs) / len(xs) if xs else None\n```\n"
              "Make it handle generators, raise ValueError on empty input, and add tests."]}


def metrics():
    txt = urllib.request.urlopen("http://localhost:8000/metrics", timeout=10).read().decode()
    g = lambda pat: sum(float(m) for m in re.findall(pat + r"\{[^}]*\} ([0-9.e+]+)", txt))
    return {"drafts": g(r"vllm:spec_decode_num_drafts_total"), "accepted": g(r"vllm:spec_decode_num_accepted_tokens_total")}


def req(msg, kind):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": msg}], "max_tokens": 4096,
            "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": True}}
    if kind == "code":
        body["reasoning_effort"] = "high"
    s = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"}), timeout=1800))
    return r["usage"]["completion_tokens"], time.time() - s


res = {"tag": TAG, "thinking": True}
for kind, ps in P.items():
    m0 = metrics(); toks = dur = 0
    for p in ps:
        t, d = req(p, kind); toks += t; dur += d
    m1 = metrics(); dd = max(1.0, m1["drafts"] - m0["drafts"])
    res[kind] = {"tok_s": round(toks / dur, 1), "tokens": toks, "mean_accept_len": round(1 + (m1["accepted"] - m0["accepted"]) / dd, 3)}
    print(kind, res[kind], flush=True)
allp = [(p, k) for k, ps in P.items() for p in ps]
m0 = metrics(); s = time.time()
with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(lambda a: req(*a), (allp * 2)[:8]))
m1 = metrics(); dd = max(1.0, m1["drafts"] - m0["drafts"])
res["8users"] = {"total_tok_s": round(sum(t for t, _ in out) / (time.time() - s), 1),
                 "mean_accept_len": round(1 + (m1["accepted"] - m0["accepted"]) / dd, 3)}
print("8users", res["8users"], flush=True)
print(json.dumps(res), flush=True)
open("~/think_results.jsonl", "a").write(json.dumps(res) + "\n")
