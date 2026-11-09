# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Draft count by load (27-09-2026): closed-loop load curve with MTP acceptance per level.
#   bench_draftload.py TAG [LEVELS=1,2,4,8,16,32] [MAX_TOKENS=600]
# Like bench_load.py (shared ~500-token system prompt, temperature 0.7, the same 8 user prompts: prose, maths,
# code, email), but each of the c users sends requests back to back (closed loop), R = max(2, ceil(8/c)) each, so
# every level covers the whole prompt mix and the tail of a single wave weighs less. Per level: total tok/s,
# per-user decode tok/s (p50 / p5), time to first token (p50 / p95), peak running requests (traffic guard), and the
# spec-decode counter deltas: mean drafts per step (shows what a dynamic schedule actually used), acceptance length
# (1 + accepted / steps) and acceptance rate (accepted / drafted). Results -> $HK_WORK/draftload_results.jsonl.
import json, math, re, statistics, sys, threading, time, urllib.request

U = "http://localhost:8000"
TAG = sys.argv[1]
LEVELS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8,16,32").split(",")]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 600
SYSTEM = ("You are the assistant of a software company. Answer clearly and correctly, in the user's language. "
          "Use short paragraphs and lists where helpful. If a question is ambiguous, state your assumption. Never "
          "invent facts, prices or legal statements; say what you do not know. Keep answers focused on the question. "
          ) * 6
USER = ["Explain the difference between a VPN and a proxy for a small business.",
        "Write a 300-word story about a lighthouse keeper who finds a message in a bottle.",
        "A shop sells pens at 3 for 2.40 EUR and notebooks at 4.50 EUR. Anna buys 12 pens and 5 notebooks and pays "
        "with 50 EUR. Solve step by step: how much change does she get?",
        "Write a Python function that parses an ISO-8601 date and returns the weekday, with type hints and tests.",
        "Summarise the pros and cons of remote work for a team of 20 engineers.",
        "Draft a polite email to a customer explaining a two-day delivery delay.",
        "What are the main steps to set up a GDPR-compliant newsletter?",
        "Write a TypeScript Express route that validates a JSON body with zod and returns 201 or 400."]


def one(i, out, maxtok):
    body = {"model": "qwen3.8", "stream": True, "max_tokens": maxtok, "temperature": 0.7,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER[i % len(USER)]}]}
    t0 = time.time(); first = None; n = 0
    req = urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content") and first is None:
                first = time.time()
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
    t1 = time.time()
    out.append({"ttft": (first or t1) - t0, "tokens": n, "decode_tok_s": n / max(1e-6, t1 - (first or t0))})


def metrics():
    try:
        return urllib.request.urlopen(U + "/metrics", timeout=5).read().decode()
    except Exception:  # noqa: BLE001
        return ""


def counters():
    m = metrics()
    g = lambda pat: sum(float(x) for x in re.findall(pat, m))
    return {"draft": g(r"vllm:spec_decode_num_draft_tokens_total\{[^}]*\} ([0-9.e+]+)"),
            "acc": g(r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.e+]+)"),
            "steps": g(r"vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.e+]+)")}


def running():
    m = metrics()
    if not m:
        return -1.0
    return sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith("vllm:num_requests_running"))


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def closed_loop(c, rounds, maxtok):
    out = []
    def user(u):
        for r in range(rounds):
            one(u + r * c, out, maxtok)
    th = [threading.Thread(target=user, args=(u,)) for u in range(c)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


res = {"tag": TAG, "max_tokens": MAXTOK}
for c in LEVELS:
    closed_loop(c, 1, 32)                                   # warm-up: first use of this batch size, untimed
    rounds = max(2, math.ceil(8 / c))
    peak, stop = [0.0], [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], running()); time.sleep(0.5)
    w = threading.Thread(target=watch); w.start()
    c0 = counters(); t0 = time.time()
    out = closed_loop(c, rounds, MAXTOK)
    dt = time.time() - t0; c1 = counters()
    stop[0] = True; w.join()
    d = {k: c1[k] - c0[k] for k in c0}
    tt = [o["ttft"] for o in out]; ds = [o["decode_tok_s"] for o in out]
    res[str(c)] = {"total_tok_s": round(sum(o["tokens"] for o in out) / dt, 1),
                   "user_tok_s_p50": round(statistics.median(ds), 1), "user_tok_s_p5": round(pct(ds, 5), 1),
                   "ttft_p50": round(statistics.median(tt), 3), "ttft_p95": round(pct(tt, 95), 3),
                   "drafts_per_step": round(d["draft"] / d["steps"], 3) if d["steps"] else 0,
                   "accept_len": round(1 + d["acc"] / d["steps"], 3) if d["steps"] else 1,
                   "accept_rate": round(d["acc"] / d["draft"], 3) if d["draft"] else 0,
                   "requests": len(out), "secs": round(dt, 1), "peak_running": peak[0], "clean": 0 <= peak[0] <= c}
    print(c, json.dumps(res[str(c)]), flush=True)
print(json.dumps(res), flush=True)
open("$HK_WORK/draftload_results.jsonl", "a").write(json.dumps(res) + "\n")
