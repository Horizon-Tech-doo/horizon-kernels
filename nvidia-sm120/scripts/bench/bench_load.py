# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# High-load curve for capacity planning (26-09-2026): N concurrent streaming chats with a shared ~500-token system
# prompt, sampling at temperature 0.7 like real traffic. Per level: total tok/s, per-user decode tok/s and time to
# first token (p50 / p95). Results -> ~/load_results.jsonl.
#   bench_load.py TAG [LEVELS=1,4,8,16,32,64] [MAX_TOKENS=600]
# Each level: an untimed warm-up round (32 tokens), then the timed round; a traffic guard polls vLLM's
# num_requests_running and flags the level if more requests were in flight than this benchmark sent.
import json, statistics, sys, threading, time, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
TAG = sys.argv[1]
LEVELS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,4,8,16,32,64").split(",")]
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


def one(i, out, maxtok=None):
    body = {"model": "qwen3.8", "stream": True, "max_tokens": maxtok or MAXTOK, "temperature": 0.7,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER[i % len(USER)]}]}
    t0 = time.time(); first = None; n = 0
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
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


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def running():
    try:
        txt = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5).read().decode()
        return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith("vllm:num_requests_running"))
    except Exception:  # noqa: BLE001
        return -1.0


def run_level(c, maxtok=None):
    out, th = [], []
    for i in range(c):
        t = threading.Thread(target=one, args=(i, out, maxtok)); t.start(); th.append(t)
    return out, th


res = {"tag": TAG, "max_tokens": MAXTOK}
for c in LEVELS:
    _, th = run_level(c, 32)                      # warm-up: first use of this batch size, untimed
    for t in th:
        t.join()
    peak, stop = [0.0], [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], running()); time.sleep(0.5)
    w = threading.Thread(target=watch); w.start()
    out, th = [], []
    t0 = time.time()
    for i in range(c):
        t = threading.Thread(target=one, args=(i, out)); t.start(); th.append(t)
    for t in th:
        t.join()
    dt = time.time() - t0
    stop[0] = True; w.join()
    tt = [o["ttft"] for o in out]; ds = [o["decode_tok_s"] for o in out]
    res[str(c)] = {"total_tok_s": round(sum(o["tokens"] for o in out) / dt, 1),
                   "user_tok_s_p50": round(statistics.median(ds), 1), "user_tok_s_p5": round(pct(ds, 5), 1),
                   "ttft_p50": round(statistics.median(tt), 3), "ttft_p95": round(pct(tt, 95), 3),
                   "peak_running": peak[0], "clean": peak[0] <= c}
    print(c, res[str(c)], flush=True)
print(json.dumps(res), flush=True)
open("~/load_results.jsonl", "a").write(json.dumps(res) + "\n")
