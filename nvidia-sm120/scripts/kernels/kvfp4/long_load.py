# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""long_load.py TAG [N=8] [CTX=32000] [MAXTOK=400]  (29-09-2026, FP4 KV A/B)
Long-context load: N concurrent streaming conversations, each with its own ~CTX-token context (a nonce first, so the
prefix cache never helps), thinking off, temperature 0, MAXTOK output tokens. Reports per request TTFT and decode tok/s,
their medians / p95, and the aggregate decode throughput while all N streams are decoding (from the per-token arrival
times). Prompt size is calibrated with the server's /tokenize. -> $HK_WORK/kvfp4/longload-TAG.json (+ one JSON line)."""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

BASE = "http://localhost:8000"
TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 32000
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 400
WORDS = ("azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue ledger invoice "
         "harbour lantern meadow quartz violet thunder orchard compass saddle glacier ember falcon").split()


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


R = random.Random(7)
tpw = post("/tokenize", {"model": "qwen3.8", "prompt": " ".join(R.choice(WORDS) for _ in range(4000))})["count"] / 4000
nwords = int((CTX - 200) / tpw)
res = [None] * N


def one(i):
    rr = random.Random(1000 + i)
    doc = " ".join(rr.choice(WORDS) for _ in range(nwords))
    msg = f"Session {rr.random()}. Here is a long system log:\n{doc}\nNow write a detailed 300 word essay about " \
          f"operating a secure cloud platform."
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": msg}], "max_tokens": MAXTOK,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first, times, ptok, ctok = None, [], 0, 0
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            ptok, ctok = d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            now = time.time()
            first = first or now
            times.append(now)
    end = time.time()
    res[i] = {"prompt_tokens": ptok, "completion_tokens": ctok, "ttft_s": round(first - t0, 3),
              "decode_tok_s": round((ctok - 1) / max(1e-6, end - first), 2), "t_first": first, "t_end": end,
              "chunk_times": times}


th = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in th:
    t.start()
for t in th:
    t.join()
lo = max(r["t_first"] for r in res)      # window where every stream has started decoding
hi = min(r["t_end"] for r in res)
chunks = sum(sum(1 for x in r["chunk_times"] if lo <= x <= hi) for r in res)
ttft = sorted(r["ttft_s"] for r in res)
dec = sorted(r["decode_tok_s"] for r in res)
out = {"tag": TAG, "n": N, "ctx_target": CTX, "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in res),
       "ttft_p50_s": statistics.median(ttft), "ttft_p95_s": ttft[min(N - 1, int(0.95 * N))],
       "decode_tok_s_per_user_p50": statistics.median(dec), "decode_tok_s_per_user_min": dec[0],
       "overlap_window_s": round(max(0.0, hi - lo), 2),
       "overlap_stream_chunks_per_s": round(chunks / (hi - lo), 1) if hi > lo else None,
       "total_completion_tokens": sum(r["completion_tokens"] for r in res)}
for r in res:
    r.pop("chunk_times")
out["requests"] = res
os.makedirs(os.path.expanduser("$HK_WORK/kvfp4"), exist_ok=True)
json.dump(out, open(os.path.expanduser(f"$HK_WORK/kvfp4/longload-{TAG}.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "requests"}), flush=True)
