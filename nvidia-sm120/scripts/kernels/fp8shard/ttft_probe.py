# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# ttft_probe.py TAG SIZES [REPS] (fp8fix, 27-09-2026): cold time-to-first-token of long prompts (unique nonce first,
# so the prefix cache never helps), streaming chat request on :8000, thinking off. Prompt size calibrated with
# /tokenize. Prints one JSON line {tag, size: {median_s, all_s, prompt_tokens}}.
import json, random, statistics, sys, time, urllib.request
TAG, SIZES = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
BASE = "http://localhost:8000"
W = "azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue ledger".split()
R = random.Random()


def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                                                   {"Content-Type": "application/json"}), timeout=900))


try:
    tpw = post("/tokenize", {"model": "qwen3.8", "prompt": " ".join(R.choice(W) for _ in range(4000))})["count"] / 4000
except Exception:  # noqa: BLE001  (a proxy without /tokenize)
    tpw = 1.25
res = {"tag": TAG}
for size in SIZES:
    ts, pt = [], 0
    for _ in range(REPS):
        doc = " ".join(R.choice(W) for _ in range(int((size - 60) / tpw)))
        body = {"model": "qwen3.8", "messages": [{"role": "user", "content": f"[{R.random()}] Summarize in one sentence: {doc}"}],
                "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0, first = time.time(), None
        for line in urllib.request.urlopen(req, timeout=900):
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                pt = d["usage"]["prompt_tokens"]
            if first is None and d.get("choices") and d["choices"][0]["delta"].get("content") is not None:
                first = time.time() - t0
        ts.append(round(first if first is not None else time.time() - t0, 3))
    res[str(size)] = {"median_s": statistics.median(ts), "all_s": ts, "prompt_tokens": pt}
print("TTFT " + json.dumps(res), flush=True)
