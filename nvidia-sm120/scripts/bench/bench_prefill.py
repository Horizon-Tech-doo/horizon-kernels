# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Prefill benchmark: time to first token (TTFT) and prefill tok/s for several prompt lengths.
# Every request starts with a unique random prefix so vLLM's prefix cache cannot skip work. Median of 3 runs.
import json, random, statistics, sys, time, urllib.request

URL = "http://localhost:8000/v1/chat/completions"
WORDS = ("azure rust tenant subscription pipeline kernel policy firewall cluster gateway storage queue model "
         "tensor memory bandwidth latency throughput scheduler token prompt decode cache layer").split()
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
LENGTHS = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else "1000,4000,16000,64000".split(","))]


def ttft(n_words):
    r = random.Random()
    doc = f"[{r.getrandbits(64):x}] " + " ".join(r.choice(WORDS) for _ in range(n_words))
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": "Summarize in one sentence: " + doc}],
            "max_tokens": 8, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); first = None; ptok = 0
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            ptok = d["usage"]["prompt_tokens"]
        if first is None and d.get("choices") and d["choices"][0]["delta"].get("content") is not None:
            first = time.time() - t
    return ptok, first


ttft(200)  # warm-up
res = {"tag": TAG}
for n in LENGTHS:
    runs = [ttft(n) for _ in range(3)]
    ptok = runs[0][0]; t = statistics.median(r[1] for r in runs)
    res[f"{ptok}_tok"] = {"ttft_s": round(t, 3), "prefill_tok_s": round(ptok / t)}
    print(json.dumps({"prompt_tokens": ptok, "ttft_s": round(t, 3), "prefill_tok_s": round(ptok / t)}), flush=True)
print(json.dumps(res)); open("~/prefill_results.jsonl", "a").write(json.dumps(res) + "\n")
