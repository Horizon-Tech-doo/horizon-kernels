# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Exact MTP acceptance per workload: delta of vLLM's cumulative spec-decode counters around identical requests.
import json, re, sys, threading, time, urllib.request
U = "http://localhost:8000"
W = {
 "prose": ["Write a detailed essay about the history of printing.", "Describe a walk through an autumn forest.",
           "Explain why the Roman Empire declined.", "Write a short story about a lighthouse keeper."],
 "maths": ["Solve step by step: 3x + 7 = 2(x - 4) + 19.", "A shop discounts 15% then adds 20% VAT on 240 EUR. Final price? Show steps.",
           "Compute the sum of the first 50 odd numbers, explaining each step.", "Find the derivative of x^3 * ln(x) step by step."],
 "code":  ["Write a Python class implementing an LRU cache with tests.", "Write a TypeScript function that debounces another function, with JSDoc.",
           "Write a Rust function that parses a CSV line handling quoted fields.", "Write a SQL schema for a library with books, members and loans."]}
def counters():
    m = urllib.request.urlopen(U + "/metrics").read().decode()
    g = lambda pat: sum(float(x) for x in re.findall(pat, m))
    return {"draft": g(r"vllm:spec_decode_num_draft_tokens_total\{[^}]*\} ([0-9.e+]+)"),
            "acc": g(r"vllm:spec_decode_num_accepted_tokens_total\{[^}]*\} ([0-9.e+]+)"),
            "drafts": g(r"vllm:spec_decode_num_drafts_total\{[^}]*\} ([0-9.e+]+)"),
            **{f"p{i}": g(r'accepted_tokens_per_pos_total\{[^}]*position="%d"\} ([0-9.e+]+)' % i) for i in range(16)}}
def running():
    m = urllib.request.urlopen(U + "/metrics").read().decode()
    g = lambda name: sum(float(x) for x in re.findall(r"vllm:%s\{[^}]*\} ([0-9.e+]+)" % name, m))
    return g("num_requests_running") + g("num_requests_waiting")


class Watch(threading.Thread):
    """Polls the server while a window runs; our own requests are sequential, so >1 in flight means other requests."""
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0.0; self.stop = False
    def run(self):
        while not self.stop:
            try: self.peak = max(self.peak, running())
            except Exception: pass
            time.sleep(0.25)


def chat(p):
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": 500, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    s = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(U + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})))
    return r["usage"]["completion_tokens"], time.time() - s
tag = sys.argv[1]; chat("warm up"); res = {"tag": tag}
for cat, ps in W.items():
    assert running() == 0, "server busy before the window: other requests, measurement refused"
    w = Watch(); w.start()
    c0 = counters(); toks = secs = 0
    for p in ps:
        t, d = chat(p); toks += t; secs += d
    c1 = counters(); w.stop = True; w.join()
    d = {k: c1[k] - c0[k] for k in c0}
    k = int(round(d["draft"] / d["drafts"])) if d["drafts"] else 0
    res[cat] = {"tok_s": round(toks / secs, 1), "mean_accept_len": round(1 + d["acc"] / d["drafts"], 3),
                "per_pos": [round(d[f"p{i}"] / d["drafts"], 3) for i in range(k)],
                "valid": w.peak <= 1, "peak_in_flight": w.peak}
print(json.dumps(res)); open("$HK_WORK/acc_results.jsonl", "a").write(json.dumps(res) + "\n")
