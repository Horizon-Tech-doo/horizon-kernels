# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K5 step 1: profile real decode steps of the live config through vLLM's own torch profiler.
# Run inside an ab_arm.sh arm whose VLLM_EXTRA_ARGS enables --profiler-config (see run_k5_profile.sh).
# Writes one trace per window (1 user, then 8 users) into the profiler dir; k5_trace.py summarises them.
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
U = "http://localhost:8001"
TOPICS = ["bridges", "printing", "railways", "radio", "vaccines", "maps", "clocks", "glass"]


def post(p, b=None, t=900):
    return urllib.request.urlopen(urllib.request.Request(U + p, json.dumps(b or {}).encode(),
                                                         {"Content-Type": "application/json"}), timeout=t).read()


def chat(i, n):
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": "Write a detailed essay about the history of "
                                           + TOPICS[i % 8] + "."}],
         "max_tokens": n, "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    s = time.time(); r = json.loads(post("/v1/chat/completions", b)); return r["usage"]["completion_tokens"], time.time() - s


chat(0, 64)
with ThreadPoolExecutor(8) as ex: list(ex.map(lambda i: chat(i, 64), range(8)))
for users in (1, 8):
    post("/start_profile"); time.sleep(0.5)
    s = time.time()
    with ThreadPoolExecutor(users) as ex: r = list(ex.map(lambda i: chat(i, 128 if users == 1 else 96), range(users)))
    d = time.time() - s
    post("/stop_profile")
    print(json.dumps({"users": users, "tokens": sum(t for t, _ in r), "s": round(d, 2),
                      "tok_s_under_profiler": round(sum(t for t, _ in r) / d, 1)}), flush=True)
    time.sleep(45)   # trace export happens in the engine after /stop_profile
