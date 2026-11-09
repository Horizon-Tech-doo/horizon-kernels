# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# At the char where kernel-on/off outputs diverge, how close were the model's top-2 choices?
import json, urllib.request
URL = "http://localhost:8000/v1/chat/completions"
P = json.load(open("$HK_WORK/outcheck_prompts.json"))
on = json.load(open("$HK_WORK/out_k1_on.json")); off = json.load(open("$HK_WORK/out_k1_off.json"))
for i, (a, b) in enumerate(zip(on, off)):
    if a == b: continue
    b_ = {"model": "qwen3.8", "messages": [{"role": "user", "content": P[i]}], "max_tokens": 300, "temperature": 0,
          "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b_).encode(), {"Content-Type": "application/json"})))
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    pos = 0
    for t in r["choices"][0]["logprobs"]["content"]:
        if pos + len(t["token"]) > k:
            tl = t["top_logprobs"]
            print(f"prompt {i}: split at char {k}; top-2 = {tl[0]['token']!r} {tl[0]['logprob']:.3f} vs {tl[1]['token']!r} {tl[1]['logprob']:.3f}  gap={tl[0]['logprob']-tl[1]['logprob']:.3f}")
            break
        pos += len(t["token"])
