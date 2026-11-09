# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Temperature-0 output fingerprint: same prompts, save full texts, to diff kernel-on vs kernel-off runs.
import json, sys, urllib.request
URL = "http://localhost:8000/v1/chat/completions"
P = ["Explain how a suspension bridge carries load.", "Write a Python function that merges two sorted lists.",
     "What is 17 * 23 + 144 / 12? Show steps.", "Summarise the causes of the First World War in 5 bullet points.",
     "Translate to German: The weather is nice today and we are going to the lake.",
     "Write a SQL query for the top 3 customers by revenue per country.", "Give me a haiku about GPUs.",
     "What are the differences between TCP and UDP?", "Solve: a train leaves at 14:05 and travels 180 km at 72 km/h. When does it arrive?",
     "Write a short product description for a reusable water bottle."]
out = []
for p in P:
    b = {"model": "qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": 300, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"})))
    out.append(r["choices"][0]["message"]["content"])
json.dump(out, open(sys.argv[1], "w"))
print("saved", len(out), "outputs to", sys.argv[1])
