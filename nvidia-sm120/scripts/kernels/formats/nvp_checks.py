# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# NVP arm checks: determinism (8 prompts, temperature 0, sequential, twice) and a tool-call smoke (qwen3_coder parser).
import json, urllib.request
URL = "http://localhost:8000/v1/chat/completions"
def chat(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())
P = ["Write a Python function that returns the n-th Fibonacci number iteratively.",
     "Explain in three sentences why the sky is blue.", "What is 1234 * 5678? Show the steps.",
     "Translate to German: The server restarts every night at two o'clock.",
     "List five prime numbers greater than 100.", "Write a haiku about GPUs.",
     "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
     "Summarise the plot of Hamlet in two sentences."]
same = 0
for p in P:
    outs = [chat({"model": "qwen3.8", "messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0,
                  "chat_template_kwargs": {"enable_thinking": False}})["choices"][0]["message"]["content"] for _ in range(2)]
    same += outs[0] == outs[1]
print(f"DET {same}/{len(P)} identical")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
          "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}]
ok = 0
for i in range(3):
    r = chat({"model": "qwen3.8", "messages": [{"role": "user", "content": "What's the weather in Sarajevo right now, in celsius? Use the tool."}],
              "tools": tools, "tool_choice": "auto", "max_tokens": 512, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}})
    tc = r["choices"][0]["message"].get("tool_calls") or []
    good = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "city" in json.loads(tc[0]["function"]["arguments"])
    ok += good
    if i == 0:
        print(json.dumps({"tool_calls": tc, "finish": r["choices"][0]["finish_reason"]})[:400])
print(f"TOOL {ok}/3 valid tool calls")
