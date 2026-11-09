# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Tool-call smoke test (27-09-2026, MoE track): a few chat requests WITH tools through the served endpoint (vLLM's
# --tool-call-parser qwen3_coder), thinking off, temperature 0. Checks that valid tool calls come back (known function
# name, JSON arguments with the required keys), that a parallel two-tool request yields two calls, that a no-tool
# question gets a plain answer, and that a tool result in a follow-up turn is used in the final answer.
#   tool_smoke.py TAG [URL]   -> one JSON line per case + a summary line
import json, sys, urllib.request

TAG = sys.argv[1]
URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8001/v1/chat/completions"
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Search internal documentation",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"]}}},
]
REQ = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}


def chat(messages, tools=TOOLS, tool_choice="auto"):
    body = {"model": "qwen3.8", "messages": messages, "tools": tools, "tool_choice": tool_choice, "temperature": 0,
            "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))["choices"][0]


def calls(ch):
    out = []
    for c in ch["message"].get("tool_calls") or []:
        f = c["function"]
        try:
            args = json.loads(f["arguments"]); valid = isinstance(args, dict)
        except Exception:  # noqa: BLE001
            args, valid = f["arguments"], False
        valid = valid and f["name"] in REQ and all(k in args for k in REQ[f["name"]])
        out.append({"name": f["name"], "args": args, "valid": valid, "id": c.get("id")})
    return out


res = []
cases = [("weather", "What's the weather in Zurich right now? Use celsius.", {"get_weather"}, 1),
         ("calc", "Use the calculator to compute 1234 * 5678.", {"calculator"}, 1),
         ("docs", "Search our docs for 'vLLM restart procedure' and return at most 3 results.", {"search_docs"}, 1),
         ("parallel", "I need both: the weather in Bern and the result of 17*23. Call the tools.",
          {"get_weather", "calculator"}, 2)]
for name, q, want, n in cases:
    try:
        ch = chat([{"role": "user", "content": q}])
        cs = calls(ch)
        ok = len(cs) >= n and all(c["valid"] for c in cs) and want <= {c["name"] for c in cs}
        res.append({"case": name, "ok": ok, "finish": ch.get("finish_reason"), "calls": cs,
                    "content": (ch["message"].get("content") or "")[:120]})
    except Exception as e:  # noqa: BLE001
        res.append({"case": name, "ok": False, "error": str(e)[:200]})
try:   # no tool needed
    ch = chat([{"role": "user", "content": "Say hello in German, one word."}])
    res.append({"case": "no_tool", "ok": not calls(ch) and bool(ch["message"].get("content")),
                "content": (ch["message"].get("content") or "")[:80]})
except Exception as e:  # noqa: BLE001
    res.append({"case": "no_tool", "ok": False, "error": str(e)[:200]})
try:   # follow-up with a tool result
    m = [{"role": "user", "content": "What's the weather in Zurich? Use celsius."}]
    ch = chat(m); cs = calls(ch)
    if cs:
        m.append({"role": "assistant", "content": ch["message"].get("content") or "",
                  "tool_calls": [{"id": cs[0]["id"] or "call_0", "type": "function",
                                  "function": {"name": cs[0]["name"], "arguments": json.dumps(cs[0]["args"])}}]})
        m.append({"role": "tool", "tool_call_id": cs[0]["id"] or "call_0",
                  "content": json.dumps({"city": "Zurich", "temperature_c": 17.5, "condition": "light rain"})})
        ch2 = chat(m); txt = ch2["message"].get("content") or ""
        res.append({"case": "follow_up", "ok": "17" in txt and not calls(ch2), "content": txt[:160]})
    else:
        res.append({"case": "follow_up", "ok": False, "error": "no tool call in turn 1"})
except Exception as e:  # noqa: BLE001
    res.append({"case": "follow_up", "ok": False, "error": str(e)[:200]})
for r in res:
    print(json.dumps({"tag": TAG, **r}), flush=True)
print(json.dumps({"tag": TAG, "tool_smoke_passed": sum(r["ok"] for r in res), "of": len(res)}), flush=True)
