"""Client checks against a local vLLM server: python serve_check.py <port> <served name> [--needle].  Real generations, not just /v1/models."""
import json, sys, time, urllib.request
port, name = sys.argv[1], sys.argv[2]
U = f"http://127.0.0.1:{port}/v1"
def post(body, timeout=900):
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(U + "/chat/completions", json.dumps({"model": name, "temperature": 0, "seed": 0, **body}).encode(),
                               {"Content-Type": "application/json"}), timeout=timeout)
    o = json.load(r); o["_s"] = round(time.time() - t, 2); return o
models = json.load(urllib.request.urlopen(U + "/models", timeout=30))
print("MODELS", [(m["id"], m.get("max_model_len")) for m in models["data"]], flush=True)
q = [{"role": "user", "content": "What is 17*23? Answer with the number only."}]
o = post({"messages": q, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}); m = o["choices"][0]["message"]
print("THINK_OFF", repr(m["content"]), "reasoning", repr(m.get("reasoning_content") or m.get("reasoning")), o["usage"], flush=True)
assert "391" in (m["content"] or "")
o = post({"messages": q, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": True}}); m = o["choices"][0]["message"]
print("THINK_ON", repr(m["content"]), "reasoning_chars", len(m.get("reasoning_content") or m.get("reasoning") or ""), o["usage"], flush=True)
assert "391" in (m["content"] or "") and (m.get("reasoning_content") or m.get("reasoning"))
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
o = post({"messages": [{"role": "user", "content": "What is the weather in Zurich? Use the tool."}], "tools": tools, "tool_choice": "auto", "max_tokens": 256,
          "chat_template_kwargs": {"enable_thinking": False}}); m = o["choices"][0]["message"]
print("TOOL", json.dumps(m.get("tool_calls"))[:300], flush=True)
assert m.get("tool_calls") and "Zurich" in m["tool_calls"][0]["function"]["arguments"]
if "--needle" in sys.argv:
    hay = "The harbour council recorded the broken fence before the market opened. " * 1700
    p = hay + "The access code for the amber falcon vault is 7314592. " + hay + "\nWhat is the access code for the amber falcon vault? Reply with the number only."
    o = post({"messages": [{"role": "user", "content": p}], "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}})
    print("LONG_PROMPT", o["usage"], repr(o["choices"][0]["message"]["content"]), "sec", o["_s"], flush=True)
    assert o["usage"]["prompt_tokens"] > 40000
print("SERVE_OK", name, flush=True)
