# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Maths eval for a chat API (26-09-2026): lm-eval's minerva/aime tasks use few-shot stop words ("Problem:") that cut
thinking-mode answers mid-reasoning, and expect \\boxed{} without asking for it. Here: plain chat prompt that asks for
\\boxed{}, no stop words, graded with math_verify (last \\boxed{}).
  thinking off: temperature 0, 8k tokens.  thinking on: reasoning_effort high, T 0.6 / top_p 0.95, 32k tokens.
    eval_math_chat.py TAG            -> ~/evals_math/<TAG>-<set>-<mode>.jsonl + summary lines"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "$HK_DATA/eval-pkgs"); sys.path.insert(1, "$HK_DATA/llmc-pkgs")
from datasets import load_dataset
from math_verify import parse, verify
TAG = sys.argv[1]; URL = "http://localhost:8000/v1/chat/completions"
SETS = {"math500": [(r["problem"], r["answer"]) for r in load_dataset("HuggingFaceH4/MATH-500", split="test")],
        "aime25": [(r["problem"], str(r["answer"])) for r in load_dataset("math-ai/aime25", split="test")]}
def ask(q, think):
    body = {"model": "qwen3.8", "messages": [{"role": "user", "content": q + "\n\nPlease reason step by step, and put "
            "your final answer within \\boxed{}."}]}
    body.update({"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "reasoning_effort": "high", "seed": 0} if think
                else {"temperature": 0, "max_tokens": 8192, "chat_template_kwargs": {"enable_thinking": False}})
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=7200))["choices"][0]
    return m["message"].get("content") or "", m.get("finish_reason")
def boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j, d = i + 7, 1
    while j < len(t) and d: d += {"{": 1, "}": -1}.get(t[j], 0); j += 1
    return t[i + 7:j - 1]
os.makedirs(os.path.expanduser("~/evals_math"), exist_ok=True)
for name, items in SETS.items():
    for think in (False, True):
        with ThreadPoolExecutor(32) as pool: outs = list(pool.map(lambda it: ask(it[0], think), items))
        mode = "think" if think else "nothink"; ok = 0; cut = 0
        with open(os.path.expanduser(f"~/evals_math/{TAG}-{name}-{mode}.jsonl"), "w") as f:
            for (q, gold), (ans, fin) in zip(items, outs):
                b = boxed(ans); good = bool(b is not None and verify(parse("$" + gold + "$"), parse("$" + b + "$")))
                ok += good; cut += fin == "length"
                f.write(json.dumps({"gold": gold, "pred": b, "ok": good, "finish": fin, "answer": ans}) + "\n")
        print(json.dumps({"tag": TAG, "set": name, "mode": mode, "acc": round(ok / len(items), 4), "n": len(items),
                          "hit_token_limit": cut}), flush=True)
print("MATH_CHAT_DONE")
