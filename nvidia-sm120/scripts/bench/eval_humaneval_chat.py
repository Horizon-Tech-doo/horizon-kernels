# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""HumanEval for a chat API (26-09-2026). lm-eval's humaneval_instruct pre-fills the answer, which a chat endpoint
can't do; here the model gets the prompt as a normal request, we take its ```python block, and run the official
check() in a sandboxed subprocess (10 s). Thinking off, temperature 0.
    eval_humaneval_chat.py TAG  -> ~/evals_humaneval/<TAG>.jsonl + one summary line"""
import json, os, re, subprocess, sys, tempfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "$HK_DATA/llmc-pkgs")
from datasets import load_dataset
TAG = sys.argv[1]; URL = "http://localhost:8000/v1/chat/completions"
ds = load_dataset("openai/openai_humaneval", split="test")
def ask(p):
    body = {"model": "qwen3.8", "temperature": 0, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Complete the following Python function. Return the full function "
                          "(with any imports it needs) in a single ```python code block.\n\n" + p}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
def run(ex, ans):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", ans, re.S); code = m[-1] if m else ans
    src = code + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
    try: return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: os.unlink(f.name)
exs = list(ds)
with ThreadPoolExecutor(16) as pool: answers = list(pool.map(lambda e: ask(e["prompt"]), exs))
res = [run(e, a) for e, a in zip(exs, answers)]
os.makedirs(os.path.expanduser("~/evals_humaneval"), exist_ok=True)
with open(os.path.expanduser(f"~/evals_humaneval/{TAG}.jsonl"), "w") as f:
    for e, a, r in zip(exs, answers, res): f.write(json.dumps({"task_id": e["task_id"], "pass": r, "answer": a}) + "\n")
print(json.dumps({"tag": TAG, "humaneval_pass@1": round(sum(res) / len(res), 4), "n": len(res)}))
