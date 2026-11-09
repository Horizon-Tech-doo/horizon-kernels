# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Paired quality eval: same prompts and parsing as eval_quality.py, but every question is recorded, so two
# configurations can be compared question-by-question (McNemar) instead of by totals.
#   eval_quality2.py <tag> [n] [concurrency]     -> ~/eval2/<tag>.jsonl (one line per question) + summary line
# concurrency=1 gives deterministic runs; with >1, measure the noise floor by running one configuration twice.
import hashlib, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "$HK_DATA/llmc-pkgs")
from datasets import load_dataset

TAG = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 1
URL = "http://localhost:8000/v1/chat/completions"
OUT = os.path.expanduser(f"~/eval2/{TAG}.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def chat(msg, max_tokens=1536):
    b = json.dumps({"model": "qwen3.8", "messages": [{"role": "user", "content": msg}], "max_tokens": max_tokens,
                    "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, b, {"Content-Type": "application/json"}), timeout=900))
    return r["choices"][0]["message"]["content"] or ""


def gsm(item):
    i, ex = item
    out = chat(ex["question"] + "\nSolve step by step, then give the final answer on the last line as: Answer: <number>")
    gold = ex["answer"].split("####")[-1].strip().replace(",", "")
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", out) or re.findall(r"(-?[\d,]*\.?\d+)", out)
    pred = m[-1].replace(",", "").rstrip(".") if m else ""
    try:
        ok = float(pred) == float(gold)
    except ValueError:
        ok = False
    return {"task": "gsm8k", "idx": i, "pred": pred, "gold": gold, "correct": ok, "out_sha": hashlib.sha1(out.encode()).hexdigest()[:12]}


L = "ABCDEFGHIJ"


def mmlu(item):
    i, ex = item
    opts = "\n".join(f"{L[j]}. {o}" for j, o in enumerate(ex["options"]))
    out = chat(f"{ex['question']}\n\n{opts}\n\nThink briefly, then answer on the last line as: Answer: <letter>", 1024)
    m = re.findall(r"Answer:\s*\(?([A-J])\b", out)
    pred = m[-1] if m else ""
    return {"task": "mmlu_pro", "idx": i, "pred": pred, "gold": ex["answer"], "correct": pred == ex["answer"],
            "out_sha": hashlib.sha1(out.encode()).hexdigest()[:12]}


g = list(enumerate(load_dataset("openai/gsm8k", "main", split="test").select(range(min(N, 1319)))))
mp = list(enumerate(load_dataset("TIGER-Lab/MMLU-Pro", split="test").shuffle(seed=0).select(range(N))))
t = time.time()
with ThreadPoolExecutor(CONC) as ex, open(OUT, "w") as f:
    for fn, items in ((gsm, g), (mmlu, mp)):
        for r in ex.map(fn, items):
            r["tag"] = TAG
            f.write(json.dumps(r) + "\n"); f.flush()
rows = [json.loads(l) for l in open(OUT)]
acc = {task: sum(r["correct"] for r in rows if r["task"] == task) / max(1, sum(r["task"] == task for r in rows))
       for task in ("gsm8k", "mmlu_pro")}
res = {"tag": TAG, "n": N, "concurrency": CONC, **{k: round(v, 4) for k, v in acc.items()}, "eval_min": round((time.time() - t) / 60, 1)}
print(json.dumps(res))
open(os.path.expanduser("~/eval2/summary.jsonl"), "a").write(json.dumps(res) + "\n")
