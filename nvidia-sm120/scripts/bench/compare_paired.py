#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Question-by-question comparison of two models' eval runs (27-09-2026), e.g. production (A) vs
nvidia/Qwen3.8-27B-NVFP4 (B). For every benchmark: accuracy of both, paired wins (B right, A wrong), losses (B wrong,
A right) and the exact two-sided McNemar p on the discordant pairs.
    compare_paired.py SUITE_A SUITE_B TAG_A TAG_B [--json OUT]
  SUITE_*: lm-eval --output_path dirs holding nothink/<model>/samples_<task>_*.jsonl (gsm8k_cot_llama, ifeval,
           mbpp_instruct; MBPP is re-scored from the ```python block like rescore_suite.py)
  TAG_*:   tags of eval_humaneval_chat.py (~/evals_humaneval/<TAG>.jsonl) and eval_math_chat*.py
           (~/evals_math/<TAG>-<set>-<mode>.jsonl)"""
import argparse, glob, json, math, os, re, subprocess, sys, tempfile

ap = argparse.ArgumentParser()
ap.add_argument("suite_a"); ap.add_argument("suite_b"); ap.add_argument("tag_a"); ap.add_argument("tag_b")
ap.add_argument("--json", default="")
A = ap.parse_args()


def samples(d, task):
    fs = sorted(glob.glob(f"{d}/nothink/*/samples_{task}_*.jsonl"))
    return [json.loads(l) for l in open(fs[-1])] if fs else []


def text(r):
    x = r.get("resps")
    while isinstance(x, list):
        x = x[0] if x else ""
    return x or ""


def run_py(src):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(f.name)


def code_block(t):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    return (m[-1] if m else t).strip()


def lm(d, task, key, flt=None):
    out = {}
    for r in samples(d, task):
        if flt and r.get("filter") != flt:
            continue
        if key == "mbpp":
            setup = r["doc"].get("test_setup_code", "")
            out[str(r["doc_id"])] = run_py(code_block(text(r)) + "\n\n" + (setup + "\n" if setup else "")
                                           + "\n".join(r["doc"]["test_list"]))
        else:
            v = r[key]
            out[str(r["doc_id"])] = (float(v) > 0.5) if not isinstance(v, bool) else v
    return out


def jl(path, key, idkey=None):
    if not os.path.exists(path):
        return {}
    rows = [json.loads(l) for l in open(path)]
    return {(r[idkey] if idkey else str(i)): bool(r[key]) for i, r in enumerate(rows)}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


H = os.path.expanduser("~")
BENCH = [
    ("GSM8K (lm-eval, strict)", lambda d, t: lm(d, "gsm8k_cot_llama", "exact_match", "strict-match"), "suite"),
    ("GSM8K (lm-eval, flexible)", lambda d, t: lm(d, "gsm8k_cot_llama", "exact_match", "flexible-extract"), "suite"),
    ("MBPP (rescored)", lambda d, t: lm(d, "mbpp_instruct", "mbpp"), "suite"),
    ("IFEval prompt strict", lambda d, t: lm(d, "ifeval", "prompt_level_strict_acc"), "suite"),
    ("IFEval prompt loose", lambda d, t: lm(d, "ifeval", "prompt_level_loose_acc"), "suite"),
    ("HumanEval (chat)", lambda d, t: jl(f"{H}/evals_humaneval/{t}.jsonl", "pass", "task_id"), "tag"),
    ("MATH-500 thinking off", lambda d, t: jl(f"{H}/evals_math/{t}-math500-nothink.jsonl", "ok"), "tag"),
    ("AIME 2025 thinking off", lambda d, t: jl(f"{H}/evals_math/{t}-aime25-nothink.jsonl", "ok"), "tag"),
    ("AIME 2025 thinking on", lambda d, t: jl(f"{H}/evals_math/{t}-aime25-think.jsonl", "ok"), "tag"),
    ("MATH-500 thinking on", lambda d, t: jl(f"{H}/evals_math/{t}-math500-think.jsonl", "ok"), "tag"),
]
res = []
print(f"| Benchmark | A: {A.tag_a} | B: {A.tag_b} | Paired n | B wins | B losses | McNemar p |")
print("|---|---|---|---|---|---|---|")
for name, fn, _ in BENCH:
    a, b = fn(A.suite_a, A.tag_a), fn(A.suite_b, A.tag_b)
    if not a or not b:
        print(f"| {name} | {'%.1f%%' % (100 * sum(a.values()) / len(a)) if a else '–'} | "
              f"{'%.1f%%' % (100 * sum(b.values()) / len(b)) if b else '–'} | – | – | – | – |")
        continue
    keys = sorted(set(a) & set(b))
    wins = sum(1 for k in keys if b[k] and not a[k]); loss = sum(1 for k in keys if a[k] and not b[k])
    row = {"bench": name, "acc_a": sum(a.values()) / len(a), "n_a": len(a), "acc_b": sum(b.values()) / len(b),
           "n_b": len(b), "paired": len(keys), "b_wins": wins, "b_losses": loss, "p": mcnemar(wins, loss)}
    res.append(row)
    print(f"| {name} | {100 * row['acc_a']:.1f}% ({row['n_a']}) | {100 * row['acc_b']:.1f}% ({row['n_b']}) | "
          f"{len(keys)} | {wins} | {loss} | {row['p']:.3f} |")
if A.json:
    json.dump(res, open(A.json, "w"), indent=1)
