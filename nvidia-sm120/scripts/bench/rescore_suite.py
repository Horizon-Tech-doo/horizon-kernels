# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Re-score the lm-eval suite samples where the harness doesn't fit a chat API (26-09-2026):
- MBPP: extract the ```python block from the full answer and run the dataset's asserts (sandboxed subprocess, 10 s).
- AIME / MATH-500: take the last \\boxed{...} and compare with math_verify.
- also reads lm-eval's own results json for every task. Prints one table.
    rescore_suite.py <suite dir>"""
import glob, json, os, re, subprocess, sys, tempfile
sys.path.insert(0, "$HK_DATA/eval-pkgs")
from math_verify import parse, verify

D = sys.argv[1]
def samples(run, task):
    fs = sorted(glob.glob(f"{D}/{run}/*/samples_{task}_*.jsonl"))
    return [json.loads(l) for l in open(fs[-1])] if fs else []
def text(r):
    x = r.get("resps")
    while isinstance(x, list): x = x[0] if x else ""
    return x or ""
def code_block(t):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    return (m[-1] if m else t).strip()
def run_py(src):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(src)
    try: return subprocess.run([sys.executable, f.name], capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: os.unlink(f.name)
def last_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0: return None
    j, depth = i + 7, 1
    while j < len(t) and depth:
        depth += {"{": 1, "}": -1}.get(t[j], 0); j += 1
    return t[i + 7:j - 1]
def math_acc(rows):
    ok = 0
    for r in rows:
        b = last_boxed(text(r)); gold = str(r["doc"].get("answer", r.get("target")))
        ok += bool(b is not None and verify(parse("$" + gold + "$"), parse("$" + b + "$")))
    return ok, len(rows)

out = []
for run in ("nothink", "think"):
    for task in ("aime25", "minerva_math500"):
        rows = samples(run, task)
        if rows: ok, n = math_acc(rows); out.append((run, task, "boxed + math_verify (rescored)", ok, n))
rows = samples("nothink", "mbpp_instruct")
if rows:
    ok = sum(run_py(code_block(text(r)) + "\n\n" + "\n".join(r["doc"].get("test_setup_code", "") and [r["doc"]["test_setup_code"]] or []) + "\n" + "\n".join(r["doc"]["test_list"])) for r in rows)
    out.append(("nothink", "mbpp_instruct", "pass@1 (rescored)", ok, len(rows)))
for run, task, metric, ok, n in out:
    print(f"{run:8s} {task:18s} {metric:34s} {ok:4d}/{n:<4d} = {100*ok/n:5.1f}%")
for f in sorted(glob.glob(f"{D}/*/*/results_*.json")):
    run = f.split("/")[-3]
    if run.startswith("smoke"): continue
    for task, m in json.load(open(f))["results"].items():
        for k, v in m.items():
            if isinstance(v, float) and "stderr" not in k and k != "alias":
                print(f"{run:8s} {task:18s} {k:34s} {100*v:5.1f}%  (lm-eval)")
