# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Paired comparison of two eval_quality2 runs: McNemar's exact test on the questions where they disagree.
#   compare_eval2.py <tagA> <tagB> [tagA_repeat]
# With a third tag (config A run again), also reports A-vs-A disagreement: the run-to-run noise floor.
import json, math, os, sys


def load(tag):
    return {(r["task"], r["idx"]): r for r in map(json.loads, open(os.path.expanduser(f"~/eval2/{tag}.jsonl")))}


def mcnemar(b, c):
    """Exact two-sided McNemar p-value: b = only-A-correct, c = only-B-correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(ta, tb):
    A, B = load(ta), load(tb)
    out = {}
    for task in ("gsm8k", "mmlu_pro"):
        keys = [k for k in A if k[0] == task and k in B]
        only_a = sum(A[k]["correct"] and not B[k]["correct"] for k in keys)
        only_b = sum(B[k]["correct"] and not A[k]["correct"] for k in keys)
        acc_a = sum(A[k]["correct"] for k in keys) / len(keys)
        acc_b = sum(B[k]["correct"] for k in keys) / len(keys)
        diff_out = sum(A[k]["out_sha"] != B[k]["out_sha"] for k in keys)
        out[task] = {"n": len(keys), "acc_" + ta: round(acc_a, 4), "acc_" + tb: round(acc_b, 4),
                     "delta_pts": round(100 * (acc_b - acc_a), 2), "only_A_correct": only_a, "only_B_correct": only_b,
                     "mcnemar_p": round(mcnemar(only_a, only_b), 4), "different_outputs": diff_out}
    return out


ta, tb = sys.argv[1], sys.argv[2]
print(json.dumps({f"{ta} vs {tb}": compare(ta, tb)}, indent=1))
if len(sys.argv) > 3:
    print(json.dumps({f"noise floor {ta} vs {sys.argv[3]}": compare(ta, sys.argv[3])}, indent=1))
