# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare gdn_longgen_check.py results of X arms against P arms; hz `compare` format (29-09-2026).

usage: gdn_longgen_compare.py OUT.json P1=path X1=path [P2=path X2=path ...]
  [--max-drop 4] [--alpha 0.10] [--max-limit-increase 3] [--max-token-ratio 1.10]
Pairs are formed by round number (P1 with X1, P2 with X2). Per problem (set, index) the X and P outcomes are paired.
Gates (the thresholds are part of this frozen command line):
  complete     every file exists and says complete=true (no item aborted at the time budget)
  accuracy     mean over pairs of correct(X) - correct(P) >= -max_drop (of 130), AND the pooled exact two-sided sign
               test on discordant problems is not significantly worse (not: p < alpha with X-only-wrong > P-only-wrong)
  limit_hits   mean over pairs of hit_limit(X) - hit_limit(P) <= max_limit_increase (drift often shows as rambling)
  length       total generated tokens X / P <= max_token_ratio
Writes {"gates": {name: {"result": "pass"|"fail", "observed": ..., "params": ...}}, "summary_md": "..."}; exit 0.
"""
import argparse
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("files", nargs="+")
ap.add_argument("--max-drop", type=float, default=4)
ap.add_argument("--alpha", type=float, default=0.10)
ap.add_argument("--max-limit-increase", type=float, default=3)
ap.add_argument("--max-token-ratio", type=float, default=1.10)
a = ap.parse_args()
params = {"max_drop": a.max_drop, "alpha": a.alpha, "max_limit_increase": a.max_limit_increase,
          "max_token_ratio": a.max_token_ratio}
F = dict(f.split("=", 1) for f in a.files)
D, bad = {}, []
for k, p in F.items():
    try:
        D[k] = json.load(open(p))
    except (OSError, ValueError) as e:
        bad.append(f"{k}: {e}")
pairs = sorted({k[1:] for k in F if k[0] == "X"} & {k[1:] for k in F if k[0] == "P"})
G = {}
complete = not bad and bool(pairs) and all(D.get(f"{r}{n}", {}).get("complete") for n in pairs for r in "PX")
G["complete"] = {"result": "pass" if complete else "fail",
                 "observed": {k: v.get("complete") for k, v in D.items()} if not bad else bad, "params": {}}
if pairs and not bad:
    dc, dl, tx, tp, only_x, only_p, rows = [], [], 0, 0, 0, 0, []
    for n in pairs:
        P, X = D[f"P{n}"], D[f"X{n}"]
        ps = {(r["set"], r["i"]): r for r in P["items"]}
        xs = {(r["set"], r["i"]): r for r in X["items"]}
        for key in ps.keys() & xs.keys():
            if ps[key]["ok"] and not xs[key]["ok"]:
                only_p += 1
            elif xs[key]["ok"] and not ps[key]["ok"]:
                only_x += 1
        sp, sx = P["summary"]["all"], X["summary"]["all"]
        dc.append(sx["correct"] - sp["correct"])
        dl.append(sx["hit_limit"] - sp["hit_limit"])
        tx += sx["gen_tokens"]
        tp += sp["gen_tokens"]
        rows.append(f"| {n} | {sp['correct']} / {sx['correct']} of {sp['n']} | {sp['hit_limit']} / {sx['hit_limit']} | "
                    f"{sp['gen_tokens']:,} / {sx['gen_tokens']:,} | {P['wall_s']} / {X['wall_s']} s |")
    nd = only_x + only_p
    p_sign = 1.0 if nd == 0 else min(1.0, 2 * sum(math.comb(nd, i) for i in range(max(only_x, only_p), nd + 1)) / 2 ** nd)
    mdc, mdl, ratio = sum(dc) / len(dc), sum(dl) / len(dl), (tx / tp if tp else float("inf"))
    acc_ok = mdc >= -a.max_drop and not (p_sign < a.alpha and only_p > only_x)
    G["accuracy"] = {"result": "pass" if complete and acc_ok else "fail",
                     "observed": {"mean_delta_correct": mdc, "only_P_right": only_p, "only_X_right": only_x,
                                  "p_sign": round(p_sign, 4)}, "params": params}
    G["limit_hits"] = {"result": "pass" if complete and mdl <= a.max_limit_increase else "fail",
                       "observed": {"mean_delta_hit_limit": mdl}, "params": params}
    G["length"] = {"result": "pass" if complete and ratio <= a.max_token_ratio else "fail",
                   "observed": {"token_ratio_X_P": round(ratio, 4)}, "params": params}
    md = ("Long generation (AIME 2025 x30 at 12k + MATH-500[:100] at 6k, thinking on), P / X:\n\n"
          "| round | correct | hit limit | generated tokens | wall |\n|---|---|---|---|---|\n" + "\n".join(rows) +
          f"\n\nDiscordant problems: only P right {only_p}, only X right {only_x}, sign test p = {p_sign:.3f}.")
else:
    for g in ("accuracy", "limit_hits", "length"):
        G[g] = {"result": "fail", "observed": "no complete P/X pair", "params": params}
    md = "Long generation: no complete P/X pair (" + "; ".join(bad) + ")"
json.dump({"gates": G, "summary_md": md}, open(a.out, "w"), indent=1)
print(json.dumps({g: v["result"] for g, v in G.items()}))
