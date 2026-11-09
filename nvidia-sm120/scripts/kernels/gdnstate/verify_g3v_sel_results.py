# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Verify test_g3v_sel_gpu.py records (30-09-2026 evening). Exit 0 = PASS.
Gate: complete run; all 9 equivalence cases x 3 hook kinds x 2 repetitions present with state / meta / ridx diff 0;
each hook kind expanded rings (rings_expanded > 0) in at least 3 cases (the comparison is not vacuous); the hook-cost
records for n = 1 / 8 / 64 present (reported, not gated)."""
import json
import sys

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
eq = [r for r in recs if r.get("test") == "equiv"]
hk = [r for r in recs if r.get("test") == "hooks"]
end = [r for r in recs if r.get("test") == "end"]
fails = []
if not end or not end[-1].get("complete"):
    fails.append("run incomplete (no end record)")
cases = {(r["n"], r["B"]) for r in eq}
if len(cases) != 9:
    fails.append(f"{len(cases)} equivalence cases, want 9")
if len(eq) != 9 * 3 * 2:
    fails.append(f"{len(eq)} equivalence records, want 54")
bad = [r for r in eq if r["state_diff"] or r["meta_diff"] or r["ridx_diff"]]
if bad:
    fails.append(f"{len(bad)} records differ, first {bad[0]}")
for kind in ("pre_pure", "pre_mixed", "post"):
    k = sum(1 for r in eq if r["kind"] == kind and r["rings_expanded"] > 0 and r["floats_rewritten"] > 0)
    if k < 3:
        fails.append(f"{kind}: only {k} records expanded anything (vacuous comparison)")
if sorted(r["n"] for r in hk) != [1, 8, 64]:
    fails.append("hook-cost records missing")
for r in hk:
    print(f"hooks n={r['n']}: stream/step old {r['old']['stream_us_per_step_median']} us -> new "
          f"{r['new']['stream_us_per_step_median']} us (x{r['stream_ratio_new_old']}); GPU ops/step "
          f"{r['old']['gpu_ops_per_step']} -> {r['new']['gpu_ops_per_step']}; host/step {r['old']['host_us_per_step']} "
          f"-> {r['new']['host_us_per_step']} us")
print(f"equivalence: {len(eq)} records, {len(bad)} differing; "
      f"rings expanded per kind: " + ", ".join(f"{k} {sum(r['rings_expanded'] for r in eq if r['kind'] == k)}"
                                               for k in ("pre_pure", "pre_mixed", "post")))
print("PASS: complete, one-launch expand bit-identical to the per-group path" if not fails else "FAIL: " + "; ".join(fails))
sys.exit(1 if fails else 0)
