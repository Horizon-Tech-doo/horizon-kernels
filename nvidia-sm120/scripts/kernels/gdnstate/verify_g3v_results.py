# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Completeness + gate verifier for test_g3v_gpu.py results (30-09-2026). CPU only. Exit 0 = complete and every
correctness gate passed. Gates: every chain row (nr 2/8/32/64 full, 2/32 quick) with out/copy/committed/block diffs
all 0 and copies > 0; graph determinism + replay; bench rows for nr 1/8/32/64 x QL 2/4 with finite positive times
for both arms; a complete terminal record. Speed is reported, never gated here.
usage: python verify_g3v_results.py RESULTS.jsonl [quick]"""
import json
import math
import sys


def verify(recs, quick=False):
    by = {}
    for r in recs:
        by.setdefault(r.get("test"), []).append(r)
    for nr in ((2, 32) if quick else (2, 8, 32, 64)):
        rows = [r for r in by.get("chain", []) if r.get("nr") == nr]
        if not rows:
            return False, f"missing chain nr={nr}"
        for r in rows:
            if any(r.get(k) != 0 for k in ("out_diff", "copy_diff", "committed_diff", "blocks_diff")):
                return False, f"chain nr={nr} mismatch: " + json.dumps({k: r.get(k) for k in (
                    "out_diff", "copy_diff", "committed_diff", "blocks_diff")})
            if not r.get("copies", 0) > 0:
                return False, f"chain nr={nr}: no align copies exercised"
    g = by.get("graph", [])
    if not g or not all(x.get("deterministic") is True and x.get("graph_replay_equal") is True for x in g):
        return False, "graph/determinism failed or missing"
    for nr in (1, 8, 32, 64):
        for ql in (2, 4):
            rows = [r for r in by.get("bench", []) if r.get("nr") == nr and r.get("ql") == ql]
            if not rows:
                return False, f"missing bench nr={nr} ql={ql}"
            us = rows[-1].get("us", {})
            for arm in ("vllm_f32", "g3v_f32"):
                v = us.get(arm)
                if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                    return False, f"bench nr={nr} ql={ql} {arm} invalid"
    e = by.get("end", [])
    if not e or e[-1].get("complete") is not True:
        return False, "no complete terminal record"
    return True, "complete, all correctness gates passed"


if __name__ == "__main__":
    recs = []
    for line in open(sys.argv[1]):
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    ok, why = verify(recs, len(sys.argv) > 2 and sys.argv[2] == "quick")
    print(("PASS: " if ok else "FAIL: ") + why)
    sys.exit(0 if ok else 1)
