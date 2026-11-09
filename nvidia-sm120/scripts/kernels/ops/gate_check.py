# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# prints the arms that passed their standalone gate: attn (A1) and rs (row-scale fusion)
import json, os
D = os.path.expanduser("$HK_WORK/ops")


def rows(f):
    p = f"{D}/{f}"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


ok = []
a = [r for r in rows("attn_a1.jsonl") if r.get("test") == "attn_correct"]
if a and all(r["deterministic"] and r["ours_rel_l2"] <= 1.5 * r["xqa_rel_l2"] + 1e-3 for r in a):
    ok.append("attn")
s = [r for r in rows("rowscale.jsonl") if r.get("summary") == "rowscale"]
if s and s[-1]["bit_diffs_total"] == 0:
    ok.append("rs")
print(" ".join(ok))
