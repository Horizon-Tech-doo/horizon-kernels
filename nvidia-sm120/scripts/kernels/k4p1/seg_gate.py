# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Exit 0 if the standalone segment test (seg_test2.py output) passed for variant `ra` (bit-identical everywhere, no
# error flags, attention-next variant identical too); print the verdict. usage: python seg_gate.py SEG.jsonl [variant]
import json, sys
v = sys.argv[2] if len(sys.argv) > 2 else "ra"
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
a = [r for r in rows if r.get("part") == "A" and r.get("ncta") == v]
a2 = [r for r in rows if r.get("part") == "A2" and r.get("variant") == v]
ok = bool(a) and all(r.get("ok") for r in a) and bool(a2) and all(r.get("bit_identical") and r.get("err") == 0 for r in a2)
print(f"seg gate {v}: {'PASS' if ok else 'FAIL'} (A rows {len(a)}, A2 rows {len(a2)})")
sys.exit(0 if ok else 1)
