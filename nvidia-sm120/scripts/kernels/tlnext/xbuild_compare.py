# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Compare two xbuild_dump.py JSON files: every case must have the same hash (bit-identical output) and, for the
# adaptive cases, the same selected variant.   xbuild_compare.py cur.json next.json
import json, sys
from collections import Counter
a, b = (json.load(open(p)) for p in sys.argv[1:3])
keys = sorted(k for k in set(a) | set(b) if not k.startswith("_"))
bad, fam, fam_bad = [], Counter(), Counter()
for k in keys:
    f = k.split(":")[0]
    fam[f] += 1
    if k not in a or k not in b or a[k] != b[k]:
        bad.append((k, a.get(k), b.get(k)))
        fam_bad[f] += 1
print("meta A:", a["_meta"]); print("meta B:", b["_meta"])
for f in sorted(fam):
    print(f"{f:10s} {fam[f] - fam_bad[f]:4d}/{fam[f]:4d} bit-identical")
for k, x, y in bad[:40]:
    print("DIFF", k, x, y)
print(json.dumps({"cases": len(keys), "identical": len(keys) - len(bad), "RESULT": "IDENTICAL" if not bad else "DIFFERENT"}))
