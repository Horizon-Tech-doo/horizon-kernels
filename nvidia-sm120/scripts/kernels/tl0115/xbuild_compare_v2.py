# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare two xbuild_dump_v2.py dumps (production build vs the v0.1.15 build).

    python xbuild_compare_v2.py dump_old.json dump_new.json [out.json]

PASS requires:
  1. the same case keys in both dumps and every output hash identical across builds (bit-identical outputs);
  2. the same adaptive variant chosen for every (shape, M);
Also reported (not part of PASS, since 1. already forces both builds to agree): inside each build,
  horizon_ops.row_scale's quantize_rs == prefill_rs's torch quantize_rs, and the native prefill with row_scale == with
  torch row scales == K3 (specified bit-identical; a mismatch here would be a pre-existing property of both builds).
Prints a per-family table and XBUILD_V2 PASS/FAIL.
"""
import collections
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ma, mb = a.pop("_meta"), b.pop("_meta")
fam = collections.defaultdict(lambda: [0, 0])
diff, missing, variant = [], sorted(set(a) ^ set(b)), []
for k in sorted(set(a) & set(b)):
    f = k.split(":")[0]
    fam[f][0] += 1
    if a[k]["sha"] == b[k]["sha"]:
        fam[f][1] += 1
    else:
        diff.append(k)
    if a[k].get("variant") != b[k].get("variant"):
        variant.append((k, a[k].get("variant"), b[k].get("variant")))

inner = []
for name, d in (("old", a), ("new", b)):
    for k in d:
        if k.startswith("rsq_torch:"):
            k2 = "rsq_rowscale:" + k.split(":", 1)[1]
            if d[k]["sha"] != d.get(k2, {}).get("sha"):
                inner.append((name, k, k2))
        if k.startswith("nat:") and k.endswith(":torchrs"):
            base = k[:-len(":torchrs")]
            shape_m = base.split(":", 1)[1]
            for k2 in (base + ":rowscale", "k3:" + shape_m):
                if d[k]["sha"] != d.get(k2, {}).get("sha"):
                    inner.append((name, k, k2))

ok = not diff and not missing and not variant
print(f"old: {ma.get('tilelang_file')} {ma.get('tilelang_version')}  new: {mb.get('tilelang_file')} {mb.get('tilelang_version')}")
print("| family | cases | bit-identical |\n|---|---|---|")
for f, (n, s) in sorted(fam.items()):
    print(f"| {f} | {n} | {s} |")
print(f"total {sum(v[0] for v in fam.values())} cases, identical {sum(v[1] for v in fam.values())}; "
      f"differing {len(diff)}, missing {len(missing)}, variant changes {len(variant)}, in-build identity failures {len(inner)}")
for x in (diff[:20], missing[:20], variant[:20], inner[:20]):
    for y in x:
        print("  ", y)
out = {"pass": ok, "families": fam, "diff": diff, "missing": missing, "variant": variant, "inner": inner,
       "old_meta": ma, "new_meta": mb}
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
print("IN_BUILD_IDENTITIES", "OK" if not inner else f"{len(inner)} MISMATCHES (old {sum(i[0] == 'old' for i in inner)}, new {sum(i[0] == 'new' for i in inner)})")
print("XBUILD_V2", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
