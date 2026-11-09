# Cap table: ratio ours over hb, throughput, higher is better, per step per GCD, by arm and B offset. argv: raw dir, shape tags.
import json, glob, statistics as st, sys, os
d = sys.argv[1]
for s in sys.argv[2:]:
    v = {}; fn = set()
    for f in glob.glob(f"{d}/cap_{s}_d*.jsonl"):
        off = {}; R = {}
        for l in open(f):
            if not l.startswith("{"): continue
            r = json.loads(l)
            if "boff" in r: off[r["step"]] = r["boff"]
            if "impl" in r: R.setdefault(r["step"], {})[r["impl"]] = r["TF"]
            if "check" in r: fn.add((r["fnv"], r["differ_from_hb"]))
        for t, a in R.items():
            for k in a:
                if k != "hb" and "hb" in a: v.setdefault((k, off[t]), []).append(a[k] / a["hb"])
    print(s, "fnv set", fn)
    for k in sorted(v): x = sorted(v[k]); print(" ", k, len(x), "median", round(st.median(x), 3), "range", round(x[0], 3), round(x[-1], 3))
