"""Summary of e2run jsonl: mean over rounds of per-round median total; min..max; mean stage medians."""
import json, sys, collections
R = collections.defaultdict(list)
for fn in sys.argv[1:]:
    for l in open(fn):
        if not l.startswith("{"): continue
        d = json.loads(l)
        if d.get("kind") == "time":
            R[(d["T"], d["stack"], d["flush"], d["arm"])].append(d)
for k in sorted(R, key=lambda x: (-x[0], x[1], x[2])):
    v = R[k]; t = [x["tot_med"] for x in v]; n = len(v[0]["stages"])
    st = [sum(x["stages"][i] for x in v) / len(v) for i in range(n)]
    print(f"T={k[0]} {k[1]:4s} fl={k[2]} {k[3]:14s} n={len(t)} mean={sum(t)/len(t):7.3f} range={min(t):7.3f}-{max(t):7.3f} min1={min(x['tot_min'] for x in v):7.3f} stages=" + " ".join(f"{s:.3f}" for s in st))
