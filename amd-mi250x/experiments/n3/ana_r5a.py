"""N3 round 5 analysis of raw/r5a-*.jsonl. Ratio vLLM ms / ours ms, higher is better. Gain = cur ms / skip ms - 1, same job."""
import json, glob, statistics as st, collections as C, sys
files = sorted(glob.glob("raw/r5a-*.jsonl")); PICK = None
for f in files:
    j = f[-13:-6]; T = C.defaultdict(list); X = {}; ERR = []
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "err": ERR.append(r); continue
        k = (r["part"], r.get("dt"), r["Hkv"], r["B"], r["L"], r["mode"], r["seed"]); T[k + (r["impl"],)].append(r["ms"]); X[k] = r
    M = {k: st.median(v) for k, v in T.items()}
    print("==", j, "err rows", len(ERR), "max rel", max((e["rel"] for e in ERR if e["part"] != "N"), default=None))
    for e in ERR:
        if e["part"] == "N" : print("  N", e["B"], e["L"], e["mode"], e["impl"], "nan", e["nan"], f"{e['rel']:.2e}")
    # Part A
    A = {k: r for k, r in X.items() if k[0] == "A"}
    for cell in sorted({k[3:6] for k in A}):
        ks = [k for k in A if k[3:6] == cell]
        g = [M[k + ("cur",)] / M[k + ("skip",)] - 1 for k in ks]; tc = [M[k + ("tri",)] / M[k + ("cur",)] for k in ks]; tx = [M[k + ("tri",)] / M[k + ("skip",)] for k in ks]
        print(f" A {cell} n{len(ks)} gain med {st.median(g)*100:+.2f}% range {min(g)*100:+.2f} {max(g)*100:+.2f} | tri/cur med {st.median(tc):.3f} min {min(tc):.3f} <1 {sum(a<1 for a in tc)} | tri/skip med {st.median(tx):.3f} min {min(tx):.3f} <1 {sum(a<1 for a in tx)}")
    ks = [k for k in A if k[3:6] == (64, 1024, "skew")]
    for lo, hi in ((964, 972), (972, 980)):
        z = [M[k + ("tri",)] / M[k + ("skip",)] for k in ks if lo <= k[6] < hi]
        if z: print(f"   64x1k skew seeds {lo}-{hi-1} tri/skip med {st.median(z):.3f} min {min(z):.3f} <1 {sum(a<1 for a in z)}/{len(z)}")
    for k in sorted(X):
        if k[0] == "F": print(" F", k[5], {i: M[k + (i,)] for i in ("kk", "S5", "tri")}, "tri/kk %.3f tri/S5 %.3f" % (M[k + ("tri",)] / M[k + ("kk",)], M[k + ("tri",)] / M[k + ("S5",)]))
    # Part T
    TT = {k: r for k, r in X.items() if k[0] == "T"}
    for cell in sorted({k[2:5] for k in TT}):
        ks = sorted(k for k in TT if k[2:5] == cell); forms = sorted({i[7] for i in M if i[:1] == ("T",) and i[2:5] == cell and i[7] != "tri"})
        rows = C.defaultdict(list)
        for k in ks:
            r = X[k]; t = M[k + ("tri",)]; v = {i: t / M[k + (i,)] for i in forms}
            tb = v["kk" if r["table"] == 0 else f"S{r['table']}"]; rows["table"].append(tb); rows["rule"].append(v[f"S{r['rule']}"]); rows["oracle"].append(max(v.values())); rows["kk"].append(v["kk"])
        print(f" T {cell} draws {len(ks)}: " + " ".join(f"{n} med {st.median(a):.3f} min {min(a):.3f}" for n, a in rows.items()) + " rule-vs-table med %+.2f%%" % ((st.median([a / b for a, b in zip(rows['rule'], rows['table'])]) - 1) * 100))
