"""N3 round 5 Part T: table candidates on held-out seeds 972-979, per job. Ratio tri ms / ours ms. Each candidate replaces the current table's choice in an r band; compare to current on the affected draws."""
import json, glob, statistics as st, collections as C
CAND = {"H4 16x4k r>=.84 S5": ((4,16,4096), .84, 9, "S5"), "H2 8x32k r>=.9 S20": ((2,8,32768), .9, 9, "S20"), "H2 8x32k r>=.8 S20": ((2,8,32768), .8, 9, "S20"), "H2 32x8k r>=.9 S5": ((2,32,8192), .9, 9, "S5"), "H2 32x8k r>=.8 S5": ((2,32,8192), .8, 9, "S5"),
        "H2 16x8k r<.5 kk": ((2,16,8192), 0, .5, "kk"), "H2 16x8k r<.5 S20": ((2,16,8192), 0, .5, "S20"), "H2 16x8k r<.6 kk": ((2,16,8192), 0, .6, "kk"), "H2 16x8k .72-.82 S10": ((2,16,8192), .72, .82, "S10"),
        "H4 8x8k .72-.82 kk": ((4,8,8192), .72, .82, "kk"), "H4 8x8k .72-.82 S10": ((4,8,8192), .72, .82, "S10"), "H4 8x8k .72-.82 S16": ((4,8,8192), .72, .82, "S16"), "H4 16x4k .58-.84 S8": ((4,16,4096), .58, .84, "S8"), "H4 16x4k r<.58 S10": ((4,16,4096), 0, .58, "S10"), "H4 16x4k r<.58 S6": ((4,16,4096), 0, .58, "S6")}
for f in sorted(glob.glob("raw/r5a-*.jsonl")):
    T = C.defaultdict(list); X = {}
    for l in open(f):
        if l.startswith("{"):
            r = json.loads(l)
            if r["t"] == "seed" and r["part"] == "T": k = (r["Hkv"], r["B"], r["L"], r["mode"], r["seed"]); T[k + (r["impl"],)].append(r["ms"]); X[k] = r
    if not X: continue
    M = {k: st.median(v) for k, v in T.items()}; print("==", f[-13:-6])
    for nm, (cell, lo, hi, form) in CAND.items():
        a, b = [], []
        for k, r in X.items():
            if k[:3] == cell and lo <= r["r"] < hi:
                t = M[k + ("tri",)]; cur = "kk" if r["table"] == 0 else f"S{r['table']}"; a.append(t / M[k + (form,)]); b.append(t / M[k + (cur,)])
        if a: print(f" {nm:24s} n{len(a):3d} cand med {st.median(a):.3f} min {min(a):.3f} | current med {st.median(b):.3f} min {min(b):.3f} | paired med {st.median([x/y for x,y in zip(a,b)]):.3f}")
    for cell in sorted({k[:3] for k in X}):
        lows = sorted(((round(X[k]['r'],2), k[3], k[4], X[k]['table'], round(M[k+('tri',)]/M[k+('kk' if X[k]['table']==0 else f"S{X[k]['table']}",)],2)) for k in X if k[:3]==cell), key=lambda a:a[4])[:3]
        print("  lowest table draws", cell, lows)
