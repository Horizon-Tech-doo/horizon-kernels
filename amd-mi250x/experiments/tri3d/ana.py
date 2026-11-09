"""TRI3D analysis. raw/tri3d-*.jsonl. Ratios vLLM ms / ours ms (higher is better for ours). tri = Triton 2D as N3 raced; tri3 = Triton 3D as serving runs it; best = min(tri, tri3) per draw."""
import json, glob, statistics as st, collections as C
Rows = {}
for f in sorted(glob.glob("raw/tri3d-*.jsonl")):
    J = f.split("-")[-1][:-6]; T = C.defaultdict(list); E = []; X = {}
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "err": E.append(r); continue
        if r["t"] != "seed": continue
        k = (r["part"], r["Hkv"], r["B"], r["L"], r["mode"], r["seed"]); T[k + (r["impl"],)].append(r["ms"]); X[k] = r
    M = {k: st.median(v) for k, v in T.items()}; Rows[J] = (M, X, E)
    print("==", J, "err rows", len(E), "by impl max", {i: f"{max(e['rel'] for e in E if e['impl']==i):.2e}" for i in sorted({e['impl'] for e in E})}, "nan", sum(e["rel"] != e["rel"] for e in E))
    errv = C.defaultdict(list)
    for e in E: errv[(e['part'], e['Hkv'], e['B'], e['L'], e['mode'], e['seed'])].append((e['impl'], e['rel']))
    print("  tri3 err minus tri err, max", max((dict(v)['tri3'] - dict(v)['tri']) for v in errv.values() if 'tri3' in dict(v)))
    for part, Hkv in (("Z", 4), ("K", 2), ("V", 2)):
        cells = sorted({k[2:5] for k in X if k[0] == part})
        print(f" part {part} Hkv{Hkv}: B L mode n | tri3/pick med min max | tri/pick med | tri/tri3 (2D/3D) med min max | roc/pick med | roc/tri3 med | ms pick tri3 tri roc")
        for c in cells:
            ks = [k for k in X if k[0] == part and k[2:5] == c]
            g = lambda a, b: [M[k + (a,)] / M[k + (b,)] for k in ks if k + (a,) in M and k + (b,) in M]
            t3 = g("tri3", "pick"); tp = g("tri", "pick"); d = g("tri", "tri3"); rp = g("roc", "pick"); rt = g("roc", "tri3")
            ms = lambda i: st.median([M[k + (i,)] for k in ks if k + (i,) in M]) if any(k + (i,) in M for k in ks) else float("nan")
            a = f"{st.median(t3):.3f} {min(t3):.3f} {max(t3):.3f}" if t3 else "2D only"
            b = f"{st.median(d):.3f} {min(d):.3f} {max(d):.3f}" if d else "-"
            kk = g("tri3", "kk")
            print(f"  {c[0]:3d} {c[1]:6d} {c[2]:5s} {len(ks):2d} | {a} | {st.median(tp):.3f} | {b} | {st.median(rp):.2f} | {st.median(rt) if rt else 0:.2f} | {ms('pick'):.4f} {ms('tri3'):.4f} {ms('tri'):.4f} {ms('roc'):.4f}" + (f" | kk {ms('kk'):.4f} tri3/kk {st.median(kk):.3f}" if kk else ""))
        ks = [k for k in X if k[0] == part and k + ("tri3",) in M]
        v = [M[k + ("tri3",)] / M[k + ("pick",)] for k in ks]; w = [min(M[k + ("tri3",)], M[k + ("tri",)]) / M[k + ("pick",)] for k in ks]; d = [M[k + ("tri",)] / M[k + ("tri3",)] for k in ks]
        cm = C.defaultdict(list)
        for k, x in zip(ks, v): cm[k[2:5]].append(x)
        cv = [st.median(x) for x in cm.values()]
        print(f"  ALL 3D draws {len(ks)}: tri3/pick {min(v):.3f}-{max(v):.3f} below1 {sum(x<1 for x in v)}; cell medians {min(cv):.3f}-{max(cv):.3f}; tri/tri3 {min(d):.3f}-{max(d):.3f} 3D faster on {sum(x>1 for x in d)}")
    # V1 prediction, 16 D256 layers
    print(" V1 cells: predicted attention ms per step = 16 x kernel ms")
    for c in sorted({k[2:4] for k in X if k[0] == "V"}):
        k = ("V", 2) + c + ("v1", 0)
        print(f"  x{c[0]} L{c[1]}: pick {16*M[k+('pick',)]:.2f} kk {16*M[k+('kk',)]:.2f} tri3 {16*M[k+('tri3',)]:.2f} tri2d {16*M[k+('tri',)]:.2f} roc {16*M[k+('roc',)]:.2f} | tri3/kk {M[k+('tri3',)]/M[k+('kk',)]:.3f} tri3/pick {M[k+('tri3',)]/M[k+('pick',)]:.3f} tri/tri3 {M[k+('tri',)]/M[k+('tri3',)]:.3f}")
