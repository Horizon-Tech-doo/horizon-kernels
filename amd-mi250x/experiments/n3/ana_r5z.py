"""N3 round 5 final bytes, raw/r5z-*.jsonl. Ratio vLLM ms / ours ms, higher is better. Per job and per cell."""
import json, glob, statistics as st, collections as C
for f in sorted(glob.glob("raw/r5[zy]-*.jsonl")):
    T = C.defaultdict(list); X = {}; E = []; hd = []
    for l in open(f):
        if l.startswith("{"):
            r = json.loads(l)
            if r["t"] == "err": E.append(r); continue
            k = (r["part"], r["Hkv"], r["B"], r["L"], r["mode"], r["seed"]); T[k + (r["impl"],)].append(r["ms"]); X[k] = r
        elif l.startswith(("#", "7", "f")) or "gqa_decode" in l: hd.append(l.strip()[:60])
    M = {k: st.median(v) for k, v in T.items()}
    print("==", f[-13:-6], hd[:2], "err n", len(E), "max", max((e["rel"] for e in E), default=None), "nan", sum(e["rel"] != e["rel"] for e in E))
    for part, Hkv in (("Z", 4), ("K", 2)):
        for sel, name in ((None, "all"), (((8, 8192), (16, 4096), (32, 2048), (32, 1024)), "real4")):
            kz = [k for k in X if k[0] == part and (sel is None or k[2:4] in sel)]
            if not kz: continue
            tp = [M[k + ("tri",)] / M[k + ("pick",)] for k in kz]; rp = [M[k + ("roc",)] / M[k + ("pick",)] for k in kz]
            cm = {}
            for k in kz: cm.setdefault(k[2:5], []).append(M[k + ("tri",)] / M[k + ("pick",)])
            cv = [st.median(v) for v in cm.values()]
            print(f" {part} Hkv{Hkv} {name}: draws {len(kz)} tri/pick {min(tp):.3f}-{max(tp):.3f}, cell medians {min(cv):.3f}-{max(cv):.3f}, <1 {sum(a<1 for a in tp)}; roc/pick {min(rp):.2f}-{max(rp):.2f}")
        for cell in sorted({k[2:4] for k in X if k[0] == part}):
            for mode in ("base", "full", "part", "skew", "mid", "near"):
                ks = [k for k in X if k[0] == part and k[2:5] == cell + (mode,)]
                if not ks: continue
                tp = [M[k + ("tri",)] / M[k + ("pick",)] for k in ks]; rp = [M[k + ("roc",)] / M[k + ("pick",)] for k in ks]
                g = [M[k + ("pick0",)] / M[k + ("pick",)] - 1 for k in ks if k + ("pick0",) in M]; kkg = [M[k + ("kk",)] / M[k + ("pick",)] for k in ks if k + ("kk",) in M]
                print(f"   {part}{Hkv} {cell[0]}x{cell[1]} {mode:5s} n{len(ks)} tri/pick med {st.median(tp):.3f} min {min(tp):.3f} max {max(tp):.3f} | roc/pick med {st.median(rp):.2f} {min(rp):.2f}-{max(rp):.2f} | kk/pick {st.median(kkg) if kkg else 0:.3f} | skip gain n{len(g)} med {(st.median(g)*100 if g else 0):+.2f}% range {(min(g)*100 if g else 0):+.2f} {(max(g)*100 if g else 0):+.2f} | ms {st.median(M[k+('pick',)] for k in ks):.4f} tri {st.median(M[k+('tri',)] for k in ks):.4f} roc {st.median(M[k+('roc',)] for k in ks):.3f}")
    kz = [k for k in X if k[0] == "R"]
    if kz:
        a = [M[k + ("roc",)] / M[k + ("rh",)] for k in kz]; b = [M[k + ("tri",)] / M[k + ("rh",)] for k in kz]
        print(f" R rocm_hybrid draws {len(kz)}: roc/rh {min(a):.2f}-{max(a):.2f} tri/rh {min(b):.2f}-{max(b):.2f}")
