"""Pooled summary of round 4 raw: 64 x 1k skew by seed set and job, pick against tri by draw; real-frame ranges. Ratio vLLM ms / ours ms."""
import json, glob, statistics as st, collections as C
def load(tag):
    out = {}
    for f in glob.glob(f"raw/{tag}-*.jsonl"):
        T = C.defaultdict(list); RR = {}
        for l in open(f):
            if l.startswith("{"):
                r = json.loads(l)
                if r["t"] == "seed": k = (r.get("part"), r.get("Hkv", 4), r["B"], r["L"], r["mode"], r["seed"], r["impl"]); T[k].append(r["ms"]); RR[k[:6]] = r["r"] if "r" in r else r["sum"] / (r["B"] * r["max"])
        out[f[-13:-6]] = ({k: st.median(v) for k, v in T.items()}, RR)
    return out
A, C4, D_ = load("r4a"), load("r4c"), load("r4d")
print("64 x 1k skew, S4 swap rc bits, against tri:")
for j, (M, R) in A.items():
    x = {k[5]: M[k[:6] + ("tri",)] / M[k] for k in M if k[0] == "B" and k[6] == "TTb"}; h = {k[5]: M[k[:6] + ("tri",)] / M[k] for k in M if k[0] == "B" and k[6] == "TTh"}
    for lo, hi in ((924, 932), (932, 940)):
        v = [x[s] for s in x if lo <= s < hi]; print(f" a {j} seeds {lo}-{hi-1} med {st.median(v):.3f} min {min(v):.3f} max {max(v):.3f} wins {sum(a > 1 for a in v)}/{len(v)}")
    v = [h[s] for s in h]; print(f" a {j} hip seeds 924-939 med {st.median(v):.3f} min {min(v):.3f} wins {sum(a > 1 for a in v)}/{len(v)}")
pool = C.defaultdict(list)
for src, Mm in (("c", C4), ("d", D_)):
    for j, (M, R) in Mm.items():
        rows = []
        for k in M:
            if k[1:5] == (4, 64, 1024, "skew") and k[6] == "pick":
                t = M[k[:6] + ("tri",)] / M[k]; rows.append((k[5], round(R[k[:6]], 3), round(t, 3))); pool[k[5]].append(t)
        v = [a[2] for a in rows]; print(f" {src} {j} med {st.median(v):.3f} min {min(v):.3f} max {max(v):.3f} wins {sum(a > 1 for a in v)}/{len(v)}", sorted(rows))
print("pick vs tri, per job, Hkv 4 from r4c (held out 948-955), ranges of draw ratios and of cell medians; roc on seed 948 only")
SHAPES = {"real": [(8, 8192), (16, 4096), (32, 2048), (32, 1024)], "all6": [(8, 8192), (16, 4096), (32, 2048), (32, 1024), (64, 1024), (64, 2048)]}
for j, (M, R) in C4.items():
    for nm, shs in SHAPES.items():
        tv, cv, rv = [], [], []
        for sh in shs:
            for mode in ("base", "full", "part", "skew", "mid", "near"):
                ks = [k for k in M if k[0] == "H" and k[2:4] == sh and k[4] == mode and k[6] == "pick"]
                v = [M[k[:6] + ("tri",)] / M[k] for k in ks]; tv += v; cv.append(st.median(v))
                rv += [M[k[:6] + ("roc",)] / M[k] for k in ks if k[:6] + ("roc",) in M]
        print(f" {j} {nm}: tri draws {min(tv):.3f}-{max(tv):.3f}, cell medians {min(cv):.3f}-{max(cv):.3f}; roc {min(rv):.2f}-{max(rv):.2f}; n<1 {sum(a < 1 for a in tv)}/{len(tv)}")
    # rocm_hybrid and pc effect
    x = [M[k[:6] + ("roc",)] / M[k] for k in M if k[0] == "R" and k[6] == "rh"]; y = [M[k[:6] + ("tri",)] / M[k] for k in M if k[0] == "R" and k[6] == "rh"]; z = [M[k[:6] + ("rhh",)] / M[k] for k in M if k[0] == "R" and k[6] == "rh"]
    print(f"  rocm_hybrid vs roc {min(x):.2f}-{max(x):.2f}; vs tri {min(y):.2f}-{max(y):.2f}; bits gain {min(z):.3f}-{max(z):.3f}")
    z = [M[k[:6] + ("kkh",)] / M[k] for k in M if k[0] == "H" and k[6] == "kk"]; print(f"  lbnhc bits/hip ms ratio {min(z):.3f}-{max(z):.3f}, median {st.median(z):.3f}")
    # Hkv 2 from r4c: kk and rh, roc
    for arm in ("kk", "rh", "S320", "S640"):
        ks = [k for k in M if k[0] == "K" and k[6] == arm]
        print(f"  Hkv2 {arm}: tri {min(M[k[:6] + ('tri',)] / M[k] for k in ks):.2f}-{max(M[k[:6] + ('tri',)] / M[k] for k in ks):.2f}, roc {min(M[k[:6] + ('roc',)] / M[k] for k in ks):.2f}-{max(M[k[:6] + ('roc',)] / M[k] for k in ks):.2f}")
for j, (M, R) in D_.items():
    ks = [k for k in M if k[1] == 2 and k[6] == "pick"]; kk = [k for k in M if k[1] == 2 and k[6] == "kk"]
    v = [M[k[:6] + ("tri",)] / M[k] for k in ks]; w = [M[k[:6] + ("kk",)] / M[k] for k in ks]
    print(f" d {j} Hkv2 pick vs tri draws {min(v):.3f}-{max(v):.3f} n<1 {sum(a < 1 for a in v)}/{len(v)}; pick vs kk (kk ms/pick ms) {min(w):.3f}-{max(w):.3f} n<0.98 {sum(a < 0.98 for a in w)}/{len(w)}")
