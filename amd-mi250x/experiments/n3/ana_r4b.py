"""Form grid from n3r4b raw, per job. usage: ana_r4b.py file... Ratio tri ms / ours ms, median of cycles per draw, then median [min..max] over draws."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = C.defaultdict(float); Rr = {}
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "seed": T[(r["B"], r["L"], r["mode"], r["seed"], r["impl"])].append(r["ms"]); Rr[(r["B"], r["L"], r["mode"], r["seed"])] = r["r"]
        elif r["t"] == "err": E["max"] = max(E["max"], r["rel"]) if r["impl"] != "tri" else E["max"]; E["tri"] = max(E["tri"], r["rel"]) if r["impl"] == "tri" else E["tri"]
    M = {k: st.median(v) for k, v in T.items()}; print("==", f, dict(E))
    for key in sorted({k[:3] for k in M}, key=lambda k: (k[1] * k[0], k[0], ["full", "base", "near", "part", "mid", "skew"].index(k[2]))):
        seeds = sorted({k[3] for k in M if k[:3] == key}); impls = sorted({k[4] for k in M if k[:3] == key and k[4] != "tri"}, key=lambda x: (x != "kk", int(x[1:]) if x != "kk" else 0))
        rr = st.median(Rr[key + (s,)] for s in seeds)
        out = []
        for i in impls:
            x = [M[key + (s, "tri")] / M[key + (s, i)] for s in seeds]
            out.append(f"{i} {st.median(x):.3f}" + (f"[{min(x):.2f}]" if len(x) > 1 else ""))
        best = max(impls, key=lambda i: st.median(M[key + (s, "tri")] / M[key + (s, i)] for s in seeds))
        print(*key, f"r={rr:.2f} tri {st.median(M[key + (s, 'tri')] for s in seeds):.3f} best {best} |", " ".join(out))
