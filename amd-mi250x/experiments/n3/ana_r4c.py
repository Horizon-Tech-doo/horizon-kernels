"""Tables from n3r4c raw, per job. usage: ana_r4c.py file... Ratio X ms / ours ms; median of cycles per draw; mode row prints median [min..max] over draws, wins counted against 1.0."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = C.defaultdict(float); PC = C.defaultdict(list)
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "seed":
            k = (r["part"], r["Hkv"], r["B"], r["L"], r["mode"], r["seed"], r["impl"]); T[k].append(r["ms"]); PC[k].append(r["pct"])
        elif r["t"] == "err": E[(r["part"], r["dt"], r["impl"])] = max(E[(r["part"], r["dt"], r["impl"])], r["rel"])
    M = {k: st.median(v) for k, v in T.items()}; print("==", f); print("err", {"/".join(k): round(v, 5) for k, v in E.items()})
    for key in sorted({k[:5] for k in M}, key=lambda k: (k[0], -k[2] * k[3], k[2], ["full", "base", "near", "part", "mid", "skew"].index(k[4]))):
        seeds = sorted({k[5] for k in M if k[:5] == key}); impls = sorted({k[6] for k in M if k[:5] == key and k[6] not in ("tri", "roc")})
        s = []
        for i in impls:
            x = [M[key + (d, "tri")] / M[key + (d, i)] for d in seeds]
            y = [M[key + (d, "roc")] / M[key + (d, i)] for d in seeds if key + (d, "roc") in M]
            s.append(f"{i} {st.median(M[key + (d, i)] for d in seeds):.3f}ms T{st.median(x):.3f}" + (f"[{min(x):.3f} w{sum(a > 1 for a in x)}/{len(x)}]" if len(x) > 1 else "") + (f" R{st.median(y):.2f}" if y else "") + f" {st.median(st.median(PC[key + (d, i)]) for d in seeds):.0f}%")
        print(*key, f"tri {st.median(M[key + (d, 'tri')] for d in seeds):.3f}", "|", " | ".join(s))
