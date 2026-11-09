"""Tables from n3r4a raw. usage: ana_r4a.py file... Ratio vLLM ms / ours ms, medians over cycles."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = C.defaultdict(float); PP = C.defaultdict(list); pct = C.defaultdict(list)
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "seed": T[(r["part"], r["B"], r["L"], r["mode"], r["seed"], r["impl"])].append(r["ms"]); pct[(r["part"], r["B"], r["L"], r["mode"], r["seed"], r["impl"])].append(r["pct"])
        elif r["t"] == "err": E[r["impl"]] = max(E[r["impl"]], r["rel"])
        elif r["t"] == "part": PP[r["impl"]].append(r["ms"])
    M = {k: st.median(v) for k, v in T.items()}; print("==", f, "err", {k: round(v, 5) for k, v in E.items()})
    print("part D ms", {k: round(st.median(v), 4) for k, v in PP.items()})
    for key in sorted({k[:4] for k in M}):
        seeds = sorted({k[4] for k in M if k[:4] == key}); impls = sorted({k[5] for k in M if k[:4] == key})
        if key[0] == "A":
            print(key, "tri", M[key + (seeds[0], "tri")], "roc", M[key + (seeds[0], "roc")], "| " + " ".join(f"{i} {M[key + (seeds[0], i)]:.4f}/{M[key + (seeds[0], 'tri')] / M[key + (seeds[0], i)]:.3f}" for i in impls if i not in ("tri", "roc")), "| pct " + " ".join(f"{i} {st.median(pct[key + (seeds[0], i)]):.1f}" for i in impls if i in ("S4b", "S4h", "S5b")))
        else:
            def R(i, ss): return [M[key + (s, "tri")] / M[key + (s, i)] for s in ss]
            print(key, f"n={len(seeds)} tri {st.median(M[key + (s, 'tri')] for s in seeds):.4f} | " + " ".join(f"{i} {st.median(R(i, seeds)):.3f} [{min(R(i, seeds)):.3f}..{max(R(i, seeds)):.3f}] w{sum(a > 1 for a in R(i, seeds))}" for i in impls if i != "tri"))
