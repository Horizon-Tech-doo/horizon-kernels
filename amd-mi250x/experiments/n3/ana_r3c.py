"""Tables from n3r3c raw. usage: ana_r3c.py file. Ratio vLLM ms / our ms. Median over cycles per seed, then median and range over 8 seeds (900-907 select, 908-915 held out)."""
import json, sys, statistics as st, collections as C
f = sys.argv[1]; T = C.defaultdict(list); E = {}; FL = {}; SP = {}; 
for line in open(f):
    if not line.startswith("{"): continue
    try: r = json.loads(line)
    except Exception: continue
    if r.get("t") == "seed": T[(r["B"], r["L"], r["mode"], r["seed"], r["impl"])].append(r["ms"])
    elif r.get("t") == "err": E[(r["B"], r["L"], r["mode"], r["impl"])] = r["rel"]
    elif r.get("t") == "floor": FL[(r["B"], r["L"], r["impl"])] = (r["S"], r["ms"])
    elif r.get("t") == "split": SP[(r["B"], r["L"], r["mode"], r["impl"])] = (r["S"], r["k1"], r["k2"])
print("max err", max(E.values()), "n cells", len(T))
M = {k: st.median(v) for k, v in T.items()}
for B, L in sorted({k[:2] for k in M}):
    for mode in ("skew", "part"):
        impls = sorted({k[4] for k in M if k[:3] == (B, L, mode)})
        print(f"== {B}x{L} {mode}: columns ms med(900-907) | /tri med [lo..hi] wins of 8 | held out (908-915) med [lo..hi]")
        rows = []
        for i in impls:
            r = []
            for lo, hi in ((900, 908), (908, 916)):
                sd = range(lo, hi)
                a = [M[(B, L, mode, s, i)] for s in sd if (B, L, mode, s, i) in M]; tr = [M[(B, L, mode, s, "tri")] / M[(B, L, mode, s, i)] for s in sd if (B, L, mode, s, i) in M and (B, L, mode, s, "tri") in M]
                r.append((st.median(a) if a else 0, st.median(tr) if tr else None, min(tr) if tr else 0, max(tr) if tr else 0, sum(x > 1 for x in tr)))
            rows.append((r[0][1] or 0, i, r))
        for _, i, r in sorted(rows, reverse=True):
            fl = FL.get((B, L, i)); sp = SP.get((B, L, mode, i))
            print(f"{i:10s} {r[0][0]:.3f} | {r[0][1] or 0:.3f} [{r[0][2]:.2f}..{r[0][3]:.2f}] {r[0][4]}/8 | {r[1][1] or 0:.3f} [{r[1][2]:.2f}..{r[1][3]:.2f}] {r[1][4]}/8 | S={fl[0] if fl else '-'} floor={fl[1] if fl else '-'} k1={sp[1] if sp else '-'} k2={sp[2] if sp else '-'}")
