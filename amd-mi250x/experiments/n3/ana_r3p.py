"""Tables from n3r3p raw. usage: ana_r3p.py file... Ratio vLLM ms / our ms: median over cycles per seed, then median and range over draws."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = C.defaultdict(float); W = {}
    for line in open(f):
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("t") == "seed": T[(r["stack"], r["B"], r["L"], r["N"], r["mode"], r["seed"], r["impl"])].append(r["ms"]); W[(r["stack"], r["B"], r["L"], r["N"], r["mode"], r["seed"])] = r["which"]
        elif r.get("t") == "err": E[(r["stack"], r["impl"])] = max(E[(r["stack"], r["impl"])], r["rel"])
    print("==", f, "max err", {k: round(v, 5) for k, v in E.items()})
    M = {k: st.median(v) for k, v in T.items()}
    for k in sorted({k[:5] for k in M}):
        sds = sorted({q[5] for q in M if q[:5] == k})
        pk = [M[k + (s, "pick")] for s in sds]; ol = [M[k + (s, "old")] for s in sds]
        g = st.median(ol[i] / pk[i] for i in range(len(sds)))
        line = f"{k} n={len(sds)} old={st.median(ol):.3f} pick={st.median(pk):.3f} pick_gain={g:.3f} which={sorted({W[k + (s,)] for s in sds})}"
        if k + (sds[0], "tri") in M:
            for v in ("tri", "roc"):
                a = [M[k + (s, v)] / M[k + (s, "pick")] for s in sds]; b = [M[k + (s, v)] / M[k + (s, "old")] for s in sds]
                line += f" | pick/{v} {st.median(a):.2f} [{min(a):.2f}..{max(a):.2f}] {sum(x > 1 for x in a)}/{len(a)} old/{v} {st.median(b):.2f}"
        print(line)
