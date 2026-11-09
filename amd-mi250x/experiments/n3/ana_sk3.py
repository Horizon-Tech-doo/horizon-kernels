"""Tables from n3sk3 raw (BF16 repeat of the pick test). usage: ana_sk3.py file... Ratio vLLM ms / ours ms, median of 3 cycles per cell."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); P = C.defaultdict(list); E = C.defaultdict(float); W = {}; other = []
    for l in open(f):
        if not l.startswith("{"):
            if "Trace" in l or "Error" in l: other.append(l[:150])
            continue
        r = json.loads(l)
        if r["t"] == "seed":
            k = (r["B"], r["L"], r["mode"], r["seed"], r["impl"]); T[k].append(r["ms"]); P[k].append(r["pct"]); W[(r["B"], r["L"], r["mode"], r["seed"])] = r["which"]
        elif r["t"] == "err": E[r["impl"]] = max(E[r["impl"]], r["rel"])
    M = {k: st.median(v) for k, v in T.items()}
    print("==", f, "max err", {k: round(v, 5) for k, v in E.items()}, "errors:", other)
    for sh in sorted({k[:3] for k in M}):
        sd = sorted({k[3] for k in M if k[:3] == sh})
        g = lambda i, s: M[sh + (s, i)]
        x = [g("tri", s) / g("pick", s) for s in sd]; o = [g("tri", s) / g("old", s) for s in sd]; r = [g("roc", s) / g("pick", s) for s in sd]
        print(*sh, "which", sorted({W[sh + (s,)] for s in sd}), f"pick/tri med {st.median(x):.3f} [{min(x):.3f}..{max(x):.3f}] wins {sum(a > 1 for a in x)}/{len(x)} | old/tri {st.median(o):.3f} | pick/roc {st.median(r):.2f} | pick ms {st.median(g('pick', s) for s in sd):.3f} tri ms {st.median(g('tri', s) for s in sd):.3f} | %HBM {st.median(st.median(P[sh + (s, 'pick')]) for s in sd):.1f}")
