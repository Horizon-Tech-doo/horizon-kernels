"""Tables from n3r3h raw. usage: ana_r3h.py file... Ratio vLLM ms / our ms, median of cycles per job."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = C.defaultdict(float); M0 = None
    for line in open(f):
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("t") == "time": T[(r["stack"], r["dt"], r["B"], r["L"], r["mode"], r["impl"])].append((r["ms"], r["pct"]))
        elif r.get("t") == "err": E[(r["stack"], r["dt"], r["impl"])] = max(E[(r["stack"], r["dt"], r["impl"])], r["rel"])
        elif r.get("t") == "meta": M0 = r
    print("==", f, "meta", M0); print("max err", {k: round(v, 5) for k, v in E.items()})
    M = {k: st.median(a for a, _ in v) for k, v in T.items()}; Pc = {k: st.median(b for _, b in v) for k, v in T.items()}
    for k in sorted({k[:5] for k in M}, key=lambda k: (k[0], k[1], k[2], k[3], ["base", "full", "part", "skew"].index(k[4]))):
        g = lambda i: M.get(k + (i,))
        s = " ".join(f"{i}={g(i):.3f}" for i in ["rh", "rh_vk", "n", "ns", "roc", "tri"] if g(i))
        rt = f" | rh/roc {g('roc')/g('rh'):.2f} n/tri {g('tri')/g('n'):.2f} n/roc {g('roc')/g('n'):.2f} roc/tri {g('tri')/g('roc'):.2f} | %rh {Pc[k+('rh',)]:.0f} %n {Pc[k+('n',)]:.0f}" if g("roc") else ""
        print(*k, s, rt)
