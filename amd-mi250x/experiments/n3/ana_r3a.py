"""Tables from n3r3a raw files. usage: ana_r3a.py file... Ratio vendor ms / our ms, medians per file."""
import json, sys, statistics as st, collections as C
for f in sys.argv[1:]:
    T = C.defaultdict(list); E = {}; N = {}
    stack = None
    for line in open(f):
        if not line.startswith("{"): continue
        try: r = json.loads(line)
        except Exception: continue
        if r.get("t") == "time": T[(r["stack"], r["B"], r["L"], r["mode"], r["impl"])].append((r["ms"], r["pct"]))
        elif r.get("t") == "err": E[(r["stack"], r["B"], r["L"], r["mode"], r["impl"])] = r["rel"]
        elif r.get("t") == "nan_tail": N[(r["stack"], r["B"], r["L"], r["impl"])] = r["rel"]
    print("==", f, "max err", max(E.values()) if E else None, "nan", {k: round(v, 6) for k, v in N.items()} if max(N.values() or [0]) > 1e-3 else "ok(max %.2e)" % max(N.values() or [0]))
    M = {k: st.median(a for a, _ in v) for k, v in T.items()}; Pc = {k: st.median(b for _, b in v) for k, v in T.items()}
    keys = sorted({k[:4] for k in M}, key=lambda k: (k[0], k[1], k[2], ["base", "full", "part", "skew"].index(k[3])))
    for k in keys:
        row = {i: M.get(k + (i,)) for i in ["land_b", "land_bs", "land_n", "land_ns", "land_r", "uni", "tri", "roc"]}
        s = " ".join(f"{i}={v:.3f}" for i, v in row.items() if v)
        rt = ""
        if row["tri"]:
            rt = f" | b/uni {row['uni']/row['land_b']:.2f} bs/uni {row['uni']/row['land_bs']:.2f} n/tri {row['tri']/row['land_n']:.2f} ns/tri {row['tri']/row['land_ns']:.2f} r/roc {row['roc']/row['land_r']:.2f} r/tri {row['tri']/row['land_r']:.2f} | %r {Pc[k+('land_r',)]:.0f} %n {Pc[k+('land_n',)]:.0f}"
        print(k[0][:3], k[1], k[2], k[3], s, rt)
