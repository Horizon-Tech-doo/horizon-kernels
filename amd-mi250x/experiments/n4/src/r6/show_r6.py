# Ratio ours over torch hb, throughput, higher is better, per round, same process. argv: pad.jsonl files.
import json, sys, statistics as st
for f in sys.argv[1:]:
    print("==", f)
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    R = {}; rel = {}; mod = {}; 
    for r in rows:
        if "rel" in r: rel[(r["tag"], r["arm"])] = r["rel"]; mod[(r["tag"], r["arm"])] = r["mod256"]
        if "impl" in r: R.setdefault(r["tag"], {}).setdefault(r["round"], {})[r["impl"]] = r["TF"]
    for tag, d in R.items():
        arms = list(next(iter(d.values())).keys())
        print(tag, "rounds", len(d))
        for a in sorted(arms):
            hb = "hb|" + a.split("|")[1] if "|" in a else "hb"
            v = sorted(x[a] / x[hb] for x in d.values())
            print(f"  {a:12s} {st.median(v):.3f} [{v[0]:.3f}-{v[-1]:.3f}] TF {st.median(x[a] for x in d.values()):.1f} rel {rel.get((tag, a))} mod {mod.get((tag, a))}")
