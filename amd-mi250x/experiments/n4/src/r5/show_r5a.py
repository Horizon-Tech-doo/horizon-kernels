# Ratio ours over hb, throughput, higher is better, per step in one hbr process. argv: jsonl files.
import json, sys, statistics as st
for f in sys.argv[1:]:
    off = {}; R = {}; chk = {}; order = []
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    bad = [l for l in open(f) if not l.startswith("{") and l.strip()]
    for r in rows:
        if "boff" in r: off[r["step"]] = r["boff"]
        if "impl" in r:
            R.setdefault(r["step"], {})[r["impl"]] = r["TF"]
            if r["impl"] not in order: order.append(r["impl"])
        if "check" in r: chk[r["check"]] = (r["fnv"], r["differ_from_hb"], r["n"])
    print("==", f, "steps", len(R), "other lines", len(bad), bad[:1])
    hbs = {o: [v["hb"] for s, v in R.items() if off[s] == o and "hb" in v] for o in sorted(set(off.values()))}
    print(" hb TF", {o: (round(min(v), 1), round(max(v), 1)) for o, v in hbs.items() if v})
    for a in order:
        if a == "hb": continue
        c = []
        for o in sorted(set(off.values())):
            v = sorted(R[s][a] / R[s]["hb"] for s in R if off[s] == o and a in R[s] and "hb" in R[s])
            t = [R[s][a] for s in R if off[s] == o and a in R[s]]
            if v: c.append(f"off{o}: {st.median(v):.3f} [{v[0]:.3f}-{v[-1]:.3f}] {st.median(t):.1f}TF")
        print(f" {a:18s}", " | ".join(c), chk.get(a, ""))
