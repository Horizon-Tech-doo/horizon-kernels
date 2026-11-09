# Ratio ours over torch hb, throughput, higher is better. argv: jsonl files of verify_r5 or smallm.
import json, sys, glob, statistics as st
for f in sys.argv[1:]:
    H = {}; R = {}
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        k = (r.get("M"), r.get("N"), r.get("K"), r.get("mode", f.split("_")[-1][:-6] if "sm_" in f else ""))
        if "rel" in r or "rel_err" in r or r.get("rule", 1) is None: H[k] = r
        if "impl" in r: R.setdefault(k, {}).setdefault(r["round"], {})[r["impl"]] = r["ms"]
    for k in H:
        h = H[k]; d = R.get(k, {})
        hb = "hb" if d and "hb" in next(iter(d.values())) else "hipblaslt"
        v = sorted(x[hb] / x["rule"] for x in d.values() if len(x) == 2)
        print(k, h.get("tile"), h.get("mod256"), (h.get("rel", h.get("rel_err")), h.get("rel_hb", h.get("rel_err_hb"))), [round(x, 3) for x in v], "ourTF", round(st.median(2*k[0]*k[1]*k[2]/x["rule"]*1e-9 for x in d.values()), 1) if d else None)
