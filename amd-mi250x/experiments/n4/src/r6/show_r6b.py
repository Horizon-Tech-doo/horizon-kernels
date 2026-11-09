# Ratio ours over torch hb, throughput, higher is better. argv: job dirs. sm rows: hb on b0 and on the padded B's rows. alloc rows: hb on the same B, per draw.
import json, sys, glob, os, statistics as st
for D in sys.argv[1:]:
    print("==", D)
    for f in sorted(glob.glob(D + "/sm_*.jsonl")):
        rows = [json.loads(l) for l in open(f) if l.startswith("{")]; R = {}
        h = rows[0] if rows else {}
        for r in rows:
            if "impl" in r: R.setdefault(r["round"], {})[r["impl"]] = r["TF"]
        if not R: print(os.path.basename(f), "EMPTY", open(f[:-5] + "err").read()[-300:]); continue
        o = [os.path.basename(f), h.get("tile"), h.get("rel_err"), h.get("rel_err_hb")]
        for hb in ("hipblaslt", "hipblaslt_padb"):
            if hb in next(iter(R.values())):
                v = sorted(x["rule"] / x[hb] for x in R.values()); o.append(f"/{hb} {st.median(v):.3f} [{v[0]:.3f}-{v[-1]:.3f}] hbTF {st.median(x[hb] for x in R.values()):.1f}")
        o.append(f"ourTF {st.median(x['rule'] for x in R.values()):.1f}"); print(*o)
    p = D + "/alloc.jsonl"
    if os.path.exists(p):
        R = {}; rel = {}
        for l in open(p):
            if not l.startswith("{"): continue
            r = json.loads(l)
            if "rel" in r: rel[(r["M"], r["draw"])] = (r["rel"], r["rel_hb"], r["a_b_low"])
            if "impl" in r: R.setdefault((r["M"], r["draw"]), {}).setdefault(r["round"], {})[r["impl"]] = r["TF"]
        by = {}
        for (M, d), x in sorted(R.items()):
            v = [y["rule"] / y["hb"] for y in x.values()]; by.setdefault(M, []).extend(v)
            print(f" M{M} d{d} ratio {st.median(v):.3f} hbTF {st.median(y['hb'] for y in x.values()):.1f} ourTF {st.median(y['rule'] for y in x.values()):.1f} rel {rel[(M, d)][:2]}")
        for M, v in by.items(): print(f" M{M} all draws min {min(v):.3f} med {st.median(v):.3f} max {max(v):.3f}")
