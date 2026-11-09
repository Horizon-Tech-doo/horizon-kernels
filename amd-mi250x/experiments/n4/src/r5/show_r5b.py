# Ratio ours over hb, throughput, higher is better. argv: job B raw dir.
import json, sys, glob, os, statistics as st, re
d = sys.argv[1]
print("rule gap, through the script (hand asm)")
for f in sorted(glob.glob(d + "/gap_*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    h = next((r for r in rows if "rule" in r), None); by = {}
    for r in rows:
        if "impl" in r: by.setdefault(r["round"], {})[r["impl"]] = r["TF"]
    v = sorted(x["rule"] / x["hipblaslt"] for x in by.values() if len(x) == 2)
    print(f" {os.path.basename(f)[4:-6]:24s}", h and h.get("rule"), h and h.get("hand_tile"), h and h.get("a_minus_b_mod256"), h and (h.get("rel_err"), h.get("rel_err_hb")), [round(x, 3) for x in v], "ourTF", by and round(st.median(x["rule"] for x in by.values()), 1), "hbTF", by and round(st.median(x["hipblaslt"] for x in by.values()), 1))
print("small-M, compiled TileLang; ratio over same-placement torch hb, per round")
rows = [json.loads(l) for l in open(d + "/sm.jsonl") if l.startswith("{")]
sh = {}; rel = {}
for r in rows:
    k = (r.get("M"), r.get("N"), r.get("K"))
    if "impl" in r: sh.setdefault(k, {}).setdefault(r["impl"], {})[r["round"]] = r["ms"]
    if "rel_err" in r: rel[(k, r["arm"])] = r["rel_err"]
for k, a in sh.items():
    dh = os.path.join(d, f"hbdirect_{ {(64,5120,5120):'64x5k',(32,5120,5120):'32x5k',(16,17408,5120):'16x17k',(64,17408,5120):'64x17k',(64,5120,17408):'64x5kK17k',(1,5120,5120):'1x5k'}[k]}.jsonl")
    dm = [json.loads(l)["ms"] for l in open(dh) if l.startswith('{"impl"')]
    hu, hp = st.median(a["hb|u"].values()), st.median(a["hb|p"].values())
    print(k, "hb torch ms u", hu, "p", hp, "direct", round(st.median(dm), 5) if dm else None, "rel hb", rel.get((k, "hb|u")))
    for n in sorted(a):
        if n.startswith("hb"): continue
        pl = n[-1]; hbk = "hb|" + pl
        v = sorted(a[hbk][r] / a[n][r] for r in a[n])
        ms = st.median(a[n].values())
        print(f"   {n:20s} {[round(x, 3) for x in v]} over direct {st.median(dm) / ms if dm else 0:.3f} rel {rel.get((k, n))}")
