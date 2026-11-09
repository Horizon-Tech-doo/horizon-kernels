"""Delivered clock and power per timed window, from gpu_metrics (pollm.py, 50 ms) joined on t0, t1 in run.jsonl. Our card is the one with highest activity inside windows. Usage: r3_power.py <job dir>."""
import json, sys, statistics as S, collections, bisect
d = sys.argv[1]
M = [json.loads(l) for l in open(d + "/metrics.jsonl") if l.endswith("}\n")]
T = [m["t"] for m in M]
R = [json.loads(l) for l in open(d + "/run.jsonl")]
R = [r for r in R if r.get("t0") and "tflops" in r]
cards = [k for k in M[0] if k.startswith("card") and "act" in M[0][k]]
win = lambda r: M[bisect.bisect_left(T, r["t0"] + 0.3): bisect.bisect_right(T, r["t1"])]
act = {c: S.mean(m[c]["act"] for r in R for m in win(r)) for c in cards}
ours = max(act, key=act.get)
twin = "card%d" % (int(ours[4:]) ^ 1)
print("ours", ours, "act", round(act[ours], 1), "samples", len(M))
g = collections.defaultdict(lambda: collections.defaultdict(list))
for r in R:
    n = r["name"].replace(r["case"] + "_", "") + ("/" + r["slot"] if r.get("slot") else "")
    k = (r["case"], r["input"], n)
    g[k]["tf"].append(r["tflops"])
    for m in win(r):
        c = m[ours]
        for f in ("agfx", "cgfx", "act", "umc", "th"):
            g[k][f].append(c[f])
        g[k]["pw"].append(m[ours if int(ours[4:]) % 2 == 0 else twin]["pw"])
        g[k]["thr"].append(c["thr"])
print("case input arm n TF agfx(med) cgfx act umc pkgW hot thr")
for k in sorted(g):
    v = g[k]
    if not v["agfx"]:
        continue
    md = lambda f: round(S.median(v[f]), 1)
    print(*k, len(v["tf"]), md("tf"), md("agfx"), md("cgfx"), md("act"), md("umc"), md("pw"), md("th"), sorted(set(v["thr"]))[:3])
