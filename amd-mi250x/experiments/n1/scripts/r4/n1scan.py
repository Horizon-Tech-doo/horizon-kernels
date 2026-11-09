"""Ratio table of a T scan job: per T and routing, vLLM ms, ours on each stack, ratio over vLLM total (same job). Ratio: ours over vLLM, time, lower is better."""
import json, collections, sys
R = collections.defaultdict(list); E = {}
for l in open(sys.argv[1]):
    if not l.startswith("{"): continue
    try: d = json.loads(l)
    except Exception: continue
    k = d.get("kind")
    if k == "time": R[(d["T"], d["route"], d["stack"], d["arm"])].append(d["tot_med"])
    if k == "err": E[(d["T"], d["route"], d["stack"], d["arm"])] = (d["rel_vs_fp32"], d["nonfinite"])
def m(k):
    v = R.get(k)
    return sum(v) / len(v) if v else float("nan")
Ts = sorted({k[0] for k in R})
rts = ("uniform", "zipf1", "zipf2")
print("T | routing: vLLM ms, image ms ratio, lumi ms ratio")
over = []
for T in Ts:
    row = []
    for r in rts:
        v, a, b = m((T, r, "vllm", "vllm_total")), m((T, r, "vllm", "ship")), m((T, r, "lumi", "ship"))
        row.append(f"{r} {v:.3f} {a:.3f} {a/v:.3f} {b:.3f} {b/v:.3f}")
        if a / v > 1 or b / v > 1: over.append((T, r, round(a / v, 3), round(b / v, 3)))
    print(T, " | ".join(row))
print("over 1.0:", over)
print("A_u20k against ship, our stack, ms:")
for T in Ts:
    if 257 <= T <= 511:
        print(T, [f"{m((T,r,'lumi','ship')):.3f}/{m((T,r,'lumi','A_u20k')):.3f}" for r in rts])
ou = {k: e for k, e in E.items() if k[3] not in ("vllm_total", "vllm_stages")}
vl = {k: e for k, e in E.items() if k[3] in ("vllm_total", "vllm_stages")}
print("checks ours", len(ou), "worst", max(e[0] for e in ou.values()) if ou else None, "nonfinite", sum(e[1] for e in ou.values()))
print("checks vLLM", len(vl), "worst", max(e[0] for e in vl.values()) if vl else None, "nonfinite", sum(e[1] for e in vl.values()))
ser = [(k[0], k[1], k[2], len(v)) for k, v in R.items() if k[3] == "ship"]
print("rounds per (ship): min", min(s[3] for s in ser) if ser else None, "max", max(s[3] for s in ser) if ser else None)
