"""Table for set s: per (T, route): vLLM total and stages, each arm total, k1, ratio over vLLM total and over vLLM stages, err.
Means over all rounds; 'by cycle' columns check that cycle 1 and the reversed cycle 2 agree."""
import collections, json, sys
R = collections.defaultdict(list); ERR = {}
for fn in sys.argv[1:]:
    cyc = 0
    for l in open(fn):
        if not l.startswith("{"): continue
        d = json.loads(l)
        k = d.get("kind")
        if k == "time": R[(d["T"], d["route"], d["arm"])].append(d)
        elif k == "err": ERR.setdefault((d["T"], d["route"], d["arm"]), []).append(d)
m = lambda v, f=lambda x: x["tot_med"]: sum(f(x) for x in v) / len(v)
for Tr in sorted({k[:2] for k in R}, key=lambda x: (x[0], x[1])):
    vt, vs = R.get((*Tr, "vllm_total")), R.get((*Tr, "vllm_stages"))
    if not vt: print(Tr, "no vllm"); continue
    print(f"T={Tr[0]} {Tr[1]} vllm_total {m(vt):.4f} [{min(x['tot_med'] for x in vt):.4f}-{max(x['tot_med'] for x in vt):.4f}] vllm_stages_plain {m(vs):.4f} stage_sum {m(vs, lambda x: x['st_tot']):.4f} st {[round(m(vs, lambda x: x['stages'][i]), 3) for i in range(5)]}")
    rows = [(m(v), k[2], v) for k, v in R.items() if k[:2] == Tr and not k[2].startswith("vllm")]
    for t, arm, v in sorted(rows, key=lambda a: a[0]):
        h = len(v) // 2
        e = ERR.get((*Tr, arm), [{}])
        print(f"  {arm:14s} {t:7.4f} [{min(x['tot_med'] for x in v):.4f}-{max(x['tot_med'] for x in v):.4f}] n={len(v)} /tot={t / m(vt):.3f} /st={t / m(vs):.3f} c1={m(v[:h]) if h else 0:.4f} c2={m(v[h:]):.4f} k1={m(v, lambda x: x['stages'][1]):.3f} k2={m(v, lambda x: x['stages'][2]):.3f} sum={m(v, lambda x: x['stages'][3]):.3f} al={m(v, lambda x: x['stages'][0]):.3f} err={max(x.get('rel_vs_fp32', 0) for x in e)} nf={max(x.get('nonfinite', 0) for x in e)}")
