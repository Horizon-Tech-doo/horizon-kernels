"""Sweep table: ours over vLLM by median of per-round medians and by fastest call, per job; round-median spread as the quiet test (rule p90 under 1%). args: run jsonl files."""
import json, sys, statistics as st, collections as C
for fn in sys.argv[1:]:
    A = C.defaultdict(lambda: C.defaultdict(list)); calls = C.defaultdict(lambda: C.defaultdict(list)); cfg = {}; er = {}
    for l in open(fn):
        if not l.startswith("{"): continue
        d = json.loads(l)
        if d.get("kind") == "time":
            A[d["T"]][d["arm"]].append(d)
            calls[d["T"]][d["arm"]].append(d["calls"]) if "calls" in d else None
        if d.get("kind") == "ship_cfg": cfg[d["T"]] = d["cfg"]
        if d.get("kind") == "err": er[(d["T"], d["arm"])] = d
    sp = []; rows = []; print(fn)
    for T in sorted(A):
        a = A[T]
        for arm, v in a.items():
            m = [x["tot_med"] for x in v]; sp.append((max(m) - min(m)) / min(m))
        v = [x["tot_med"] for x in a["vllm_total"]]; s = [x["tot_med"] for x in a["ship"]]
        rm, rf = st.median(s) / st.median(v), min(x["tot_min"] for x in a["ship"]) / min(x["tot_min"] for x in a["vllm_total"])
        pr = [x / y for x, y in zip(s, v)]
        nov = sum(1 for x in pr if x > 1)
        rows.append((T, rm, rf))
        print(f"T {T} vllm {st.median(v):.4f} ours {st.median(s):.4f} ratio_med {rm:.4f} ratio_min {rf:.4f} round_ratio {min(pr):.4f}-{max(pr):.4f} rounds_over1 {nov}/{len(pr)} err ours {er.get((T,'ship'),{}).get('rel_vs_fp32')} vllm {er.get((T,'vllm_total'),{}).get('rel_vs_fp32')} nf {er.get((T,'ship'),{}).get('nonfinite')}")
    sp.sort()
    print("peak med", max(rows, key=lambda r: r[1])[:2], "peak fastest", max(rows, key=lambda r: r[2])[0], round(max(r[2] for r in rows), 4), "over1 med", [r[0] for r in rows if r[1] > 1], "over1 min", [r[0] for r in rows if r[2] > 1])
    print("spread p50 %.4f p90 %.4f max %.4f n=%d" % (sp[len(sp) // 2], sp[int(len(sp) * .9)], sp[-1], len(sp)))
    print("cfgs", {json.dumps(c, sort_keys=True) for c in cfg.values()})
