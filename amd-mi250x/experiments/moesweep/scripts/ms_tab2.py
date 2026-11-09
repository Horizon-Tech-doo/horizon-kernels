"""k1 and err tables from raw. args: run jsonl pair, then err jsonl pair."""
import json, sys, statistics as st, collections as C
def rows(fn):
    for l in open(fn):
        if l.startswith("{"):
            try: yield json.loads(l)
            except Exception: pass
for fn in sys.argv[1:3]:
    A = C.defaultdict(lambda: C.defaultdict(list)); E = {}
    for d in rows(fn):
        if d.get("kind") == "time": A[(d["T"], d["route"])][d["arm"]].append(d)
        if d.get("kind") == "err": E[(d["T"], d["route"], d["arm"])] = d
    print(fn); sp = []
    for k in sorted(A):
        a = A[k]; v = st.median(x["tot_med"] for x in a["vllm_total"]); vm = min(x["tot_min"] for x in a["vllm_total"])
        vs = [st.median(r["stages"][i] for r in a["vllm_stages"]) for i in range(5)]
        print(k, f"vllm {v:.3f} min {vm:.3f} stages {[round(x,3) for x in vs]}")
        for arm, r in sorted(a.items(), key=lambda it: st.median(x["tot_med"] for x in it[1])):
            m = [x["tot_med"] for x in r]; sp.append((max(m) - min(m)) / min(m))
            if arm.startswith("vllm"): continue
            s = [st.median(x["stages"][i] for x in r) for i in range(4)]
            e = E.get((k[0], k[1], arm), {})
            print(f"  {arm:9s} {st.median(m):.3f} med {st.median(m)/v:.4f} rounds {min(m)/v:.4f}-{max(m)/v:.4f} fast {min(x['tot_min'] for x in r)/vm:.4f} stages {[round(x,3) for x in s]} err {e.get('rel_vs_fp32')}")
    sp.sort(); print("spread p50 %.4f p90 %.4f max %.4f n=%d" % (sp[len(sp)//2], sp[int(len(sp)*.9)], sp[-1], len(sp)))
for fn in sys.argv[3:]:
    print(fn)
    for d in rows(fn):
        k = d.get("kind")
        if k == "err_all_T": print({x: d[x] for x in ("route","order","worst_rel","worst_T","worst_token_rel","worst_token_T","nonfinite","over_1e3","rel_T_ge_16","n")})
        if k == "err_per_T": print(" ", d["route"], "n", len(d["rel"]), "max rel by band", {b: round(max(d["rel"][lo-1:hi]),7) for b,(lo,hi) in {"1-15":(1,15),"16-256":(16,256),"257-2048":(257,2048),"2049-4096":(2049,4096)}.items()}, "max tok", round(max(d["tok"]),7))
    v = [d for d in rows(fn) if d.get("kind") == "vllm_err"]
    print("  vllm sampled", len(v), "worst rel", max(x["rel"] for x in v), "worst tok", max(x["tok"] for x in v), "T of worst", max(v, key=lambda x: x["rel"])["T"], max(v, key=lambda x: x["rel"])["route"])
