"""Summary: per (T, route, arm, stack) mean of per-round medians, range, stages."""
import collections, json, sys
R = collections.defaultdict(list); ERR = {}; RT = {}; F = []; CFG = {}
for fn in sys.argv[1:]:
    for l in open(fn):
        if not l.startswith("{"): continue
        try: d = json.loads(l)
        except Exception: continue
        k = d.get("kind")
        if k == "time": R[(d["T"], d["route"], d["stack"], d["arm"])].append(d)
        elif k == "err": ERR[(d["T"], d["route"], d["stack"], d["arm"])] = d
        elif k == "routing": RT[(d["T"], d["route"])] = d
        elif k == "vllm_config": CFG[d["T"]] = d["cfg"]
        elif k and "failed" in k: F.append((d.get("T"), d.get("arm"), d.get("err", "")[-160:]))
for T in sorted({k[0] for k in R}, reverse=True):
    print("T", T, "vllm_cfg", {k: v for k, v in CFG.get(T, {}).items() if k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "NUM_WARPS", "num_warps", "num_stages")})
    for rt in sorted({k[1] for k in R if k[0] == T}):
        r = RT.get((T, rt), {})
        print(" route", rt, "hit", r.get("experts_hit"), "top1", r.get("top1_share"), "blk64", r.get("blocks64"), "blk16", r.get("blocks16"))
        v = R.get((T, rt, "vllm", "vllm_total"))
        vm = sum(x["tot_med"] for x in v) / len(v) if v else None
        if vm: print(f"   vllm_total {vm:.4f} range {min(x['tot_med'] for x in v):.4f}-{max(x['tot_med'] for x in v):.4f} n={len(v)}")
        rows = [(sum(x["tot_med"] for x in vs) / len(vs), k, vs) for k, vs in R.items() if k[0] == T and k[1] == rt]
        for m, k, vs in sorted(rows, key=lambda a: a[0]):
            st = vs[0].get("stages")
            ss = " ".join(f"{sum(x['stages'][i] for x in vs) / len(vs):.3f}" for i in range(len(st))) if st else ""
            e = ERR.get(k, {})
            print(f"   {k[2]:4s} {k[3]:16s} {m:8.4f} [{min(x['tot_med'] for x in vs):.4f}-{max(x['tot_med'] for x in vs):.4f}] n={len(vs)} ratio={(m / vm if vm else 0):.3f} err={e.get('rel_vs_fp32')} nf={e.get('nonfinite')} st={ss}")
if F: print("FAILED", F[:12])
