"""Summarise round 2 run.jsonl: per (case, input) median TF per form, ratio to same-process CK (ours / CK, higher is better) and to SDPA, per job and round."""
import json, sys, statistics as st, collections
for path in sys.argv[1:]:
    print("##", path)
    R = [json.loads(l) for l in open(path) if l.startswith("{") and "tflops" in l or '"err"' in l]
    by = collections.defaultdict(list)
    for r in R:
        by[(r.get("case"), r.get("input"), r.get("tag"))].append(r)
    agg = collections.defaultdict(list); chk = collections.defaultdict(list)
    for (c, i, t), rs in by.items():
        v = {}
        for r in rs:
            n = r["name"]
            if "err" in r: print("ERR", n, t, r["err"][-120:]); continue
            if r.get("impl") in ("ck", "sdpa"):
                agg[(c, i, r["impl"])].append((r["tflops"], None, None)); v.setdefault(r["impl"], []).append(r["tflops"])
        ck = st.median(v["ck"]) if v.get("ck") else None; sd = st.median(v["sdpa"]) if v.get("sdpa") else None
        for r in rs:
            if "err" in r or r.get("impl") != "tl": continue
            agg[(c, i, r["name"])].append((r["tflops"], r["tflops"] / ck if ck else None, r["tflops"] / sd if sd else None))
            chk[(c, i, r["name"])].append((r.get("rel_vs_fp64"), r.get("rel_vs_sdpa"), r.get("repeat_equal"), r.get("bitwise_same_as_first_form"), r.get("compiles")))
    for k in sorted(agg, key=str):
        a = agg[k]
        tf = [x[0] for x in a]
        ra = [x[1] for x in a if x[1]]; rs = [x[2] for x in a if x[2]]
        c = chk.get(k, [])
        print(f"{k[0]:11} {k[1]:7} {k[2]:48} TF {' '.join(f'{x:6.2f}' for x in tf)}  /CK {' '.join(f'{x:.3f}' for x in ra)}  /SDPA {' '.join(f'{x:.3f}' for x in rs)}  {c[0][:1] + c[0][2:] if c else ''}")
