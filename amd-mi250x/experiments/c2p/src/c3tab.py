"""c3tab.py run.jsonl...: per job, per case, ratio to same-process CK and SDPA."""
import json, sys, collections
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    print("==", f)
    bypr = collections.defaultdict(dict)
    for r in rows:
        if "impl" not in r: continue
        p = (r["tag"], r["land"], r["case"], r["input"])
        if r["impl"] in ("ck", "sdpa"): bypr[p][r["impl"]] = r["tflops"]
    out = collections.defaultdict(list)
    for r in rows:
        if r.get("impl") != "tl": continue
        p = (r["tag"], r["land"], r["case"], r["input"])
        ck, sd = bypr[p].get("ck"), bypr[p].get("sdpa")
        if "err" in r:
            out[(r["case"], r["input"], r["name"])].append("ERR " + r["err"][-120:]); continue
        out[(r["case"], r["input"], r["name"])].append(f'{r["tflops"]:.1f} /CK {r["tflops"]/ck if ck else 0:.3f} /SD {r["tflops"]/sd if sd else 0:.2f} fp64 {r.get("rel_vs_fp64")} rep {r.get("repeat_equal")}')
    for k in sorted(out):
        print(k[0], k[1], k[2].split(k[0] + "_")[1], " | ".join(out[k]))
    for p, v in sorted(bypr.items()):
        print("  ref", p, v)
