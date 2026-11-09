"""Flag sets against base, image stack, one job, processes rotate. Ratios: ours / same-process CK and flag set / base TF, higher is better."""
import json, sys, statistics as S, collections
rs = []
for f in sys.argv[1:]:
    rs += [json.loads(l) for l in open(f)]
rs = [r for r in rs if not r.get("info")]
ck = collections.defaultdict(list)
for r in rs:
    if r.get("impl") == "ck":
        ck[r["tag"], r["case"]].append(r["tflops"])
g = collections.defaultdict(list); er = collections.Counter(); ee = collections.defaultdict(float); rp = collections.defaultdict(set)
for r in rs:
    if r.get("impl") != "tl":
        continue
    k = (r["case"], r["name"].split(r["case"] + "_")[1], r["flagset"])
    if "tflops" in r:
        g[k].append((r["tflops"], r["tflops"] / S.mean(ck[r["tag"], r["case"]])))
        ee[k] = max(ee[k], r.get("rel_vs_fp64", 0) or r.get("rel_vs_sdpa", 0)); rp[k].add(r.get("repeat_equal"))
    else:
        er[k] += 1
base = {k[:2]: S.median(x[0] for x in v) for k, v in g.items() if k[2] == "base"}
for k in sorted(g):
    v = g[k]
    print(*k, len(v), round(S.median(x[0] for x in v), 2), "ck", round(S.median(x[1] for x in v), 3), "/base", round(S.median(x[0] for x in v) / base[k[:2]], 3), f"e {ee[k]:.2e}", rp[k])
print("errors", dict(er))
