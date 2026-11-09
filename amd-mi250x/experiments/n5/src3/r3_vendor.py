"""CK / SDPA, same process, by case and input: the vendor's own ratio. Higher is better for CK."""
import json, sys, glob, collections, statistics as S
g = collections.defaultdict(list)
for f in glob.glob("raw3/*/run.jsonl"):
    rs = [json.loads(l) for l in open(f)]
    if any(r.get("stack") == "lumi" for r in rs):
        continue
    t = collections.defaultdict(dict)
    for r in rs:
        if r.get("impl") == "sdpa": t[r["tag"], r["case"], r["input"]]["s"] = r["tflops"]
        if r.get("impl") == "ck" and r.get("slot") == "pre": t[r["tag"], r["case"], r["input"]]["c"] = r["tflops"]
    for k, v in t.items():
        if len(v) == 2:
            g[k[1], k[2]].append(v["c"] / v["s"])
for k in sorted(g):
    if k[0].startswith(("d128_", "d256_", "mha_")):
        print(*k, len(g[k]), round(min(g[k]), 3), round(S.median(g[k]), 3), round(max(g[k]), 3))
