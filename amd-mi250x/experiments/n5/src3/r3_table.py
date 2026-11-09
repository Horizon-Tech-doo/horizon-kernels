"""Per case and arm: TF median over rounds, ratio to same-process CK (mean of pre and post, median across rounds), ratio to same-process SDPA, ratio to the s0 arm of the same round, worst error, repeats, bitwise. Ratios: ours / vendor, throughput, higher is better."""
import json, sys, statistics as S, collections
rows = []
for f in sys.argv[1:]:
    job = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if r.get("info"):
            continue
        r["job"] = job; rows.append(r)
g = collections.defaultdict(list)
for r in rows:
    g[(r["job"], r["tag"], r["case"], r["input"])].append(r)
out = collections.defaultdict(lambda: collections.defaultdict(list))
ck_ = {}
for key, rs in g.items():
    ck = [r["tflops"] for r in rs if r["impl"] == "ck" and "tflops" in r]
    sd = [r["tflops"] for r in rs if r["impl"] == "sdpa" and "tflops" in r]
    base = {}
    for r in rs:
        if r["impl"] == "tl" and r.get("stride") == 0 and "tflops" in r:
            base[(r["P"], r["bN"], r["th"], r["rows"])] = r["tflops"]
    if ck:
        ck_.setdefault((key[2], key[3]), []).append(ck)
    for r in rs:
        if r["impl"] != "tl":
            continue
        k = (key[2], key[3], r["name"].split(key[2] + "_")[1])
        d = out[k]
        if "err" in r:
            d["err"].append(r["err"][-100:]); continue
        d["tf"].append(r["tflops"]); d["pos"].append(r.get("pos"))
        d["job"].append(key[0])
        if ck: d["ck"].append(r["tflops"] / S.mean(ck)); d["ckpre"].append(r["tflops"] / ck[0])
        if sd: d["sd"].append(r["tflops"] / sd[0])
        b = base.get((r["P"], r["bN"], r["th"], r["rows"]))
        if b: d["s0"].append(r["tflops"] / b)
        for e in ("rel_vs_fp64", "rel_vs_sdpa"):
            if e in r: d[e].append(r[e])
        if "repeat_equal" in r: d["rep"].append(r["repeat_equal"])
        if "bitwise_same_as_first" in r: d["bit"].append(r["bitwise_same_as_first"])
f3 = lambda x: f"{min(x):.3f}-{max(x):.3f}" if x else ""
print("case input arm n TF(med) /CK(med range) /SDPA /s0 err rep bit")
for k in sorted(out):
    d = out[k]
    if d["err"]:
        print(k, "ERR", d["err"][0]); continue
    print(f"{k[0]:12s} {k[1]:7s} {k[2]:34s} {len(d['tf'])} {S.median(d['tf']):7.2f} ck {S.median(d['ck']) if d['ck'] else 0:.3f} [{f3(d['ck'])}] sd {S.median(d['sd']) if d['sd'] else 0:.3f} [{f3(d['sd'])}] s0 {S.median(d['s0']) if d['s0'] else 0:.3f} [{f3(d['s0'])}] e {max(d['rel_vs_fp64'] or d['rel_vs_sdpa'] or [0]):.2e} rep {set(d['rep']) if len(set(d['rep']))<3 else 'mixed'} bit {set(d['bit'])}")
print("CK pre post by process:")
for k, v in sorted(ck_.items()):
    print(k, [tuple(x) for x in v])
