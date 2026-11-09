"""Round 4 tables. Ratio: ours / same-process vendor, TF, higher is better. CK = mean of pre and post reads when both exist; SDPA likewise, and the pre-only ratio is printed too. One row per (job, tag). Usage: r4_table.py job1.jsonl[:J1] ..."""
import collections, json, statistics as S, sys
rs = []
for f in sys.argv[1:]:
    j = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if not r.get("info"):
            r["job"] = j
            rs.append(r)
K = lambda r: (r["job"], r["tag"], r["case"], r["input"])  # noqa: E731
ven = collections.defaultdict(dict)
for r in rs:
    if r.get("impl") in ("ck", "sdpa"):
        ven[K(r)][r["impl"], r["slot"]] = r["tflops"]
g = collections.defaultdict(list); er = collections.defaultdict(float); rp = collections.defaultdict(set); bad = []
for r in rs:
    if r.get("impl") != "tl":
        continue
    if "tflops" not in r:
        bad.append((r["job"], r["tag"], r["name"], r.get("err", "")[-120:])); continue
    v = ven[K(r)]
    ck = S.mean(x for (i, _), x in v.items() if i == "ck") if any(i == "ck" for i, _ in v) else None
    sd = S.mean(x for (i, _), x in v.items() if i == "sdpa")
    k = (r["case"], r["input"], r["name"].split(r["case"] + "_")[1], r.get("flagset", ""))
    g[k].append(dict(job=r["job"], tf=r["tflops"], ck=r["tflops"] / ck if ck else None, sd=r["tflops"] / sd, sdpre=r["tflops"] / v["sdpa", "pre"], vck=ck, vsd=sd, comp=r.get("compiles")))
    er[k] = max(er[k], r.get("rel_vs_fp64", 0) or r.get("rel_vs_sdpa", 0)); rp[k].add(r.get("repeat_equal"))
    if r.get("bitwise_same_as_first") is False:
        bad.append((r["job"], r["tag"], r["name"], "bitwise differs"))
fmt = lambda xs: f"{min(xs):.3f}-{max(xs):.3f}" if xs else "-"  # noqa: E731
key = lambda k: (k[0].split("_")[0], int(k[0].split("_")[1]), k[1], k[2], k[3])  # noqa: E731
print("case input arm flagset n | TF med | /CK min-max med | /SDPA min-max med | vendor CK, SDPA TF med | err repeats")
for k in sorted(g, key=key):
    v = g[k]
    c = [x["ck"] for x in v if x["ck"]]
    print(*k, len(v), "|", round(S.median(x["tf"] for x in v), 2), "|", fmt(c), round(S.median(c), 3) if c else "", "|", fmt([x["sd"] for x in v]), round(S.median(x["sd"] for x in v), 3), "|",
          round(S.median(x["vck"] for x in v), 1) if c else "", round(S.median(x["vsd"] for x in v), 1), "|", f"{er[k]:.2e}", sorted(rp[k], key=str), "compiled" if any(x["comp"] for x in v) else "")
print("bad", bad)
