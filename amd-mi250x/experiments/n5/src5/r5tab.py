"""Round 5 tables from raw5/<job>/run.jsonl. Ratio: ours / same-process vendor mean of pre and post, TF, higher is better. Usage: r5tab.py grid J1 J2 | val J1 J2 | attr J1 J2 | sweep J | conf J1 J2"""
import collections, json, statistics as S, sys
mode, jobs = sys.argv[1], sys.argv[2:]


def load(j):
    return [json.loads(l) for l in open(f"raw5/{j}/run.jsonl") if l.startswith("{")]


def groups(rs):
    g = collections.defaultdict(lambda: dict(ck=[], sdpa=[], tl=[]))
    for r in rs:
        if "case" not in r or "tflops" not in r or r.get("impl") not in ("ck", "sdpa", "tl", "sdpa_end"):
            continue
        g[r["tag"], r["case"], r["input"]]["sdpa" if r["impl"] == "sdpa_end" else r["impl"]].append(r)
    return g


def tflop(xs):
    return S.mean(x["tflops"] for x in xs) if xs else float("nan")


T = collections.defaultdict(lambda: collections.defaultdict(list))
errs = []
for j in jobs:
    rs = load(j)
    errs += [(j, r["name"], r["err"][-120:]) for r in rs if r.get("err")]
    for (tag, case, inp), v in groups(rs).items():
        ck, sd = tflop(v["ck"]), tflop(v["sdpa"])
        for r in v["tl"]:
            T[case, inp, r["name"]][j].append(dict(tf=r["tflops"], ck=r["tflops"] / ck if v["ck"] else None, sd=r["tflops"] / sd, ckr=ck / sd if v["ck"] else None, e=r.get("rel_vs_fp64"), es=r.get("rel_vs_sdpa"), rep=r.get("repeat_equal"), bit=r.get("bitwise_same_as_first"), comp=r.get("compiles"), pos=r.get("pos")))
print("errors", errs)
for k in sorted(T, key=lambda k: (k[0], k[1], -S.median(x["ck"] or x["sd"] for j in T[k] for x in T[k][j]))):
    cells = []
    for j in jobs:
        xs = T[k].get(j, [])
        if xs:
            key = "ck" if xs[0]["ck"] is not None else "sd"
            cells.append(f"{key} " + "/".join(f"{x[key]:.3f}" for x in xs) + " tf " + "/".join(f"{x['tf']:.1f}" for x in xs))
    print(*k, " | ".join(cells))
allx = [x for k in T for j in T[k] for x in T[k][j]]
ef = [x["e"] for x in allx if x["e"] is not None]; es = [x["es"] for x in allx if x["es"] is not None]
print("rows", len(allx), "max fp64", max(ef, default=None), "max rel_sdpa", max(es, default=None), "compiles", sum(x["comp"] or 0 for x in allx), "bad repeats", [x["rep"] for x in allx if x["rep"] and x["rep"].split("/")[0] != x["rep"].split("/")[1]], "bit False", sum(1 for x in allx if x["bit"] is False))
