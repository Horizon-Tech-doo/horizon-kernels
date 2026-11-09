"""Explained negative, d128 8k and 16k on the image, randn inputs, processes from raw3 (round 3) as used for 12 of 51, plus round 4. Ratio: same-process TF, higher is better."""
import collections, glob, json, statistics as S
for rd in ("raw3", "raw4"):
    for case in ("d128_8192", "d128_16384"):
        P = collections.defaultdict(lambda: dict(s=[], cp=[], ca=[], t=collections.defaultdict(list)))
        for f in glob.glob(f"{rd}/*/run.jsonl"):
            rs = [json.loads(l) for l in open(f) if l.startswith("{")]
            if any(r.get("stack") == "lumi" for r in rs):
                continue
            for r in rs:
                if r.get("case") != case or not str(r.get("input", "")).startswith("randn") or "tflops" not in r:
                    continue
                g = P[f, r["tag"], r.get("flagset", ""), r["input"]]
                if r.get("impl") == "sdpa": g["s"].append(r["tflops"])
                elif r.get("impl") == "ck":
                    g["ca"].append(r["tflops"])
                    if r.get("slot") == "pre": g["cp"].append(r["tflops"])
                elif r.get("impl") == "tl": g["t"][r["name"]].append(r["tflops"])
        rows = [g for g in P.values() if g["s"] and g["cp"]]
        if not rows:
            continue
        cp = [g["cp"][0] / g["s"][0] for g in rows]
        cm = [S.mean(g["ca"]) / S.mean(g["s"]) for g in rows]
        print(rd, case, "processes", len(rows), "CK pre/first SDPA >=1.2:", sum(x >= 1.2 for x in cp), "CK mean/SDPA mean >=1.2:", sum(x >= 1.2 for x in cm), "CK/SDPA min med max", round(min(cm), 3), round(S.median(cm), 3), round(max(cm), 3))
        b = [(max(max(v) for v in g["t"].values()), S.mean(g["ca"]), S.mean(g["s"])) for g in rows if g["t"]]
        print("   best arm per process: /CK min med max", *[round(f([x[0] / x[1] for x in b]), 3) for f in (min, S.median, max)], " /SDPA", *[round(f([x[0] / x[2] for x in b]), 3) for f in (min, S.median, max)], "TF ours", *[round(f([x[0] for x in b]), 1) for f in (min, S.median, max)], "CK", *[round(f([x[1] for x in b]), 1) for f in (min, S.median, max)], "SDPA", *[round(f([x[2] for x in b]), 1) for f in (min, S.median, max)], "n", len(b), ">=1.2/SDPA", sum(x[0] / x[2] >= 1.2 for x in b))
