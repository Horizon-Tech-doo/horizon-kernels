"""Summary of JPORT race files. Ratio: ours / same-process vendor (mean of pre and post), throughput, higher is better; new/old is TF over TF in the same cell.
Gate: new >= old in every cell (six inputs) at 2816, 3072, 3328, 3584, in both jobs. Usage: jsum.py race1.jsonl race2.jsonl"""
import json, sys
rows = []
for p in sys.argv[1:]:
    rows += [json.loads(l) for l in open(p) if l.startswith("{")]
info = [r for r in rows if r.get("info")]
for r in info:
    print("info", {k: r[k] for k in ("tag", "order", "ver", "clang", "gqa_fa")})
cells = {}
for r in rows:
    if "case" not in r:
        continue
    c = cells.setdefault((r["tag"], r["L"], r["input"]), {})
    if r["impl"] in ("ck", "sdpa"):
        c.setdefault(r["impl"], []).append(r["tflops"])
        if "rel_vs_fp64" in r:
            c["ck_err"] = r["rel_vs_fp64"]
    else:
        c[r["arm"]] = r
errs = [r for r in rows if r.get("err")]
print("errors", len(errs), [e["err"][:120] for e in errs[:3]])
mx = {}
gate_fail = []
tags = sorted({k[0] for k in cells})
LS = sorted({k[1] for k in cells})
print("L | tag | new/CK | old/CK | new/old | new/SDPA | jx/old | worst new/old input")
for L in LS:
    for t in tags:
        ks = [k for k in cells if k[0] == t and k[1] == L]
        nc, oc, no, ns, jo = [], [], [], [], []
        worst = None
        for k in ks:
            c = cells[k]
            ck = sum(c["ck"]) / len(c["ck"]); sd = sum(c["sdpa"]) / len(c["sdpa"])
            n, o = c["new"], c["old"]
            for a in ("new", "old", "jx"):
                if a in c:
                    x = c[a]
                    mx.setdefault((a, L), []).append(x.get("rel_vs_fp64", 9))
                    assert x.get("compiles", 1) == 0 and x["repeat_equal"] == "30/30" and x["nonfinite"] == 0, x
            nc.append(n["tflops"] / ck); oc.append(o["tflops"] / ck); no.append(n["tflops"] / o["tflops"]); ns.append(n["tflops"] / sd)
            if "jx" in c:
                jo.append(c["jx"]["tflops"] / o["tflops"])
            if worst is None or no[-1] < worst[0]:
                worst = (no[-1], k[2])
            if 2816 <= L <= 3584 and n["tflops"] < o["tflops"]:
                gate_fail.append((t, L, k[2], round(no[-1], 4)))
        f = lambda v: f"{min(v):.3f}-{max(v):.3f}" if v else "-"  # noqa: E731
        print(f"{L} | {t} | {f(nc)} | {f(oc)} | {f(no)} | {f(ns)} | {f(jo)} | {worst[1]} {worst[0]:.4f}")
print("rel_vs_fp64 max per arm and L:", {f"{a}_{L}": max(v) for (a, L), v in sorted(mx.items())})
print("ck rel_vs_fp64 max:", max(c.get("ck_err", 0) for c in cells.values()))
print("gate (new >= old every cell, 2816-3584, both jobs):", "PASS" if not gate_fail and len(tags) == 2 else f"FAIL {gate_fail}")
