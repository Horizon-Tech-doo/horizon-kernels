"""Pick refit on n3r4b raw (fit seeds 940-947, both jobs). For each shape: per-draw best form against r, and the table rule TABLE against old (kk) and the per-draw oracle. usage: ana_r4fit.py file..."""
import json, sys, statistics as st, collections as C
TABLE = {(32, 1024): ((0.82, "S5"), (0.56, "S3"), (0, "S4")), (64, 1024): ((0.70, "kk"), (0.58, "S3"), (0, "S4")), (32, 2048): ((0.82, "S5"), (0.70, "kk"), (0.58, "S3"), (0, "S4")),
         (64, 2048): ((0.70, "kk"), (0.58, "S6"), (0, "S4")), (16, 4096): ((0.84, "S10"), (0.58, "S6"), (0, "S8")), (8, 8192): ((0.82, "S10"), (0.72, "S12"), (0.5, "kk"), (0, "S24"))}
T = C.defaultdict(list); Rr = {}
for f in sys.argv[1:]:
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "seed": T[(r["B"], r["L"], r["mode"], r["seed"], r["impl"])].append(r["ms"]); Rr[(r["B"], r["L"], r["mode"], r["seed"])] = r["r"]
M = {k: st.median(v) for k, v in T.items()}
for sh in TABLE:
    D = sorted({k[2:4] for k in M if k[:2] == sh}, key=lambda d: -Rr[sh + d]); pick, old, orc = [], [], []; losses = []
    forms = sorted({k[4] for k in M if k[:2] == sh and k[4] != "tri"}); rows = []
    for d in D:
        r = Rr[sh + d]; tri = M[sh + d + ("tri",)]; f = next(x for lo, x in TABLE[sh] if r >= lo); best = min(forms, key=lambda x: M[sh + d + (x,)])
        p, o, b = tri / M[sh + d + (f,)], tri / M[sh + d + ("kk",)], tri / M[sh + d + (best,)]; pick.append(p); old.append(o); orc.append(b); rows.append((round(r, 2), d[0], f, round(p, 3), best, round(b, 3)))
    print(sh, f"pick med {st.median(pick):.3f} min {min(pick):.3f} wins {sum(x > 1 for x in pick)}/{len(pick)} | kk med {st.median(old):.3f} min {min(old):.3f} | oracle med {st.median(orc):.3f} | mean loss to oracle {st.mean(a / b for a, b in zip(orc, pick)) - 1:.3f}")
    if "-v" in sys.argv[0] or True:
        worst = sorted(rows, key=lambda x: x[3] / x[5])[:4]; print("   worst vs oracle:", worst)
