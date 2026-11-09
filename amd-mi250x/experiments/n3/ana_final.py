"""Summary ranges over the round 3 jobs. Ratio vLLM ms / our ms."""
import json, statistics as st, collections as C
def load(fs, keyf, t="time"):
    T = C.defaultdict(list)
    for f in fs:
        for l in open(f):
            if not l.startswith("{"): continue
            try: r = json.loads(l)
            except Exception: continue
            if r.get("t") == t and r.get("stack") == "image": T[(f,) + keyf(r)].append(r["ms"])
    return {k: st.median(v) for k, v in T.items()}
# N=784
fs = ["raw/r3h-23508589.jsonl", "raw/r3h-23508633.jsonl"]
M = load(fs, lambda r: (r["dt"], r["B"], r["L"], r["mode"], r["impl"]))
cells = {k[:-1] for k in M}
for dt in ("bf16", "fp16"):
    c = [k for k in cells if k[1] == dt]
    for a, b in (("rh", "roc"), ("n", "roc"), ("n", "tri"), ("rh", "tri"), ("tri", "roc")):
        x = [M[k + (b,)] / M[k + (a,)] for k in c]; print(f"page784 {dt} {b}/{a}: {min(x):.2f}..{max(x):.2f} n={len(x)}")
    worst = sorted(c, key=lambda k: M[k + ("tri",)] / M[k + ("n",)])[:5]; print(" lowest tri/n", [(k[0][-13:-6],) + k[2:] + (round(M[k + ("tri",)] / M[k + ("n",)], 3),) for k in worst])
# N=16 r3a
fs = ["raw/r3a-23508444.jsonl", "raw/r3a-23508496.jsonl"]
M = load(fs, lambda r: (r["B"], r["L"], r["mode"], r["impl"])); c = {k[:-1] for k in M}
for a, b in (("land_r", "roc"), ("land_n", "tri"), ("land_b", "uni"), ("land_bs", "uni"), ("land_r", "tri"), ("land_n", "roc")):
    x = [M[k + (b,)] / M[k + (a,)] for k in c]; print(f"page16 {b}/{a}: {min(x):.2f}..{max(x):.2f}")
x = [M[k + ("land_b",)] / M[k + ("land_bs",)] for k in c]; print("page16 land_b/land_bs", round(min(x), 3), round(max(x), 3))
# pick
for fs in (["raw/r3q-23508676.jsonl"], ["raw/r3q-23508677.jsonl"]):
    M = load(fs, lambda r: (r["B"], r["L"], r["N"], r["mode"], r["seed"], r["impl"]), "seed")
    G = C.defaultdict(list)
    for k, v in M.items():
        if k[-1] == "pick":
            G[k[1:5]].append((M[k[:-1] + ("tri",)] / v, M[k[:-1] + ("roc",)] / v, M[k[:-1] + ("old",)] / v, M[k[:-1] + ("tri",)] / M[k[:-1] + ("old",)]))
    print(fs[0][-13:-6])
    for k in sorted(G):
        a = [z[0] for z in G[k]]; print(" ", k, f"pick/tri med {st.median(a):.2f} [{min(a):.2f}..{max(a):.2f}] wins {sum(q > 1 for q in a)}/{len(a)} old/tri {st.median(z[3] for z in G[k]):.2f} gain {st.median(z[2] for z in G[k]):.3f} pick/roc {st.median(z[1] for z in G[k]):.2f}")
