"""Summary from raw jobs. Ratio: ours / same-process vendor (mean of pre and post), TF, higher is better. Usage: sum.py raw/<job> ..."""
import collections, json, statistics as st, sys
R = {}
for d in sys.argv[1:]:
    R[d.split("/")[-1]] = [json.loads(x) for x in open(d + "/race.jsonl")]
J = list(R)
T = collections.defaultdict(dict); LEAD = collections.defaultdict(dict); ALL = []; V = collections.defaultdict(dict)
for j, rs in R.items():
    P = collections.defaultdict(list)
    for r in rs:
        if "impl" in r:
            P[(r["case"], r["input"])].append(r)
    for (case, inp), x in P.items():
        ck = {r["slot"]: r["tflops"] for r in x if r["impl"] == "ck"}; sd = {r["slot"]: r["tflops"] for r in x if r["impl"] == "sdpa"}
        ckm, sdm = sum(ck.values()) / len(ck), sum(sd.values()) / len(sd)
        V[(case, inp)][j] = dict(ck=ck, sd=sd, cks=ckm / sdm, ckerr=[r.get("rel_vs_fp64") for r in x if r["impl"] == "ck" and r["slot"] == "pre"][0], ckerrs=[r.get("rel_vs_sdpa") for r in x if r["impl"] == "ck" and r["slot"] == "pre"][0])
        arms = [r for r in x if r["impl"] == "tl"]; ALL += arms
        hand = [r for r in arms if r["kind"] == "hand" and "tflops" in r]
        for r in arms:
            if "tflops" not in r:
                print("ERR", j, r["name"], inp, r.get("err")); continue
            T[(case, r["name"], inp)][j] = dict(tf=r["tflops"], ck=r["tflops"] / ckm, ckpre=r["tflops"] / ck["pre"], ckpost=r["tflops"] / ck["post"], sd=r["tflops"] / sdm, sdpre=r["tflops"] / sd["pre"], sdpost=r["tflops"] / sd["post"], hand=r["tflops"] / hand[0]["tflops"] if hand else None, fp64=r.get("rel_vs_fp64"), rs=r.get("rel_vs_sdpa"))
for key in sorted(T, key=lambda k: (int(k[0].split("_")[1]), k[1], k[2])):
    d = T[key]
    print(f"{key[1]:34s} {key[2]:7s} " + " | ".join(f"{j[-3:]} TF {d[j]['tf']:6.2f} /CK {d[j]['ck']:.3f} [{d[j]['ckpre']:.3f},{d[j]['ckpost']:.3f}] /SDPA {d[j]['sd']:.3f} [{d[j]['sdpre']:.3f},{d[j]['sdpost']:.3f}] /hand {d[j]['hand']:.3f} fp64 {d[j]['fp64']} sd {d[j]['rs']}" for j in J if j in d))
print("-- vendors")
for k in sorted(V, key=lambda k: (int(k[0].split("_")[1]), k[1])):
    print(k[0], k[1], " | ".join(f"{j[-3:]} CK {V[k][j]['ck']} SDPA {V[k][j]['sd']} CK/SDPA {V[k][j]['cks']:.3f} CKerr fp64 {V[k][j]['ckerr']} sdpa {V[k][j]['ckerrs']}" for j in J if j in V[k]))
print("arms", len(ALL), "errors", sum("err" in r for r in ALL), "compiles>0", sum(r.get("compiles", 1) != 0 for r in ALL), "bad_repeat", sum(r.get("repeat_equal", "x/y").split("/")[0] != r.get("repeat_equal", "x/y").split("/")[1] for r in ALL), "nonfinite", sum(r.get("nonfinite", 0) for r in ALL))
for kind in ("form", "hand"):
    x = [r for r in ALL if r["kind"] == kind and "tflops" in r]
    print(kind, "max fp64", max((r.get("rel_vs_fp64") or 0) for r in x), "max rel_vs_sdpa", max(r["rel_vs_sdpa"] for r in x))
print("-- range by arm, all inputs: /CK, /SDPA, /hand; pre/post extremes")
A = collections.defaultdict(list)
for (c, n, i), d in T.items():
    for j, v in d.items():
        A[n].append(v)
for n in sorted(A, key=lambda n: (int(n.split("_")[1]), n)):
    v = A[n]
    print(f"{n:34s} n {len(v):2d} /CK {min(x['ck'] for x in v):.3f}-{max(x['ck'] for x in v):.3f} /SDPA {min(x['sd'] for x in v):.3f}-{max(x['sd'] for x in v):.3f} over 1.2: {sum(x['sd'] >= 1.2 for x in v)} /hand {min(x['hand'] for x in v):.3f}-{max(x['hand'] for x in v):.3f} median /hand {st.median(x['hand'] for x in v):.3f}")
