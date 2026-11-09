"""All D128 tables from raw/. Ratios: ours / same-process vendor (mean of pre and post), throughput, higher is better. Run from evidence/D128."""
import collections, glob, json, statistics as st
IN = "randn0 randn1 self big ar ramp".split()
def load(p):
    P = collections.defaultdict(list); rows = [json.loads(l) for l in open(p)]
    for r in rows:
        if "impl" in r: P[(r["case"], r["input"])].append(r)
    return P
def vend(v):
    ck = [r["tflops"] for r in v if r["impl"] == "ck"]; sd = [r["tflops"] for r in v if r["impl"] == "sdpa"]
    return sum(ck) / len(ck), sum(sd) / len(sd), ck, sd
print("### 1. Short lengths, six inputs, jobs 23528761 (fwd), 23528762 (rev). /CK pre,post range across both")
S = {"761": load("raw/23528761/race.jsonl"), "762": load("raw/23528762/race.jsonl")}
for L in (2816, 3072, 3584, 4096):
    for arm in (f"d128_{L}_r128_b64_t512_JDRF", f"d128_{L}_r128_b32_t512_SDRF", f"d128_{L}_r128_b32_t512_hand"):
        o = []; lo = []; hi = []; hd = []; sd_ = []
        for i in IN:
            xs = []
            for j in S:
                v = S[j][(f"d128_{L}", i)]; a = [r for r in v if r["name"] == arm]
                if not a: continue
                c, s, ck, sd = vend(v); t = a[0]["tflops"]; xs.append(t / c); lo.append(t / max(ck)); hi.append(t / min(ck)); sd_.append(t / s)
                h = [r for r in v if r.get("kind") == "hand"]; hd.append(t / h[0]["tflops"])
            if xs: o.append(f"{i} {xs[0]:.3f},{xs[1]:.3f}")
        if o: print(f"{L} {arm[10:]:22s} " + " | ".join(o) + f" || single-vendor-read {min(lo):.3f}-{max(hi):.3f} /SDPA {min(sd_):.3f}-{max(sd_):.3f} /hand {min(hd):.3f}-{max(hd):.3f}")
print("### 2. 8k, 16k, six inputs, 6 processes, 4 jobs: /SDPA by input; order fwd or rev in the column label")
R = {"761f": "raw/23528761/race.jsonl", "762r": "raw/23528762/race.jsonl", "864r": "raw/23528864/race_rev.jsonl", "864f": "raw/23528864/race_fwd.jsonl", "865f": "raw/23528865/race_fwd.jsonl", "865r": "raw/23528865/race_rev.jsonl"}
D = {t: load(p) for t, p in R.items()}
for L in (8192, 16384):
    print(L, list(R))
    for arm in ("r128_b64_t512_JDRF", "r256_b32_t1024_SDRF", "r256_b32_t1024_JDRF", "r256_b32_t1024_hand"):
        a_s = []; a_c = []; a_h = []
        for i in IN:
            o = []
            for t in R:
                v = D[t][(f"d128_{L}", i)]; c, s, _, _ = vend(v); x = [r for r in v if r["name"] == f"d128_{L}_{arm}"][0]["tflops"]; h = [r for r in v if r.get("kind") == "hand"][0]["tflops"]
                o.append(x / s); a_s.append(x / s); a_c.append(x / c); a_h.append(x / h)
            print(f"  {arm:22s} {i:7s} " + " ".join(f"{z:.3f}" for z in o))
        print(f"  {arm:22s} /SDPA {min(a_s):.3f}-{max(a_s):.3f} at or over 1.2: {sum(z >= 1.2 for z in a_s)}/{len(a_s)}; /CK {min(a_c):.3f}-{max(a_c):.3f}; /hand {min(a_h):.3f}-{max(a_h):.3f}, median {st.median(a_h):.3f}")
    print("  CK/SDPA", " | ".join(i + " " + ",".join(f"{vend(D[t][(f'd128_{L}', i)])[0] / vend(D[t][(f'd128_{L}', i)])[1]:.3f}" for t in R) for i in IN))
print("### 3. Same bytes, wheel wrapper with 36864 B LDS (fork) against native. 23528864, 23528865")
for j in ("23528864", "23528865"):
    for k, v in sorted(load(f"raw/{j}/wport.jsonl").items()):
        c, s, _, _ = vend(v); a = {r["kind"]: r["tflops"] for r in v if r["impl"] == "tl"}
        print(j[-3:], k, a, f"hand/native {a['hand'] / a['native']:.3f} fork/native {a['fork'] / a['native']:.3f} fork/CK {a['fork'] / c:.3f} native/CK {a['native'] / c:.3f}")
print("### 4. Hand rows 128 at bN 48, 64; fork process and native wheel. 23529074 fwd, 23529075 rev")
for j in ("23529074", "23529075"):
    for k, v in sorted(load(f"raw/{j}/race.jsonl").items(), key=lambda x: (int(x[0][0].split('_')[1]), x[0][1])):
        c, s, _, _ = vend(v); a = {r["name"][5:]: r["tflops"] for r in v if r["impl"] == "tl"}
        J = [z for n, z in a.items() if "JDRF" in n][0]
        print(j[-3:], k, " ".join(f"{n.split('_', 1)[1]} {z:.2f} /CK {z / c:.3f}" for n, z in a.items()))
    for k, v in sorted(load(f"raw/{j}/wport2.jsonl").items(), key=lambda x: (int(x[0][0].split('_')[1]), x[0][1])):
        c, s, _, _ = vend(v); a = {r["bN"]: r["tflops"] for r in v if r["impl"] == "tl"}
        print(j[-3:], "native", k, " ".join(f"b{b} {a[b]:.2f} /CK {a[b] / c:.3f} /b32 {a[b] / a[32]:.3f}" for b in (32, 48, 64)))
print("### 5. Stock wheel, stub wrapper 32768 B, frozen bytes, against native. 23529190 fwd, 23529191 rev")
for j in ("23529190", "23529191"):
    for k, v in sorted(load(f"raw/{j}/wstub.jsonl").items()):
        c, s, ck, sd = vend(v); a = {r["kind"]: r for r in v if r["impl"] == "tl"}
        x, n = a["stub"]["tflops"], a["native"]["tflops"]
        print(j[-3:], k, f"stub {x} native {n} stub/CK {x / c:.3f} native/CK {n / c:.3f} stub/SDPA {x / s:.3f} native/SDPA {n / s:.3f} stub/native {x / n:.3f} fp64 {a['stub']['rel_vs_fp64']} sdpa {a['stub']['rel_vs_sdpa']} rep {a['stub']['repeat_equal']} comp {a['stub']['compiles']}")
print("### 6. Correctness and checks, all race files")
A = []
for p in glob.glob("raw/*/*.jsonl"):
    A += [json.loads(l) for l in open(p)]
T = [r for r in A if r.get("impl") == "tl"]
print("arms", len(T), "errors", sum("err" in r for r in T), "compiles>0", sum(r.get("compiles", 1) != 0 for r in T), "not all repeats equal", sum(r["repeat_equal"].split("/")[0] != r["repeat_equal"].split("/")[1] for r in T), "max fp64", max(r.get("rel_vs_fp64") or 0 for r in T), "max rel_vs_sdpa", max(r["rel_vs_sdpa"] for r in T))
print("CK max fp64", max(r.get("rel_vs_fp64") or 0 for r in A if r.get("impl") == "ck"))
