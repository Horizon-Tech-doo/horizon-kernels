"""Round 6 summary from raw6. Ratio: ours / same-process vendor (mean of pre and post), TF, higher is better."""
import collections, glob, json, statistics as st
J = ["23526062", "23526063", "23526194", "23526295"]
def load(j, f):
    try:
        return [json.loads(x) for x in open(f"raw6/{j}/{f}")]
    except FileNotFoundError:
        return []
res = collections.defaultdict(lambda: collections.defaultdict(list)); allr = []
for j in J:
    for f in ("forms.jsonl", "tile.jsonl", "tileb.jsonl", "l18.jsonl"):
        P = collections.defaultdict(list)
        for r in load(j, f):
            if "impl" in r:
                P[(r["tag"], r["case"], r["input"])].append(r)
        for (tag, case, inp), rs in P.items():
            ck = [r["tflops"] for r in rs if r["impl"] == "ck"]; sd = [r["tflops"] for r in rs if r["impl"].startswith("sdpa")]
            for r in rs:
                if r["impl"] != "tl":
                    continue
                allr.append(r)
                d = res[(f, r["name"], inp)]
                d[j].append((r["tflops"], r["tflops"] / (sum(ck) / len(ck)) if ck else None, r["tflops"] / (sum(sd) / len(sd)), sum(ck) / len(ck) if ck else None, sum(sd) / len(sd)))
def rng(x):
    return f"{min(x):.3f}" if min(x) == max(x) else f"{min(x):.3f}-{max(x):.3f}"
for key in sorted(res, key=lambda k: (k[0], int(k[1].split("_")[1]), k[2], k[1])):
    d = res[key]
    print(key[0][:5], key[1], key[2], " ".join(f"{j[-3:]}: TF {rng([v[0] for v in d[j]]).replace('.000','')} CK {rng([v[1] for v in d[j]]) if d[j][0][1] else '-'} SDPA {rng([v[2] for v in d[j]])}" for j in J if j in d))
print("arms", len(allr), "errors", sum("err" in r for r in allr), "compiles>0", sum(r.get("compiles", 1) != 0 for r in allr), "bad_repeat", sum(r["repeat_equal"].split("/")[0] != r["repeat_equal"].split("/")[1] for r in allr), "missing_check", sum("rel_vs_sdpa" not in r for r in allr))
print("max fp64", max(r.get("rel_vs_fp64", 0) for r in allr), "max rel_vs_sdpa", max(r["rel_vs_sdpa"] for r in allr))
for inp in ("randn0", "ar", "big"):
    x = [r for r in allr if r["input"] == inp]
    print(inp, "max fp64", max(r.get("rel_vs_fp64", 0) for r in x), "max sdpa", max(r["rel_vs_sdpa"] for r in x), "arms", len(x))
# hand: lead of rows 256 over the table's pick (same process, TF)
print("-- rows 256 over table pick, same process, TF")
for f, Ls in (("tile.jsonl", {3072: "r128_b32_t512", 4096: "r128_b32_t512", 8192: "P32_b96_t768", 16384: "P32_b96_t768"}), ("tileb.jsonl", {5120: "r128_b32_t512", 6144: "r128_b32_t512", 10240: "P32_b96_t768", 12288: "P32_b96_t768"})):
    for L, pk in Ls.items():
        o = []
        for j in J:
            T = collections.defaultdict(dict)
            for r in load(j, f):
                if r.get("impl") == "tl" and r["case"] == f"d128_{L}":
                    T[r["tag"]][r["name"]] = r["tflops"]
            o += [f"{j[-3:]}/{t[-2:]} {100 * (v[f'd128_{L}_r256_b32_t1024'] / v[f'd128_{L}_{pk}'] - 1):+.2f}%" for t, v in T.items()]
        print(L, pk, " ".join(o))
print("-- clang 18 rows 128 lb4 over P32, same process, TF")
for L in (2048, 2816, 3072, 4096, 4160):
    o = []
    for j in J:
        v = {r["name"]: r["tflops"] for r in load(j, "l18.jsonl") if r.get("impl") == "tl" and r["case"] == f"d128_{L}"}
        if v:
            o.append(f"{j[-3:]} {v[f'd128_{L}_r128_b32_t512_lb4'] / v[f'd128_{L}_P32_b96_t768']:.4f}")
    print(L, " ".join(o))
print("-- vendors and negative, image, randn0: same-process TF")
for L in (8192, 16384):
    for j in J[:2]:
        for tag in ("t1", "t2"):
            rs = [r for r in load(j, "tile.jsonl") if r.get("case") == f"d128_{L}" and r["tag"].endswith(tag) and "impl" in r]
            ck = st.mean(r["tflops"] for r in rs if r["impl"] == "ck"); sd = st.mean(r["tflops"] for r in rs if r["impl"] == "sdpa")
            a = max(r["tflops"] for r in rs if r["impl"] == "tl"); pp = [r["tflops"] for r in rs if r["name"].endswith("P32_b96_t768")][0]
            print(L, j[-3:], tag, f"ours r256 {a} CK {ck:.2f} SDPA {sd:.2f} ours/CK {a / ck:.3f} CK/SDPA {ck / sd:.3f} ours/SDPA {a / sd:.3f} gain to CK {100 * (ck / a - 1):.1f}% target {1.2 * sd:.1f} gain to 1.2 {100 * (1.2 * sd / a - 1):.1f}% target/CK {1.2 * sd / ck:.3f} oldP32/SDPA {pp / sd:.3f}")
# order and transients
print("-- reads under 0.9x arm median in a job:", end=" ")
n = 0
for key, d in res.items():
    for j, v in d.items():
        m = st.median(x[0] for x in v)
        n += sum(x[0] < .9 * m for x in v)
print(n)
