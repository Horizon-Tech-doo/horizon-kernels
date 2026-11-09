"""Attribution of the rows 128 fall, clang 18, d128 16k. 5 base processes (rows 128 plain, lb4, P32 bN96) and 5 ilp processes (rows 128 lb4), alternating. Ratios: TF, same job, higher is better; /SDPA is same-process, mean of pre and post."""
import collections, json, statistics as S, sys
for j in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f"raw5/{j}/run.jsonl") if l.startswith("{")]
    t = collections.defaultdict(list); sd = collections.defaultdict(list); r_ = collections.defaultdict(list)
    for r in rs:
        if r.get("impl") == "sdpa":
            sd[r["tag"]].append(r["tflops"])
    for r in rs:
        if r.get("impl") == "tl" and "tflops" in r:
            k = r["name"].split("d128_16384_")[1] + ("_ILP" if r["tag"].endswith("ilp") else "")
            t[k].append(r["tflops"]); r_[k].append(r["tflops"] / S.mean(sd[r["tag"]]))
    print("job", j, "SDPA TF", min(min(v) for v in sd.values()), S.median(x for v in sd.values() for x in v), max(max(v) for v in sd.values()))
    for k in t:
        print(f"  {k:28s} TF {min(t[k]):.2f} {S.median(t[k]):.2f} {max(t[k]):.2f}  /SDPA {min(r_[k]):.3f} {S.median(r_[k]):.3f} {max(r_[k]):.3f}  n {len(t[k])}")
    p = S.median(t["P32_b96_t768"])
    for k in t:
        print(f"  {k:28s} / P32 median TF {S.median(t[k]) / p:.3f}")
    print("  lb4/plain", round(S.median(t["r128_b32_t512_lb4"]) / S.median(t["r128_b32_t512"]), 3), "ilp/lb4", round(S.median(t["r128_b32_t512_lb4_ILP"]) / S.median(t["r128_b32_t512_lb4"]), 3))
