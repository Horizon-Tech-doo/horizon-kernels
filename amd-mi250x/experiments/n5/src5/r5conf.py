"""Clang 18 confirm, d128 16k, fork-main TileLang forms (S staged, J joint), no hand edit. Ratio: ours / same-process SDPA (mean of start and end reads), TF, higher is better. Stop rule: an arm must beat the better of r128_b64_J and r256_b32_S by more than 1% in the same process in both jobs."""
import collections, json, statistics as S, sys
for j in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f"raw5/{j}/run.jsonl") if l.startswith("{")]
    for tag in sorted({r["tag"] for r in rs if "tag" in r and "case" in r}):
        x = [r for r in rs if r.get("tag") == tag]
        sd = [r["tflops"] for r in x if r.get("impl") in ("sdpa", "sdpa_end")]
        t = [r for r in x if r.get("impl") == "tl"]
        tf = {r["name"].replace("d128_16384_", ""): r["tflops"] for r in t}
        ref = max(tf["r128_b64_t512_JDRF"], tf["r256_b32_t1024_SDRF"])
        print(j, tag, "sdpa", sd, "order", [r["name"].replace("d128_16384_", "") for r in t], "errs", max(r.get("rel_vs_sdpa", 0) for r in t), {r.get("repeat_equal") for r in t})
        for k, v in sorted(tf.items(), key=lambda a: -a[1]):
            print(f"   {k:24s} {v:6.2f} /SDPA {v / S.mean(sd):.3f}  /best finalist {v / ref:.3f}")
