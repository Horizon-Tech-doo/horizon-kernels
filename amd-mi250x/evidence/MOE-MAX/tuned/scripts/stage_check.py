"""How much does the staged search (s1 -> s2, tune_moe.py) lose against timing every pruned config? Replays the staged
selection on the exhaustive ('full') records of a T: s1 = configs with GROUP_SIZE_M 1, waves_per_eu 0, matrix_instr_nonkdim 16,
kpack 1; s2 = every config that shares block sizes and num_warps with one of the --tops fastest s1 configs. Reports the
fastest time each would have seen, from the same single-pass timings (graph of 10 calls, 5 replays). argv: tune_T*.jsonl ... [--tops n]"""
import json, sys
tops = int(sys.argv[sys.argv.index("--tops") + 1]) if "--tops" in sys.argv else 8
print(f"{'T':>4s} {'timed':>6s} {'best us':>9s} {'s1 best':>9s} {'staged best':>11s} {'staged/full':>11s} {'rank of staged pick':>19s} | default us")
for p in [x for x in sys.argv[1:] if x.endswith(".jsonl")]:
    R = [json.loads(l) for l in open(p) if l.startswith("{")]
    F = [r for r in R if r.get("kind") == "time" and r["stage"] == "full" and not r.get("skipped")]
    if not F:
        continue
    d = [r for r in F if "waves_per_eu" not in r["cfg"]]
    F = [r for r in F if "waves_per_eu" in r["cfg"]]
    base = lambda c: (c["BLOCK_SIZE_M"], c["BLOCK_SIZE_N"], c["BLOCK_SIZE_K"], c["num_warps"])
    s1 = sorted([r for r in F if r["cfg"]["GROUP_SIZE_M"] == 1 and r["cfg"]["waves_per_eu"] == 0 and r["cfg"]["matrix_instr_nonkdim"] == 16 and r["cfg"]["kpack"] == 1], key=lambda r: r["us"])
    tb = {base(r["cfg"]) for r in s1[:tops]}
    st = min(r["us"] for r in F if base(r["cfg"]) in tb)
    allus = sorted(r["us"] for r in F)
    print(f"{F[0]['T']:4d} {len(F):6d} {allus[0]:9.2f} {s1[0]['us']:9.2f} {st:11.2f} {st / allus[0]:11.4f} {allus.index(st) + 1:19d} | {d[0]['us'] if d else float('nan'):.2f}")
