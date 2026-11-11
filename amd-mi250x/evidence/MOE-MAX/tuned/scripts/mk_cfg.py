"""Merge the per-T 'best' records of tune_moe.py into one config file in vLLM's format (as benchmark_moe.py save_configs writes
it: {"triton_version": ..., "<batch size>": {config}, ...}, file name from vLLM's get_config_file_name). A later file on the
command line overrides an earlier one for the same T. Also prints a summary table. argv: out_dir tune_T*.jsonl ..."""
import json, os, sys
best, meta = {}, {}
for p in sys.argv[2:]:
    for l in open(p):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        if r.get("kind") == "best":
            best[r["T"]] = r; meta[r["T"]] = os.path.basename(p)
names = {r["cfg_file"] for r in best.values()}; tv = {r["triton"] for r in best.values()}
assert len(names) == 1 and len(tv) == 1, (names, tv)
out = os.path.join(sys.argv[1], names.pop())
with open(out, "w") as f:
    json.dump({"triton_version": tv.pop(), **{str(T): best[T]["cfg"] for T in sorted(best)}}, f, indent=4)
    f.write("\n")
print("wrote", out)
print(f"{'T':>5s} {'mode':6s} {'pruned':>6s} {'timed':>6s} {'skip':>5s} {'fail':>4s} | {'default us':>10s} {'tuned us':>9s} {'ratio':>6s} | tuned config  [{'source file'}]")
for T in sorted(best):
    r = best[T]; c = r["cfg"]
    mode = "full" if r.get("n_full") else "staged"
    n = r.get("n_full", 0) + r["n_s1"] + r["n_s2"]
    cs = "default" if r["is_default"] else (f"{c['BLOCK_SIZE_M']}x{c['BLOCK_SIZE_N']}x{c['BLOCK_SIZE_K']} G{c['GROUP_SIZE_M']} w{c['num_warps']} st{c['num_stages']} "
                                             f"wpe{c.get('waves_per_eu')} nk{c.get('matrix_instr_nonkdim')} kp{c.get('kpack')}")
    print(f"{T:5d} {mode:6s} {r['pruned']:6d} {n:6d} {r['n_skipped']:5d} {r['n_fail']:4d} | {r['default_us']:10.2f} {r['us']:9.2f} {r['ratio']:6.3f} | {cs}  [{meta[T]}]")
