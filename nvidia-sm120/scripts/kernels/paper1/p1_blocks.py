# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Block plan for the paper-1 NVIDIA re-measurement. CPU only; this script starts nothing on a GPU.

A JOB is one race script with fixed arguments. It is run in three MODES:
  warm  build every cell and run each arm once (fills the kernel cache, first real contact with the kernels);
  fwd   the measurement with forward arm order;   rev   the same with reversed arm order (the second job).
A BLOCK is one run of one JOB MODE of at most 12 minutes. A mode that does not finish in one block is continued by
running the same JOB MODE again: finished cells are skipped (--resume).

  p1_blocks.py list [--ws-mib N]            the plan: jobs, estimated blocks, minutes and required free GPU memory
  p1_blocks.py cmd JOB MODE --out-dir DIR   the argument list of the next block, one argument per line
  p1_blocks.py status --out-dir DIR         what has been done according to the files in DIR
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK_MIN = 12
BUDGET_S = 540            # the race scripts start no new cell after this many seconds
OVERHEAD_S = 75           # checks, interpreter and CUDA start: an estimate
MODES = ("warm", "fwd", "rev")

# job id, target number, script, arguments, memory budget MiB, what it measures
JOBS = [
    ("w4a16-tuned", 1, "race_w4a16.py", ["--set", "tuned"], 4096,
     "W4A16 NVFP4 decode GEMM vs Marlin (and BF16 cuBLAS), 5 layer shapes x 32 row counts"),
    ("w4a16-heldout", 1, "race_w4a16.py", ["--set", "heldout"], 4096,
     "the same on 6 shapes outside the tuned table x 8 row counts"),
    ("gdn-state16", 2, "race_gdn_state.py", ["--variant", "state16"], 3584,
     "gated-delta decode with a 16-bit state vs vLLM's CUDA kernel (f32 and bf16 state), bit-identity gates"),
    ("gdn-ring", 2, "race_gdn_state.py", ["--variant", "ring"], 3584,
     "gated-delta decode with checkpoint + replay ring vs vLLM's CUDA kernel, bit-identity gates"),
    ("moe-block", 3, "race_moe_block.py", [], 4096,
     "fused MoE block (skinny gate, route kernel, stable alignment) vs stock routing glue around Marlin-MoE"),
    ("attn-fp8", 4, "race_attn_decode.py", ["--format", "fp8"], 4096,
     "FP8 paged decode attention vs FlashInfer XQA, full context sweep"),
    ("attn-fp4", 4, "race_attn_decode.py", ["--format", "fp4"], 5120,
     "4-bit KV decode attention vs FlashInfer XQA on FP8, full context sweep"),
    ("outproj", 4, "race_outproj.py", [], 6144,
     "FP8 output projection (W8A16) vs BF16 cuBLAS GEMM, 10 row counts"),
    ("prefill-short", 4, "race_prefill.py", ["--set", "short"], 2560,
     "short ragged prefill attention vs FlashInfer FA2"),
    ("prefill-long", 4, "race_prefill.py", ["--set", "long"], 2560,
     "prefill attention vs FlashInfer FA2 at 1k-16k keys under graph replay"),
]
BY_ID = {j[0]: j for j in JOBS}


def extra_args(job):
    """Per-job arguments from the environment (package directories etc.): P1_EXTRA_<job with - as _>."""
    v = os.environ.get("P1_EXTRA_" + job.replace("-", "_"), "")
    return v.split() if v else []


def argv_for(job, mode, extra=()):
    _, _, script, args, budget, _ = BY_ID[job]
    out = [os.path.join(HERE, script)] + list(args) + ["--mem-budget-mib", str(budget), "--time-budget-s",
                                                      str(BUDGET_S)] + list(extra)
    if mode == "warm":
        out += ["--warm-only", "--no-energy"]
    elif mode == "rev":
        out += ["--reverse"]
    return out


def dry(job, mode, more=()):
    r = subprocess.run([sys.executable, "-I"] + argv_for(job, mode, more) + ["--dry-run"], capture_output=True,
                       text=True, timeout=120)
    if r.returncode != 0:
        raise SystemExit(f"dry run of {job} {mode} failed:\n{r.stderr[-800:]}")
    kv = {}
    for tok in r.stdout.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            kv[k] = v
    return kv


def cmd_list(a):
    more = ["--ws-mib", str(a.ws_mib)] if a.ws_mib else []
    print(f"{'job':<15s}{'mode':<6s}{'cells':>6s}{'skip':>6s}{'est s':>8s}{'blocks':>8s}{'est min':>9s}{'cap min':>9s}"
          f"{'free MiB':>10s}")
    per_target = {}
    for job, prio, _, _, _, what in JOBS:
        print(f"# target {prio}: {job}: {what}")
        for mode in MODES:
            kv = dry(job, mode, more)
            cells, est = int(kv["PLANNED_CELLS"]), float(kv["EST_SECONDS"])
            if mode == "warm":
                blocks, est_min = 1, None          # compile time is not estimated
            else:
                blocks = max(1, math.ceil(est / (BUDGET_S - 30)))
                est_min = (est + blocks * OVERHEAD_S) / 60
            cap = blocks * BLOCK_MIN
            t = per_target.setdefault(prio, [0.0, 0, 0])
            t[0] += est_min or 0
            t[1] += cap
            t[2] += blocks
            print(f"{job:<15s}{mode:<6s}{cells:>6d}{kv['SKIPPED']:>6s}{est:>8.0f}{blocks:>8d}"
                  f"{'   (n/e)' if est_min is None else format(est_min, '9.1f')}{cap:>9d}{kv['REQUIRED_FREE_MIB']:>10s}")
    print("\nplanned GPU time per target (fwd + rev estimated; warm blocks counted at the 12-minute cap only):")
    tot = [0.0, 0, 0]
    for prio in sorted(per_target):
        e, cap, blocks = per_target[prio]
        tot = [tot[0] + e, tot[1] + cap, tot[2] + blocks]
        print(f"  target {prio}: {blocks} blocks, estimate {e:.0f} min measuring, hard cap {cap} min")
    print(f"  all: {tot[2]} blocks, estimate {tot[0]:.0f} min measuring, hard cap {tot[1]} min")
    print("free MiB = real free GPU memory a block needs before it starts (largest cell + fixed + reserve).")
    print("Estimates are not measurements; the first real block of each job shows the true cost.")
    return 0


def files_of(out_dir, job, mode):
    return sorted(glob.glob(os.path.join(out_dir, f"{job}.{mode}.b[0-9][0-9].jsonl")))


def cmd_cmd(a):
    if a.job not in BY_ID or a.mode not in MODES:
        raise SystemExit("unknown job or mode; see: p1_blocks.py list")
    done = files_of(a.out_dir, a.job, a.mode)
    out = os.path.join(a.out_dir, f"{a.job}.{a.mode}.b{len(done) + 1:02d}.jsonl")
    argv = argv_for(a.job, a.mode, extra_args(a.job)) + ["--out", out]
    if done:
        argv += ["--resume"] + done
    print("\n".join(argv))
    return 0


def cmd_status(a):
    for job, _, _, _, _, _ in JOBS:
        for mode in MODES:
            fs = files_of(a.out_dir, job, mode)
            done, failed, left, complete = 0, 0, None, False
            for fn in fs:
                for ln in open(fn):
                    if not ln.startswith("{"):
                        continue
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if r.get("t") in ("cell_done", "warm_done"):
                        done += 1
                    elif r.get("t") == "cell_error":
                        failed += 1
                    elif r.get("t") == "done":
                        left = len(r.get("cells_left", [])) + len(r.get("cells_failed", []))
                        complete = bool(r.get("complete"))
            state = "not started" if not fs else ("complete" if complete else
                                                  f"continue ({'?' if left is None else left} cells left)")
            print(f"{job:<15s}{mode:<6s}{len(fs):>3d} block(s) {done:>4d} cells done {failed:>3d} errors  {state}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    s = sub.add_parser("list")
    s.add_argument("--ws-mib", type=int, default=0)
    s = sub.add_parser("cmd")
    s.add_argument("job")
    s.add_argument("mode")
    s.add_argument("--out-dir", required=True)
    s = sub.add_parser("status")
    s.add_argument("--out-dir", required=True)
    a = p.parse_args()
    return {"list": cmd_list, "cmd": cmd_cmd, "status": cmd_status}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
