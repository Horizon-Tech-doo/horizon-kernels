# paper1: kernel-level re-measurement of the SM120 kernels (11 November 2026)

The measurement kit behind the NVIDIA section of the paper "Benchmarking TileLang Kernels Against Vendor Libraries on
AMD MI250X: Method, Checklist and Findings". It races our kernels against the vendor kernels under the method of the AMD
part of this repository. The results of the run of 11 November 2026 are in
[`../../../results/paper1_11-11-2026/`](../../../results/paper1_11-11-2026/RESULTS.md).

## Files

| File | Role |
|---|---|
| `p1_common.py` | shared harness: cell plan, checks, timing pass, energy pass, power sampler, masking, JSONL |
| `race_w4a16.py` | W4A16 NVFP4 decode GEMM vs Marlin and BF16 cuBLAS |
| `race_gdn_state.py` | gated-delta decode state kernels (16-bit state; replay ring) vs vLLM's CUDA kernel |
| `race_moe_block.py` | fused MoE block with stable alignment vs stock routing glue around Marlin-MoE |
| `race_attn_decode.py` | FP8 decode attention vs FlashInfer XQA; 4-bit KV decode vs XQA on FP8 |
| `race_outproj.py` | FP8 output projection vs BF16 cuBLAS |
| `race_prefill.py` | ragged prefill attention vs FlashInfer FA2 (short; long under graph replay) |
| `p1_blocks.py` | the block plan: `list`, `cmd JOB MODE`, `status` (CPU only) |
| `p1_analyze.py` | keep rule, per-cell ranges, noise bound, energy columns (CPU only) |

## Method

- One process per block. In every cell our kernel (`new`), the vendor kernel or kernels and `ctl` run on the same input
  buffers. `ctl` is the same kernel objects as `new`, captured into a second CUDA graph: identical code.
- At least 9 cycles per cell (fewer is refused). Each cycle runs every arm once; the arm order is rotated by one per
  cycle. `--reverse` reverses the arm list: that is the second job. The run of 11 November used two input seeds, each
  with a forward and a reverse job: four jobs per race.
- A timed value is the device-event time of `reps` graph replays after `warm` replays, divided by the calls. Every
  per-cycle value is written. Graphs are captured with `capture_error_mode="thread_local"` and a retry; an arm that
  cannot be captured is not timed.
- Every arm streams from more than `--ws-mib` (513) MiB of rotated weights, caches or state. Where one buffer is
  already larger than that, a single copy is used (recorded per cell).
- Correctness before timing, per cell: float64 reference or bit-exact comparison, non-finite count, repeat equality,
  graph replay after poisoning the outputs equal to the eager result. A failed check does not stop the timing; it is
  written into the cell (`gate_pass`) and the analysis does not certify the cell.
- Noise bound per job: the largest |median(ctl) / median(new) - 1| over its cells. A cell is a win or a loss only if it
  is beyond that bound in every job, with the checks passed and at least 9 cycles.
- Records are JSON lines. Software versions, a hash of each arm's code (source file, generated kernel sources where the
  module keeps them, mapped vendor libraries) and `graph_replay` are recorded. Every string passes a mask, and vendor
  libraries appear by base name and hash only.
- Inputs are synthetic and seeded. Checkpoints are opened only when a path is given (`--params-dir`, `--ckpt-dir`,
  `--bf16-file`), read-only.

## Power and energy

- Timing pass and energy pass are separate. No sampler thread exists while the timing pass runs; between timed arms
  one instantaneous reading (power, temperature, clocks, utilisation) is taken.
- Energy pass, on a subset of cells per job: after warm-up each arm replays in a tight loop for `--energy-seconds`
  (5 s). Recorded: energy over the window, replays, calls, wall time, sample count, average / min / max watts,
  temperature, clocks, utilisation.
- Source: NVML through `pynvml`. The energy counter difference is used when the driver supports it, else the integral
  of power samples at 20 Hz; without `pynvml`, an `nvidia-smi` loop at the same rate. Which one was used is recorded.
- Idle power is measured the same way at the start and end of every block. Nothing is subtracted in the raw records;
  the analysis prints a derived net figure next to the gross one.
- Every power reading is for the whole board. On a shared GPU it includes whatever another process does. The harness
  can read the activity counters of a co-resident server process (its URL from `VLLM_LOCAL_URL`) before and after
  every cell, energy window and block; a cell is marked contaminated if they moved and unknown if they could not be
  read. On 11 November the co-resident process was loaded but asleep (`P1_ALLOW_SLEEPING=1`). The counter values are
  not published. `P1_NVML_INDEX` selects the GPU for power readings.

## Running

1. `python3 p1_blocks.py list` shows the jobs, the estimated blocks and the free GPU memory each needs.
2. `python3 p1_blocks.py cmd JOB MODE --out-dir DIR` prints the argument list of the next block of a job, one argument
   per line (`MODE` is `warm`, `fwd` or `rev`); run it with the Python environment that has the kernels, vLLM and
   FlashInfer. A mode that does not finish in one block is continued by asking for the next block; finished cells are
   skipped.
3. `python3 p1_blocks.py status --out-dir DIR` shows what is left.
4. `python3 p1_analyze.py DIR/JOB.fwd.b*.jsonl DIR/JOB.rev.b*.jsonl` applies the keep rule. Its output for the run of
   11 November is in the results folder; the block records it read are not in the repository (available on request).

Per-job extra arguments (kernel package directories) are read from `P1_EXTRA_<job with - written as _>`.

## What is not included, and limits

- A site-specific launcher that ran one block at a time is not included. The scripts above are what it called.
- The gated-delta races (`race_gdn_state.py`) need the GDN decode path, which this release holds back (see
  `../../../README.md`), staged as a package `hzgs`, and a gate module from `../gdnstate/`. The 4-bit attention race
  takes a staged package `hzkv`, the MoE race a staged stable-alignment module, and the prefill races ran with a
  candidate prefill module passed by `--module-file`. The staged copies are not part of this tree; whether the modules
  of the same name under `../../../tilelang_contrib/` are byte-identical to them was not checked. With the default
  `--pkg` every prefill cell fails, because the published `attn_prefill` has no `plan_prefill_tiles`.
- Not covered: the grouped W4A16 MoE GEMM race, the tuned 4-bit serving kernels and cache write, caches captured
  from a real model run,
  token rate inside a server, an import check on stock TileLang.
- Time and memory figures printed by `p1_blocks.py list` are estimates, not measurements; `peak_mib` in every cell
  record shows what a cell used.
