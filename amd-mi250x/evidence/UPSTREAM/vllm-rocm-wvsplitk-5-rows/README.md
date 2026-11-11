# vLLM ROCm `wvSplitK` at 3 and 5 rows: reproduction (11 Nov 2026)

A slow path in stock vLLM 0.30 on gfx90a that the served-model measurements surfaced. It is not caused by our kernels
and a fix needs none of them. Nothing has been filed upstream at the time of this release.

- `ISSUE-DRAFT.md`: the draft issue text, self-contained.
- `repro_wvsplitk.py`: stand-alone reproduction (vLLM and torch only). `repro.sbatch`: how it was run on LUMI (job
  23626809, small-g, one GCD). `up-wvsplitk-23626809.out`: its output [observed].
- `served/client-*.jsonl`: the client rows of the served-model check, job 23625797 (stock against stock with
  `VLLM_ROCM_USE_SKINNY_GEMM=0`, arm order s sk sk s, small-g, 2 GCDs, one job) [observed]. The job log is not included.
  Rows with phase `w` are warm-up requests and are not used.
- `extra_gm_skinny.py`: the earlier probe over the model's five layer shapes at 1 to 32 rows (job 23625791).

Observed: the standalone times (one job each, 7 interleaved rounds; the reproduction has no identical-code arm, the
earlier probe had one and read under 2.4%) and the served step times (one job, two repetitions per arm, spread at most
1.7%). Inferred: that the served regression at 5 streams comes from the LM head (the layers run at the graph's padded 8
rows); the head's 5-row time (11.4 ms against 1.4 ms) matches the 10 ms seen, but the head was not timed inside the
server. Not tested: 3 and 7 concurrent requests in the server, MI300-class GPUs, other versions.

A fix: route by row count in `rocm_unquantized_gemm_impl`, or a general plugin that does the same.
