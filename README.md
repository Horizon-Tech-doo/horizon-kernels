# horizon-kernels

GPU kernels and vLLM integration work by Horizon Tech d.o.o. on two platforms. Apache-2.0 (see [LICENSE](LICENSE),
[NOTICE](NOTICE) and [REUSE.toml](REUSE.toml)). Author: Hamza Abdagic.

This is research code. It records what we built, how we measured it and what did not work. It is not a supported
product, and the numbers hold for the hardware, software versions and inputs named next to them.

## AMD Instinct MI250X (gfx90a), LUMI-G: [`amd-mi250x/`](amd-mi250x/README.md)

- TileLang kernels: GQA prefill attention (square, scaled and query-window forms), paged flash-decoding, fused
  mixture-of-experts, dense, small-M and bf16 GEMM.
- Vendor baselines (PyTorch SDPA, Composable Kernel, hipBLASLt, vLLM), RCCL/OFI scaling across nodes, and TileLang
  compiler patches 0001-0009.
- vLLM plugins that run the decode attention, the dense GEMMs of prefill steps and the prefill attention of a served
  model on these kernels, a profiling plugin, and the job scripts, clients and analysis of the served-model tracks:
  [`amd-mi250x/vllm-integration/`](amd-mi250x/vllm-integration/).
- Served-model evidence (result write-ups and tables per job pair), the tuned vLLM fused-MoE configuration for MI250X
  and the reproduction of a slow small-row GEMM path in stock vLLM: [`amd-mi250x/evidence/`](amd-mi250x/evidence/README.md).
- Served model (a public 27B hybrid model, BF16, one MI250X card, vLLM 0.30): time to first token is 0.964-0.971 of
  stock with the GEMM plugin at vLLM's defaults from 2.9k prompt tokens and 0.888-0.893 at 14.2k tokens with the
  prefill-attention plugin added. Synthetic prompts; decode is unchanged; greedy outputs are not identical to stock and
  no quality benchmark was run. With our decode attention the decode token rate is above vLLM's default backend, but
  stock vLLM with its Triton attention backend gives the same or more.
- Headline results with their limits, setup on LUMI, how to run each benchmark and how to reproduce the served-model
  numbers are in [`amd-mi250x/README.md`](amd-mi250x/README.md);
  every experiment, with job IDs, numbers, negative results and corrections, is in
  [`amd-mi250x/results/RESEARCH-LOG.md`](amd-mi250x/results/RESEARCH-LOG.md).

This work used compute resources of the EuroHPC Joint Undertaking on LUMI, hosted by CSC (Finland) and the LUMI
consortium, through the EuroHPC Development Access call (project EHPC-DEV-2026D11-419).

## NVIDIA Blackwell SM120 (RTX PRO 6000), vLLM: [`nvidia-sm120/`](nvidia-sm120/README.md)

- An adaptive NVFP4 linear-layer plugin for vLLM with auto-calibration and CUDA-graph capture.
- An FP4 KV-cache attention backend, a fused FP8 draft head for multi-token prediction (MTP), Gated DeltaNet (GDN)
  and MoE kernels, an MTP draft-head training recipe, a port to vLLM 0.32, and the benchmarks.
- The kernels themselves are written in TileLang and are in
  [`nvidia-sm120/tilelang_contrib/`](nvidia-sm120/tilelang_contrib/README.md); the Gated DeltaNet decode path is not
  part of this release yet.
- A kernel-level re-measurement of 11 November 2026 under the method of the AMD part (four jobs per race, an
  identical-code control, a float64 reference): the measurement kit is in
  [`nvidia-sm120/scripts/kernels/paper1/`](nvidia-sm120/scripts/kernels/paper1/README.md) and the results in
  [`nvidia-sm120/results/paper1_11-11-2026/`](nvidia-sm120/results/paper1_11-11-2026/RESULTS.md). In short: parity with
  the Marlin kernel on the 4-bit GEMM (Marlin's time over ours 0.923-1.170, median 1.00, wins and losses by row count), behind
  FlashInfer on FP8 decode attention and on prefill attention in every cell, ahead on a 16-bit recurrent state
  (mostly a gain the vendor kernel also gets from a smaller state), an FP8 output projection against a BF16 one and a
  fused MoE routing block. Synthetic inputs; no served model was measured in that run.

## Paper

The measurement method, a 32-item checklist for kernel-against-vendor comparisons and the findings are written up in
"Benchmarking TileLang Kernels Against Vendor Libraries on AMD MI250X: Method, Checklist and Findings" (H. Abdagic,
2026, preprint; its identifier is not assigned yet). [`CHECKLIST.md`](CHECKLIST.md) is the checklist as plain text; [`CITATION.cff`](CITATION.cff) says how to cite.

## Requirements and status

- Scripts are kept as they ran. Machine-specific locations are written as environment variables: `HK_DATA` (shared
  data: models, packages, caches), `HK_VENV` (the Python environment with vLLM) and `HK_WORK` (the working
  directory of an experiment). Logs and results show the same places as `<data>`, `<venv>` and `<work>`.
- Inputs that are not in this repository (model checkpoints, prompt files, generated kernels) must be supplied
  yourself; each README says which.

## Licence and third-party material

Apache-2.0 for everything we wrote. Some files contain or are derived from TileLang (MIT), vLLM (Apache-2.0) or
flash-linear-attention (MIT); [NOTICE](NOTICE) lists them and [REUSE.toml](REUSE.toml) gives the SPDX identifiers.
