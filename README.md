# horizon-kernels

GPU kernels and vLLM integration work by Horizon Tech d.o.o. on two platforms. Apache-2.0 (see [LICENSE](LICENSE),
[NOTICE](NOTICE) and [REUSE.toml](REUSE.toml)). Author: Hamza Abdagic.

This is research code. It records what we built, how we measured it and what did not work. It is not a supported
product, and the numbers hold for the hardware, software versions and inputs named next to them.

## AMD Instinct MI250X (gfx90a), LUMI-G: [`amd-mi250x/`](amd-mi250x/README.md)

- TileLang kernels: GQA prefill attention, paged flash-decoding, fused mixture-of-experts, dense and small-M GEMM.
- Vendor baselines (PyTorch SDPA, Composable Kernel, hipBLASLt, vLLM), RCCL/OFI scaling across nodes, a vLLM overlay
  that runs the decode kernels in a live server, and TileLang compiler patches 0001-0009.
- Headline results, setup on LUMI and how to run each benchmark are in [`amd-mi250x/README.md`](amd-mi250x/README.md);
  every experiment, with job IDs, numbers, negative results and corrections, is in
  [`amd-mi250x/results/RESEARCH-LOG.md`](amd-mi250x/results/RESEARCH-LOG.md).

This work used compute resources of the EuroHPC Joint Undertaking on LUMI, hosted by CSC (Finland) and the LUMI
consortium, through the EuroHPC Development Access call (project EHPC-DEV-2026D11-419).

## NVIDIA Blackwell SM120 (RTX PRO 6000), vLLM: [`nvidia-sm120/`](nvidia-sm120/README.md)

- An adaptive NVFP4 linear-layer plugin for vLLM with auto-calibration and CUDA-graph capture.
- An FP4 KV-cache attention backend, a fused FP8 draft head for multi-token prediction (MTP), Gated DeltaNet (GDN)
  and MoE kernels, an MTP draft-head training recipe, a port to vLLM 0.32, and the benchmarks.
- The kernels themselves are written in TileLang in our fork, horizon-tilelang, a separate repository that is not
  published yet; without it the NVIDIA plugin and most kernel scripts do not run. This tree holds the integration, the
  experiments and the measurements around them.

## Requirements and status

- Scripts are kept as they ran. Machine-specific locations are written as environment variables: `HK_DATA` (shared
  data: models, packages, caches), `HK_VENV` (the Python environment with vLLM) and `HK_WORK` (the working
  directory of an experiment). Logs and results show the same places as `<data>`, `<venv>` and `<work>`.
- Inputs that are not in this repository (model checkpoints, prompt files, generated kernels) must be supplied
  yourself; each README says which.

## Licence and third-party material

Apache-2.0 for everything we wrote. Some files contain or are derived from TileLang (MIT), vLLM (Apache-2.0) or
flash-linear-attention (MIT); [NOTICE](NOTICE) lists them and [REUSE.toml](REUSE.toml) gives the SPDX identifiers.
