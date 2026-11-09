Hi all, a bit of context on where these PRs come from and where I'd like to take them.

I'm Hamza Abdagic from Horizon Tech. I've been fully behind TileLang since the moment I first read about it, and I'd like to become a long-term contributor here, not just someone who opens a few PRs.

I work with RTX PRO 6000 Blackwell GPUs (SM120, 96 GB), serving Qwen3.8-27B in vLLM. When I started, SM120 was poorly served for NVFP4:
- The FlashInfer FP4 GEMMs crashed on this setup.
- vLLM's CUTLASS FP4 path gave **79.6 tok/s** single-stream.
- The official FP8 checkpoint gave **83.5 tok/s**.
- The best option was NVFP4 through Marlin (W4A16): **111.4 tok/s for 1 user and 708.5 tok/s total for 8 users**.

Since then, with kernels written in TileLang, the same card serves **143 tok/s for 1 user (+28% over Marlin) and ~835 tok/s for 8 users (+18%)**. The FP4 prefill GEMM (two-pass residual NVFP4 on the native tensor cores) runs **1.5–1.9× faster than Marlin at 512–8,192 tokens**, and quality holds in paired GSM8K / MMLU-Pro evals and log-prob checks. Part of the end-to-end gain comes from a retrained speculative-decoding draft head, but the core is the TileLang kernels, and that's the part I want to bring upstream:
- #3286: register-A speed-up for the SM120 block-scaled GEMM
- #3288: fix for the warp-specialization early-release race (#3287)
- #3290 (draft): W4A16 NVFP4 decode GEMM, pending the placement question in #1592

What you can expect from me going forward:
- **SM120 / workstation Blackwell:** I'll keep contributing kernels, fixes and tests, and I'm happy to run tests or benchmarks on sm_120 hardware for other people's PRs. Just tag me.
- **The rest of the repo:** wherever I can help, e.g. bugs I run into, examples and docs.
- **Other hardware:** if I get access to more or different GPUs, I'll contribute for those too.

My goal is simple: get as much performance out of this hardware as possible, and land it upstream so everyone benefits. I'm happy to adapt to however you prefer contributions structured. Since these are my first PRs here, CI needs a maintainer's approval to run whenever someone has a moment.
