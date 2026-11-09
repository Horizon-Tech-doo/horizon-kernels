# vLLM PR draft: restrict the GDN fwd-h kernel to num_warps=2 on Blackwell

> **NOT our cause (controlled test: no effect).** Controlled test: NVIDIA 27B with the fix (Nf)
> is bit-identical to stock (Nc) (prose 16k NLL 1.6092, 32k 3.2911), and the unchunked run (Nu, 1.656) is just as bad,
> so neither the chunk carry-over nor this kernel explains our long-prose loss. On this box the stock autotuner already
> ran a 2-warp config. Both the PR draft and the #53305 comment below are kept only as a record.

Status (27-09-2026 19:55): **DRAFT, NOT POSTED.** The evidence table is filled from that test; post only if Nf ≈ Nu (the fix removes the chunk-boundary break).
Policy check done: no open vLLM PR ports FLA #953 (searched `chunk_delta_h`, "num_warps Blackwell gated delta");
#53305 is open. No AI attribution anywhere.

- Patch: `docs/upstream/vllm-fla-blackwell-warps.patch` (vLLM main 2b9b55c7f1, 27-09-2026; 2 files, +9/−2),
  prepared in a scratch clone on branch `horizon/fla-blackwell-fwd-h-warps`, DCO sign-off by the author,
  `ruff check` + `ruff format --check` clean on the changed files.
- Regression test: not included yet. A useful one needs a Blackwell GPU (the miscompile does not show in the Triton
  interpreter or on Hopper). Proposal: `tests/kernels/mamba/test_gdn_fwd_h_blackwell.py`, skipped unless cc 10/12:
  compare `chunk_gated_delta_rule_fwd_h` over 2L tokens in one call vs two calls with the final state as `h0`, plus a
  config-space assertion (`{c.num_warps for c in tuner.configs} == {2}`), as FLA #953's test does.


## PR body (pr-body.md)

### Purpose

Triton ≤ 3.7 miscompiles the state recurrence of `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` on Blackwell when
the autotuner picks `num_warps=4` (triton-lang/triton#9871, #10590). flash-linear-attention restricted this kernel to
2 warps on Blackwell in fla-org/flash-linear-attention#953; vLLM's vendored copy still autotunes over `[2, 4]`
(vLLM pins Triton 3.7.1).

In vLLM this shows up with chunked prefill: the chunks of a long prompt that start from a carried-over GDN state
(`h0`) produce wrong results, so Qwen3.5-family models degrade sharply right after the first prefill chunk boundary.
This matches #53305 (Qwen3.5 on RTX PRO 6000; workaround: never split the prompt).

This PR applies the same restriction as FLA #953 on compute capability 10.x / 12.x. Other GPUs are unchanged.

### Test plan / results

Qwen3.8-27B checkpoints on 1× RTX PRO 6000 Blackwell (SM120), vLLM 0.29.0, Triton 3.7.1, per-token prompt log-probs of
a 32k-token document (a long chat transcript), default chunked prefill (8192 budget; first boundary at token 8000,
or 6432 for the 35B-A3B MoE because of its larger mamba block).

| Model | Setting | dNLL vs reference, tokens before boundary | after boundary |
|---|---|---|---|
| nvidia/Qwen3.8-27B-NVFP4 (ModelOpt) | stock | (#5c Nc) | (#5c Nc) |
| same | this PR | (#5c Nf) | (#5c Nf) |
| same | prompt not split (reference) | (#5c Nu) | (#5c Nu) |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | stock / this PR / not split | (#5c Mc / Mf / Mu) | |
| Qwen3.8-27B NVFP4 (another quantisation) | stock / this PR | (#5c Pc / Pf) | |

Before the fix (measured 27-09-2026): the NVIDIA 27B is *better* than our reference before token 8000
(dNLL −0.022) and loses PPL ×28 after it; the first 8000 tokens are bitwise identical between a 16k and a 32k request,
only chunks that start from `h0` diverge. Cold TTFT with the fix at 16k / 64k: (#5c TTFT lines).

## Comment draft for #53305 (NOT posted)

> **NOT our cause (controlled test: no effect).**

> We see the same failure on an RTX PRO 6000 Blackwell (SM120), vLLM 0.29.0, Triton 3.7.1, with Qwen3.8-27B
> checkpoints: per-token log-probs of a 32k prompt are fine (even slightly better than our reference) up to the first
> chunked-prefill boundary and break right after it (PPL ×28 on the rest of the prompt). The first chunk is bitwise
> identical across request lengths; only chunks that start from a carried-over GDN state go wrong. Restricting the
> vendored `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` to `num_warps=2` on Blackwell, as
> flash-linear-attention did in fla-org/flash-linear-attention#953 (Triton issues triton-lang/triton#9871, #10590),
> gives: (#5c numbers). PR: (link once opened).
