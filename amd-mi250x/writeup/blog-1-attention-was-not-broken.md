# We thought TileLang attention was broken on AMD. It was the test.

*Draft blog post, Horizon Tech. All numbers come from runs on one GCD of an AMD Instinct MI250X on LUMI-G, logged in
RESEARCH-LOG.md (story and tables 5 Nov 2026, closing numbers 6 Nov). This work used EuroHPC JU compute resources (LUMI, project EHPC-DEV-2026D11-419).*

## The symptom

We are porting attention and mixture-of-experts kernels written in [TileLang](https://github.com/tile-ai/tilelang) to
AMD's MI250X (gfx90a) on the LUMI supercomputer. The first thing we ran was TileLang's own AMD flash-attention example.
It failed its correctness check, with every output element about 5% away from the reference. A failure that uniform
looks like a real kernel bug. Our first hypothesis was the step where the softmax probabilities stay in registers and
feed the second matrix multiplication, because register layouts of AMD's matrix cores (MFMA) differ from NVIDIA's.

That hypothesis was wrong. Here is how we found out, in four short experiments, each a few minutes on one GPU.

## 1. Check each step on its own

We split the kernel into its four steps and compared each one with PyTorch on a single block of 64 queries:

- S = Q·Kᵀ;
- the row maximum and row sum of S, computed directly on the matrix-core output;
- P·V with P kept in registers, which was the suspect;
- P·V with P going through shared memory.

Every step was correct to about 1e-5 relative error, across six tile shapes and thread counts. The suspect was innocent.

## 2. Watch how the error behaves

If the steps are right, the error must come from combining many blocks: the online softmax, which rescales earlier
results whenever a larger score appears. So we grew the sequence length:

| tokens | 64 | 256 | 1,024 | 4,096 |
|---|---|---|---|---|
| relative error | 0.06% | 0.07% | 0.3% | 4.4% |

The error grew with length and was spread evenly over all rows. That is the signature of precision loss accumulating,
not of a logic bug, and it could come from either side of the comparison.

## 3. Ask a third party

So we compared both the kernel and the example's reference with a float64 computation:

| | 1,024 tokens | 4,096 tokens |
|---|---|---|
| TileLang kernel | 0.028% | 0.028% |
| PyTorch `scaled_dot_product_attention` | 0.028% | 0.028% |
| the example's reference (`einsum` + `softmax` in fp16) | 0.33% | **4.3%** |

The kernel was as accurate as PyTorch's own attention. The reference was the inaccurate side: in fp16 on ROCm, its
attention scores and softmax drift as sequences get longer. The example presumably passes on the GPU it was developed
for (MI300) because the fp16 arithmetic there happens to accumulate differently. We did not verify that.

## 4. The real problem, and what came after

One failure was real: with head size 256, the shape our Qwen model uses, the kernel did not launch at all. That turned
out to be simple arithmetic. MI250X allows 64 KB of shared memory per workgroup, and the example's tiles needed exactly
64 KB (Q tile plus K and V tiles). At 48 KB they launch.

With correctness settled, we wrote a version for grouped-query attention (GQA): one workgroup handles 32 query positions
for all 6 query heads that share a key/value head, so each K/V tile is read once instead of six times, and Q stays in
registers. On the Qwen shape (24 query / 4 key-value heads, head size 256, causal) it is faster than PyTorch's attention:

| tokens | PyTorch SDPA | TileLang GQA kernel | speedup |
|---|---|---|---|
| 1,024 | 24.5 TFLOPS | 36.3 TFLOPS | 1.48× |
| 4,096 | 53.8 TFLOPS | 65.8 TFLOPS | 1.22× |
| 8,192 | 63.1 TFLOPS | 74.6 TFLOPS | 1.18× |

(error 2.5e-4 to 2.6e-4 against float64 in every case)

To be fair about what "PyTorch's attention" means here: PyTorch 2.7.1 on ROCm 6.2 does not use AMD's best kernel.
AMD's Composable Kernel flash-attention (flash-attn 2.8.3, ROCm build) on the same GPU reaches 52.9, 84.2 and 92.4
TFLOPS on those three cases, so that first kernel was at 0.69–0.81× of the vendor-tuned one.

Those tables are the first version, and they crossed ROCm stacks. A day later, with loads and tiles reworked and both kernels in
one process on ROCm 7.2.3, we read 1.06–1.18× CK at head size 256 across lengths and real inputs (about 1.07× on random data at 16k), and 1.38–1.93× SDPA. At head
size 128 we beat CK up to about 2.5k tokens with a tile table in the script, reach 1.015–1.038× at 2.8k–3.1k tokens on six inputs only with a
hand-structured form on our fork (not yet in our module; 0.991–1.015× at 3.6k), tie it at 4k except on one input, and stay at 0.946–0.971× from 5k up with the landed tile, so that part of the gap is
still open. These kernels use C edits on TileLang's generated source, or hand-structured forms on a patched TileLang fork; stock
TileLang alone reaches 0.93× and 0.79×.

## What we take away

- **Before debugging a kernel, check the reference.** A reference computed in fp16 is not a reference at long sequence
  lengths. Compare against float64 (or at least fp32 accumulation), and against a second independent implementation.
- **An error that grows smoothly with problem size and hits every element evenly** points to precision, not to a
  logic bug in a tile or a layout. Logic bugs usually hit some rows, tiles or heads and not others.
- **On MI250X, count shared memory first.** 64 KB per workgroup is the limit. A hand kernel at exactly 65536 B launches, but TileLang adds reduce workspaces on top, so count those too.

We will report the reference issue to the TileLang maintainers, with the fix: compute `ref_program` in fp32.

*Scripts: `fa_diag.py`, `fa_diag2.py`, `fa_diag3.py`, `fa_bench.py`, `gqa_fa.py` in
this repository (`amd-mi250x/`). Our TileLang compiler fixes are prepared as a patch series; no pull request is open yet.*
