# Reading the KV cache faster than the vendor kernels on AMD MI250X

*Draft blog post, Horizon Tech, not published. Measured on one GCD of an AMD Instinct MI250X on LUMI-G, logged in
RESEARCH-LOG.md (6 Nov 2026). This work used EuroHPC JU compute resources (LUMI, project EHPC-DEV-2026D11-419).*

When a language model generates text, each new token attends to every earlier one. Per step, the GPU reads the key/value
(KV) cache once, with almost no arithmetic: about two floating-point operations per byte. Decode speed is set by memory
bandwidth, and the honest metric is the share of the GPU's memory bandwidth a kernel actually uses. One MI250X GCD has about
1.6 TB/s.

We wrote a decode kernel in TileLang for the attention shape of a Qwen 27B model: 6 query heads per key/value head (GQA),
head size 256. Everything below is compiler output: no hand-edited kernels. A later section shows that the lead over Triton below is against the wrong form of that kernel: against the form a server runs we are about level, and there is no token-rate gain over vLLM's better backend.

## We first raced the wrong thing

Our early numbers used 16-token pages, FP16 and a layout we had read out of vLLM's source. Then we started vLLM 0.30 with the
real weights and looked at the cache it allocated:

- the default backend, ROCM_ATTN, on a layout called LBHNC, with K then V inside each page;
- **784 tokens per page, BF16**, not 16 and FP16;
- a generic paged kernel, because vLLM's custom ROCm kernel refuses head size 256;
- only 111 pages on one GCD at our test settings, about 87,000 tokens per layer. Long 1 GB races are fine as kernel tests, but
  that configuration cannot serve them.

`--attention-backend TRITON_ATTN` switches to vLLM's Triton kernel on another layout (LBNHC), at the same page size. That flag
alone is 2.09–3.33× faster than the default on the shapes we ran. So the numbers below come from this frame, and we kept only
shapes that fit.

## What worked, in order

1. **Read each K/V tile once per group of query heads**, through MFMA matrix cores. A plain kernel without them was 10–50× slower.
2. **Split the cache** (flash-decoding): many workgroups take a slice each, then a small kernel combines them. Without this, one
   sequence keeps a handful of the 110 compute units busy.
3. **Small tiles for occupancy, padded rows for wavefronts.** 16-key tiles leave room for more workgroups per compute unit.
   Padding 6 query heads to 32 rows looks wasteful, but two wavefronts then share each tile and keep more reads in flight.
4. **All eight K and V loads in one round.** One kernel's time at batch 32 went from 1.004 to 0.803 ms, with identical checksums.
5. **Pick the split count per shape.** At 32 × 1k, 5 splits beat 4 by 11–13%, and 6 is worse again. The best count seems to follow
   the number of active blocks, which is still a hypothesis.
6. **Look at the assembly for BF16.** Our BF16 kernels were 2–3% slower than FP16 on short shapes, and Triton's were not. The cause
   was one line: `hip_bfloat16`'s cast adds an exec-mask branch for every element (30 `saveexec` against 22), where Triton's is
   branchless. A bit-level conversion gave back a median 2.4%.

## Result

**Read this first:** "Triton" in this section is the 2D form of vLLM's kernel, which is what our lab called. A running server uses a
3D form for batches up to the threshold, and against it these Triton ratios do not hold; see the section after next. The ratios against vLLM's default backend are a different kernel and stand.

Time per decode step in ms, base lengths, random data, two jobs (23509050, 23509051), same job for all arms, ROCM_ATTN on one
seed. Worst error is 2.24e-3 in BF16, against vLLM's 2.30e-3.

| sequences × tokens | ours, LBNHC | vLLM Triton | vLLM default | ours ÷ Triton | ours ÷ default | ours, no vLLM change | HBM share |
|---|---|---|---|---|---|---|---|
| 8 × 8k | 0.21 | 0.63 | 1.60 | 3.0× | 7.7× | 0.415 | 70–71% |
| 16 × 4k | 0.22 | 0.33 | 0.81 | 1.5× | 3.7× | 0.42 | 68% |
| 32 × 2k | 0.215 | 0.25 | 0.80 | 1.15× | 3.7× | 0.40 | 68% |
| 32 × 1k | 0.117 | 0.13 | 0.41 | 1.1× | 3.5× | 0.21 | 63% |

"Ours ÷ X" is X's ms over our ms, so higher is better. Over six length patterns on these four shapes, on our final bytes and new seeds,
we read 1.10–5.33× Triton, none of 72 draws per job below 1, and 3.07–13.47× the default. On the bytes before the last gate edit, over more seeds:
1.07–3.98× and 2.79–10.18×. The lowest bandwidth share is 52%, at 32 × 1k with skewed lengths, measured before these last changes.

The "no vLLM change" column is a kernel for the layout vLLM already allocates. It reads 1.33–4.76× the default, but it is
often slower than simply setting the Triton flag (0.54–1.87× Triton), and it uses only 29–39% of bandwidth. The fast path needs the flag.

At tensor parallel 2, each GCD holds half the heads and many more pages. Per-GCD step time, with communication not counted, reads
1.335–7.97× Triton on final bytes, none of 60 draws per job below 1.

## Where we do not win clearly

**64 × 1k.** Skipping dead splits gains about 5.5% with skewed lengths, and we gate it off where it loses (base and full lengths, small batches). Over 28 seeds that cell now reads a median 1.070× Triton, down to 0.994×, about 3% of draws below 1. The lowest cell is now mid lengths at the same shape, 1.05–1.06×, where skip does nothing. An earlier result of 8 wins out of 8 was on FP16, and it did not hold on BF16, the type actually served.

**A short-skew bandwidth gap at tensor parallel 2**, where the share falls to 39–41%, under our 50% target.

**Simulated FP8 KV**, an earlier test: MI250X has no FP8 hardware, and decoding through a lookup table was 1.9–2.2× slower than FP16.

## Then we ran it in the server, and got no gain

We wired the kernels into a running vLLM with real weights at tensor parallel 2 and measured tokens per second, plugin over stock, two jobs with opposite arm order.

| | short ×1 | short ×8 | short ×32 | 2k ×1 | 2k ×8 | 2k ×32 | 8k ×1 | 8k ×8 |
|---|---|---|---|---|---|---|---|---|
| ours over default backend | 1.012, 1.012 | 1.010, 1.013 | 1.011, 1.009 | 1.160, 1.168 | 1.103, 1.101 | 1.075, 1.073 | 1.771, 1.777 | 1.485, 1.468 |
| Triton flag over default backend | 1.022, 1.050 | 1.015, 1.025 | 1.014, 1.025 | 1.219, 1.242 | 1.126, 1.138 | 1.098, 1.114 | 1.834, 1.852 | 1.520, 1.517 |
| ours over the Triton flag | 0.991, 1.009 | 0.995, 0.991 | 0.994, 0.992 | 0.966, 0.976 | 0.983, 0.985 | 0.994, 0.994 | 0.981, 0.991 | 0.996, 0.991 |

**Against the flag there is no gain.** The one read above 1 reverses in the other job order. Three things explain most of it:

- Our lab called Triton's kernel without the arguments a server passes. Serving runs a 3D kernel with 16 segments for batches up to 64 at this head split (32 at the single-GCD split); the lab timed the 2D one. We have now raced the 3D kernel: see the next section.
- A step takes 30–51 ms and context length adds 1–2 ms of it, about 4% at 8 × 8k. Only 16 of the model's 64 layers reach this attention.
- Under vLLM's default mode decode steps replay a captured graph, so a Python hook is never called. A first job looked like it ran our kernels and did not.

Text at temperature 0 differs from stock on some rows, about as often as stock differs from itself across jobs, and 64 GSM questions cannot resolve a small quality effect. Inside a captured graph we cannot run our numeric check, so those arms rest on logprob comparisons (differences match stock against stock, with one outlier on a very unlikely token).

## Racing the kernel a server actually runs

We raced Triton's 3D kernel in two opposite-order jobs. Eager launch adds a floor of about 0.13 ms that a server does not pay, so the replayed-graph column is the one that matters; it covers one seed and three length patterns.

| heads per GCD, sequences × tokens | old ratio, 2D Triton | replayed graph, 3D Triton |
|---|---|---|
| 4, 8 × 8k | 2.88–4.08 | 0.963–1.133 |
| 4, 16 × 4k | 1.53–2.29 | 1.070–1.136 |
| 4, 32 × 2k | 1.14–1.53 | 1.224–1.248 |
| 4, 32 × 1k | 1.12–1.64 | 1.304–1.367 |
| 2, 8 × 8k | 5.37–7.44 | 0.952–1.080 |
| 2, 16 × 4k | 2.78–3.85 | 0.881–1.143 |
| 2, 16 × 8k | 2.92–4.13 | 0.932–1.119 |
| 2, 8 × 32k | 5.94–7.28 | 1.029–1.104 |
| 2, 32 × 8k | 1.58–2.33 | 0.988–1.108 |
| 2, 32 × 1k | 1.35–1.79 | 1.007–1.300 |

**The 1.10–5.33× and 1.335–7.97× above do not describe what a server runs.** With 4 heads per GCD ours reads about 0.87–1.37× (the low end is an eager draw on another seed), and with 2 heads 0.86–1.30× on replay, with real losses at several cells. Our replay arm also uses a kernel fitted to one exact length set, which a server capturing once cannot use, so this probably flatters us: the unfitted form reads below 1 at 5 of the 8 cells we ran in vLLM, lowest 0.856. At 64 sequences with 4 heads vLLM stays on the 2D path and we stay ahead at 1.06–1.27×, read in eager mode only.

This explains the flat line in the server: in the eight served cells the gap between ours and 3D Triton is −0.31 to +0.26 ms in a step of 30–50 ms, and even a kernel taking no time at all would gain about 4%. A step model of 16 layers on the 3D kernel lands within 0.8–2.2 ms of what the server shows; the 2D kernel misses 8k by 9–11 ms. We did not instrument a running server, so that it runs 3D is still an inference, a strong one.

It does not explain everything. Our plugin loses up to about 1 ms per step against stock Triton, and after launch and graph-mode costs up to 1.25 ms of that remains unexplained. We have not tested 8 × 32k or 32 × 8k in a server, where attention is a larger share.

## What this does not show yet

Why the plugin loses that last millisecond. What fills the remaining 30 ms of a step. Whether the flag wins on our production hardware, where the model and plugin differ. Cache writes are not part of the lab step; Triton's writer costs about 47 µs, a real share of a short step. Real KV data from 3 layers matched random data
within about 1% on the fast layout. We have not raced Composable Kernel's paged path at page 784: it needs page sizes divisible by 128,
which rules out both the old frame's 16 and 784.

The lesson we would pass on is the first one, twice over: before tuning a kernel against a baseline, start the real server with the real
weights and read what it allocated, and then read which kernel it actually calls. A 5× lead on a path the server does not take turned into parity for 0.1 GCD-hours of testing.

*Scripts: `gqa_decode.py`, `vllm_paged_baseline.py` and the harnesses in this repository (`amd-mi250x/`). Our TileLang
compiler fixes are prepared as a patch series; no pull request is open yet.*
