# One kernel source, two vendors: TileLang attention and mixture-of-experts kernels on AMD MI250X versus vendor libraries

*Working draft. Every number here comes from RESEARCH-LOG.md, with its corrections. "TODO" marks results that are not
measured yet. Do not fill a TODO with an estimate.*

## Abstract (draft)

Attention and mixture-of-experts (MoE) layers dominate the cost of serving and training large language models. Their
fastest kernels are usually written per GPU vendor. We study how far TileLang gets on AMD Instinct MI250X (gfx90a, LUMI-G)
against the vendor-backed stack: PyTorch SDPA, Composable Kernel (CK) flash attention, hipBLASLt, and vLLM's decode and
fused-MoE kernels. Ratios are ours over the vendor's throughput, higher is better, except MoE, which is ours over vLLM's
time, lower is better.

Prefill at head size 256, with 6 query heads per KV head, runs 1.10–1.19× CK in the same process at 1k–16k on four synthetic inputs with the landed transposed-V form (1.115–1.581× at 13 further lengths raced in the last round); with the earlier kernel it read 1.06–1.18× on real inputs, 1.10–1.48× in batch and 1.38–1.93× SDPA at 1k–16k. At d128 (clang 22, batch 1) the landed dispatch, a transposed-V form with a register prefetch and a round-toward-zero probability cast (JTPR) at every length that is a multiple of 64 from 1024 to 16384 except three, leads CK in every timed cell up to 6144 tokens (1.027–1.290× at 1024–4096, 1.002–1.037× at 4608–6144), is within 0.5% of even on the worst input at 7168 (0.995–1.013×) reads 0.988–1.011× at 8192 and 0.970–1.005× from 9216 up, trailing on every input from 11264; MHA d128 at 4k reads 0.98–1.00×. These kernels need C edits on TileLang's generated source or frozen generated source from a patched fork; stock TileLang alone reaches 0.93×, 0.79× and 0.80×. Paged decode, on the BF16 page-784 frame read from live vLLM starts, runs 2.39–9.52× vLLM's default path at 4 KV heads and 2.51–23.9× at 2 (graph replay, one seed, three length patterns). Against the 3D Triton kernel a server runs up to a batch threshold it reads 0.953–1.264× (median 1.138) at 4 KV heads per GCD and 0.870–1.308× (median 1.104) at 2, below 1 in 3 cells at each; the 1.10–5.33× we first measured holds only against the 2D form. Wired into a running vLLM an earlier version gave no token-rate gain over stock vLLM with the Triton flag (0.966–1.009× in two opposite-order jobs); a step takes 30–51 ms with only 16 of 64 layers reaching attention, and a plugin loss of up to 1.25 ms per step remains unexplained. Our compiler-generated MoE layer, under graph replay and against vLLM's untuned default Triton configuration (vLLM ships none for MI200), reads 0.706–0.900× vLLM's time at 1–4 tokens, 0.753–0.956× at 1024, and 0.883–0.960× on skewed routing from 2048 to 4096 tokens, while uniform routing there reads 0.892–1.004×, with T = 2700 at parity (0.994–1.004×); all on two synthetic routing seeds, and each new token count costs a compile, which these ratios exclude. Dense GEMM with hand-written assembly launched through TileLang now leads hipBLASLt at every shape raced in the last rounds: 1.037–1.189× at 13 large shapes and 1.109–1.122× at 1024×7680×2048 on one GCD, and 1.007–1.155× at the small-M shapes; compiled TileLang kernels read 0.66–1.08×, so this is not a compiler result. An expert-parallel MoE layer with landed kernels scales at 76% inside a node and 26% at 4 nodes, faster than the old kernels' 99% and 38% in absolute throughput; vLLM across two nodes reaches 0.65–0.87× of one node on prefill only through a hand-written transport adapter, and it hangs intermittently on both transports. We also report pitfalls: an fp16 reference that drifts 4.3% at 4k tokens and masked a correct kernel, silent failures of the direct LDS load, a register-promotion change that slowed decode 8.6–10.3× on one compiler only, and data, placement, noise and power effects that move vendor comparisons by several percent.

## 1. Introduction
- LLM cost is dominated by attention (prefill: compute; decode: KV-cache bandwidth) and, in MoE models, by grouped
  expert GEMMs (small batches: weight bandwidth; large batches: compute).
- EuroHPC runs both NVIDIA and AMD systems, and vendor-specific kernels fragment the ecosystem.
- Question: with one TileLang source plus per-backend tile choices, how close do we get to vendor-tuned code on MI250X,
  and what stops us?
- Contributions:
  1. GQA-packed prefill kernel that beats SDPA, and CK at head size 256.
  2. Flash-decoding on gfx90a, measured on vLLM's real cache frame and in a running server, with a negative token-rate result.
  3. Fused MoE with a GPU-side, host-sync-free alignment.
  4. Split-K small-M GEMM, a large-M schedule that beats hipBLASLt by hand, and the operand placement that makes it hold at the power cap.
  5. A catalogue of gfx90a / TileLang pitfalls with measurements.
  6. A TileLang compiler patch series prepared against upstream main (no pull request is open yet), and a full results log.

## 2. Platform
- **LUMI-G node:** 4× MI250X = 8 GCDs.
- **Per GCD:** 110 CUs; wave64; MFMA 16×16×16 fp16/bf16; 64 KB LDS per workgroup; ~1.6 TB/s HBM; ~191.5 TFLOPS fp16
  dense peak at a nominal 1.7 GHz; no FP8/FP4 hardware. Observed: 156.6 TFLOPS sustained for 25 s, MFMA only, at 1381–1395 MHz
  under all-CU load, one node, neighbour GCD not controlled. Whether 191.5 is reachable under other settings is unverified.
  Package power cap 500 W per two GCDs is reached in real GEMM with both busy; DPM stays active under the site's "manual" setting. That clock limits throughput is shown for the MFMA micro only.
  **Clock readings (9 Nov):** amdsmi's `current_gfxclk` reports the 1650 MHz target, not a measured clock (1812 of 2550 prefill readings are exactly 1650.0), so every prefill "% at clock" figure is unreliable; prefill is quoted as % of the nominal 191.5 TF only. GEMM's clock is `average_gfxclk` of our own GCD from `gpu_metrics` (its package includes the twin GCD's load); decode and MoE use a side probe, so their clocks are inferred, not read inside the timed kernel.
  The hardware has a direct global→LDS load, CK's binaries contain it, TileLang gates it, and we ran it (3e9+ dwords per hazard mode).
- **Software:** ROCm 6.2.4 container, PyTorch 2.7.1, TileLang 0.1.15 (with two gfx90a compiler shims, §7); patched fork where stated. Baselines run
  in the vLLM 0.30.0 ROCm 7.2.3 image (torch 2.12, flash-attn 2.8.3).
- **Method:** correctness against float64 (attention) or float32 (MoE, GEMM); timing with event timers after warm-up.

## 3. Attention prefill
- **3.1 Upstream example and the reference pitfall.** The example looked broken until checked against float64
  (blog 1; RESEARCH-LOG 5 Nov).
- **3.2 GQA packing.** One workgroup = P positions × G heads (P·G GEMM rows), K/V loaded once per group, Q in registers.
  - First tile sweep: P 32, bN 16, 768 threads, k_pack 2, 16 KB LDS.
  - Fewer warps: slower even without spill, up to 1.78×, with one exception at +7.8%; the 2–4× cases include spill and load-path effects that are not separated.
- **3.3 First tile, d256, GQA 6, causal (superseded by §3.5):**

  | tokens | 1k | 4k | 8k | 16k |
  |---|---|---|---|---|
  | vs SDPA | 1.48× | 1.22× | 1.18× | 1.14× |
  | vs CK flash-attn | 0.69× | 0.78× | 0.81× | 0.77× |

  - Batch 4 at 4k: 1.23× SDPA.
  - d128 GQA: 1.10× SDPA, 0.65× CK.
- **3.4 Same-job comparison with CK** (one job per row; TileLang on ROCm 6.2.4, CK on ROCm 7.2.3, so not a same-stack result):

  | Case | base | best TileLang | best with C patches on generated code | CK TF |
  |---|---|---|---|---|
  | d256, 8k | 0.82× | 0.93× (bN 48, edge mask, reverse order) | 0.96× | 89.70 |
  | d128, 4k | 0.67× | 0.78× (bN 96) | 0.82× | 93.23 |
  | d128 MHA, 4k | 0.60× | 0.84× (P 128, bN 32, 512 threads) | 0.83× | 82.05 |

  - Attribution at d256 8k, rule fixed in advance: no single driver. K/V loads are 25–34% of removable overhead at bN 16.
  - The patched column is withdrawn as a headline: patch A was later shown racy by a static check (§3.5).
- **3.5 Same-stack comparison, landed kernels** (TileLang inside the vLLM image, same process as CK, ROCm 7.2.3, clang 22; `gqa_flash_fast`, C edits on generated source; randn input; 2–3 jobs per row; ranges, because job-to-job motion exceeds within-job spread):

  | Case | Kernel | TF | vs CK | vs SDPA |
  |---|---|---|---|---|
  | d256, 1k | P32 bN32 | 59.5–60.4 | 1.16× | 1.91–1.93× |
  | d256, 4k | P32 bN32 | 91.5–94.3 | 1.14–1.15× | 1.48–1.51× |
  | d256, 8k | P32 bN32 | 95.3–98.6 | 1.09–1.10× | 1.38–1.43× |
  | d256, 16k | P32 bN32 | 98.2–100.7 | 1.08× | 1.41–1.44× |
  | d128, 1k | 128 rows bN32 | 59.8–60.7 | 1.19× | 2.31–2.34× |
  | d128, 4k | 128 rows bN32 | 87.9–91.2 | 0.97× | 1.30–1.35× |
  | d128, 8k | P32 bN96 | 90.9–94.4 | 0.92–0.94× | 1.14–1.16× |
  | d128, 16k | P32 bN96 | 93.2–96.5 | 0.92–0.94× | 1.08–1.10× |
  | d128 MHA, 4k | P160 bN32 | 80.5–83.9 | 0.98–0.99× | 1.19–1.21× |

  - **On ROCm 6.2.4 / clang 18 (cross-stack against CK):** 1.07–1.15× at d256, 1.09× at d128 1k, 0.91–0.95× at d128 4k–16k, 0.87–0.96× at MHA.
  - **What produced it:** one joint K+V load block (0.877 to 1.008 at d256 8k), then next-step rows loaded to registers before the V GEMM (to 1.08–1.09), two checked barriers per K step, one row-sum reduce. A static check refuses any kernel with an uncrossed LDS write-read pair.
  - **Stock TileLang alone, same process:** 0.93× at d256 8k, 0.79× at d128 4k, 0.80× at MHA 4k (best plain arm, P192 bN96; 0.75× is one tile). See §3.6 for TileLang source on the patched fork.
  - **Limits:** the d128 1k win is mostly fixed cost (2-point fit, asymptotic about 0.94×). CK always ran first in each case in these jobs. 45% of fp16 peak at 8k and above: d256 50–53%, d128 47.5–50%.
  - **Negative:** longer load gaps give no measurable gain. Reciprocal on MHA gives about +0.2%. Transposed V reaches 1.02× and stays under row-major's 1.10×. bN 64 at d256 fails at launch (65536 B plus reduce workspaces, cause inferred). LDS stride, which was GEMM's largest step, loses at every padded value here (160 B 0.947–0.992 of 128 B); ring buffers, 7 flag sets and launch bounds on clang 22 give nothing usable.
- **3.6 Later rounds** (ours over the same-process vendor, throughput, higher is better).

  | Case | Landed, C edits | TileLang source, patched fork, final tree | vs CK reading |
  |---|---|---|---|
  | d256, 8k | 1.09–1.10× | 1.099–1.101× | win, forms tie |
  | d128, 4k | 0.96–0.98× | 0.996–1.003× | parity at best |
  | d128, 8k | 0.93–0.94× | 0.959–0.967× | below |
  | d128, 16k | 0.91–0.94× | 0.944–0.961× | below |
  | d128 MHA, 4k, randn | 0.98–0.99× | 0.986–1.002×, one job of 5 above 1.0 | below |

  - "TileLang source" means no text edits on generated C, on upstream main plus our patches. The forms (joint or staged load, a reducer rescale, `T.__exp`) are hand-structured; the patches make them expressible and do not derive them. Without them, on the same fork before the last two patches: 0.75–0.81×, 0.83–0.94×, 0.48–0.72×.
  - **Other lengths:** d256 wins at every tested length (3072–12288 lowest 1.084×, median 1.135×). d128 with the best tile per length reads 1.200, 1.196, 1.045, 1.019× at 1024, 1536, 2048, 2560, then 0.992–0.998× at 3072 and median 0.960× above; no code yet chooses that tile, and one tile has a 0.884× hole at 4160.
  - **Real inputs** (27B layers 3, 15, 31): d256 1.06–1.18×, d128 0.90–0.98× (derived), MHA 0.97–1.01× (statistics only). Error reaches 4.06e-3, over our 2e-3 limit, while staying at or below CK's on all 15 cases and within 4% of SDPA; the limit is an open decision.
  - **Batch:** d256 1.10–1.48×; d128 rows 128 1.12–1.23× at 1k, 0.935–0.968× at 4k and 8k. MHA on randn with P128: 1.021–1.023× at 4×1024, 1.002–1.004× at 16×1024, 0.925–0.938× at 4×4096 and 2×8192, and 1.04–1.06× SDPA at the long ones.
  - **MHA length rule:** P160 falls from 0.984× at 4320 to 0.696× at 4608, and a rule switches to P128 at 220 blocks. Timed on clang 22, randn; clang 18 is mostly untimed under it, and it is a module helper, not yet reachable from the script's command line.
  - **1.2× SDPA at d128:** image 8k straddles (TileLang source 1.164–1.216× over 7 jobs, below 1.2 in 4; SDPA moves less than ours and CK), image 16k misses (1.06–1.12×), LUMI 16k straddles by job or hardware instance (1.17–1.272× on 4 nodes; causes not separated, tiles never swept on clang 18), LUMI 8k passes (1.29–1.31×). CK at 16k reaches 1.2× in 10–12 of 51 random-input processes and 12–13 of 18 on real inputs, depending on read convention, so the target is not generally above the vendor.
  - **Clang 18:** launch bounds gain +6.7 to +10.7% on the C-edit P160 at MHA 4k and lose 22–39% at 12 waves; they do not carry to TileLang-source forms (flat to −2.6% on the main arms). MHA 8k clears 1.2× SDPA on randn in 3 jobs on 2 GCDs (lowest 1.202×), mostly because SDPA is slower on that stack; clang 22 stays faster in absolute TF.
  - **Open, superseded in part by §3.7:** MHA 4k has no cheap lever on the image. A 32×32×8 rewrite is excluded as costly, not closed.
- **3.7 Final rounds** (ours over the same-process vendor, throughput, higher is better; clang 22, image, vendor read as mean of pre and post):

  *Superseded for the d128 dispatch by §3.8: form J was landed for 2816–3584 on 9 Nov (JPORT), then replaced by JTP everywhere except 3200.*

  | d128, L | TileLang source form, rows 128 bN64 J: randn0 | `ar` | `big` | C edits, landed tile |
  |---|---|---|---|---|
  | 2816 | 1.031–1.037 | 1.043, 1.041 | 1.031, 1.040 | 0.988–1.001 |
  | 3072 | 1.028–1.030 | 1.034, 1.033 | 1.021, 1.030 | 0.994–0.997 |
  | 3584 | 1.007–1.011 | 1.013, 1.016 | 0.997, 1.009 | 0.964–0.976 |
  | 4096 | 0.999–1.001 | 1.003, 1.004 | 0.984, 0.992 | rows 128 keeps; rows 256 0.974–0.976 |

  - **Six inputs, 2 opposite-order jobs, fork process (randn0, randn1, self, big, ar, ramp):** J/CK 2816 1.017–1.038, 3072 1.015–1.036, 3584 0.991–1.015 with `big` a tie or loss, 4096 0.978–1.005 with `big` a loss. J over the landed tile 1.029–1.046. The landed-tile column of the table above stands.
  - **J at 8k and 16k, same process, 6 processes, 36 cells:** 8k 1.172–1.230× SDPA, at or over 1.2 in 16 of 36, 0.938–0.972× CK, 1.006–1.045× the landed tile; on randn alone 4 of 17 reads over 9 jobs reach 1.2, median 1.186, one of them on a low SDPA read. The job or instance decides, not arm order. 16k 1.093–1.151× SDPA, 0 of 36, 0.920–0.957× CK. So J does not reliably close the 8k row. Rows 256 forms read 0 of 36 at 8k.
  - **Reaching the wheel.** J's bytes through the wheel's plain wrapper read 0.898–0.920 of native; the wheel asks 36864 B LDS per block where the fork form asks 32768 B, and two blocks per CU fit only at 32768 B (causal reading inferred; dispatch not measured). A stub wrapper asking 32768 B on the stock wheel reads 1.017–1.037× CK at 3072 on 3 inputs, 0.940–0.972× at 8192; 2816, 3584 and 4096 have instruction identity only. Hand rows 128 bN64 (132 VGPRs, occupancy 3 computed) reads 0.734–0.853× CK and bN48 0.954–0.991 of bN32 natively, so neither is a table entry. No code path picked J at the time; JPORT landed it for 2816–3584 on 9 Nov, and §3.8 then replaced it everywhere but 3200. Output on these arms at most 2.74e-4 against fp64 up to 8192, 0 errors in 578 reads. Bytes are kept.
  - **Landed tile tables:** `d128_tile` chooses rows 128, P32 or rows 256 per compiler at 16 timed lengths (+1.2 to +2.1% for rows 256 from 6144, randn0, order not balanced at 6144, 8192, 10240); clang 22 rows 256 reads 0.946–0.971× CK from 5120 up. `mha_tile` replaces a 220-block rule that lost 16.6–29.3% at 8 of 12 cases; P160 reads 1.20–1.25× at 2048, ties at 3584 and 2×2048, 0.983–0.997× at 4096. MHA batch on `ar` and `big`: 1.020–1.034× at 4×1024, 1.002–1.016× at 16×1024, 0.915–0.942× at 4×4096, 0.899–0.929× at 2×8192.
  - **1.2× SDPA, the negatives, with ours/SDPA = ours/CK × CK/SDPA, image, randn0:** at 8k 0.955–0.958 × 1.217–1.225 = 1.165–1.170, a real miss of a target CK meets; D128's direct reads of the same landed form are 1.163–1.176 on randn0 and 1.139–1.178 across six inputs, so closing the row needs up to +5.4%. At 16k 0.949–0.953 × 1.155–1.165 = 1.098–1.108; reaching 1.2 needs 1.030–1.039× CK's own throughput, while CK clears 1.2 in 12 of 51 random-input processes. On real inputs CK clears it in 13 of 18, so the vendor misses too on random inputs only.
  - **Clang 18, LUMI 16k:** a 60-arm tile sweep (no correctness checks in that job) found nothing better than the finalists (at most 1.002×); confirm arms 1.205–1.220× on 2 nodes, landed P32 1.178–1.210×, with a 2.4% factor between GCDs on identical bytes. Closed.
  - **Instance effects:** all 24 reads under 0.9× an arm's median across 1,366 reads sit in 2 jobs on one GCD and node tag, and hit SDPA and CK as well as ours.

- **3.8 Kernel-max rounds, 9 Nov** (`evidence/PREFILL-MAX/`, rounds 1–5; ours over the same-process CK, flash-attn 2.8.3, throughput, higher is better; clang 22 on the image, batch 1, G 6, Hkv 4, causal; vendor read as the mean of pre and post; ranges are per cell over inputs and two opposite-order jobs). Landed `gqa_fa.py` md5 bb0808db (was ffc58f6a, e3caf113, then 98b6e5ec).
  - **Lever.** V goes to LDS transposed with an XOR swizzle, so each PV MFMA operand is one 8-byte LDS read instead of 128 `ds_read_u16` per K step (d128 form JT; d256 `vt`). Form JTP adds a register prefetch of the next K/V step: 126 VGPRs, occupancy 4 kept. The two levers in JTP were separated only at 4096–16384.
  - **Dispatch now.** `d128_tile` picks JTPR (JTP with the P cast by `v_cvt_pkrtz_f16_f32`, round toward zero, two values per instruction) at every L % 64 == 0 from 1024 to 16384 except 1088, 1536 and 3200, which keep the previous dispatch (rows 128 bN32, P32 bN96, form J). `d256_tile` picks `vt` at every L % 32 == 0 from 1024 to 16384 except 1056 and 5152; nothing in production calls `d256_tile` (only the module CLI). Clang 18, B > 1 and MHA are unchanged.

  The table below is JTP (rounds 1 and 3) before the JTPR cast; rounds 4–5 replaced JTP with JTPR at 238 lengths, and the JTPR/CK ratios measured at the 22 timed lengths are in the second table. Only 22 of the 238 changed lengths are timed; the other 216 are interpolated [inferred] and fp64-checked.

  | d128, L | JTP / CK | source |
  |---|---|---|
  | 1024 | 1.253–1.274 | round 1, 3 inputs; round 3 1.255–1.268 |
  | 1152–2048 (14 lengths) | 1.166–1.267 | round 3, 3 inputs |
  | 2240 | 1.133–1.166 | round 3 |
  | 2816–3584 (12 lengths) | 1.026–1.101 | round 3 |
  | 4096 | 1.015–1.028 | round 1, final module |
  | 4608 | 1.000–1.026 | round 3 |
  | 5120 | 0.994–1.020 | round 1, research sweep |
  | 5632 | 0.991–1.014 | round 3 |
  | 6144 | 0.990–1.007 | round 1, one job |
  | 7168 | 0.973–1.002 | round 3 |
  | 8192 | 0.976–0.995 | round 1, final module |
  | 9216–16320 (7 lengths) | 0.960–0.992 | round 3 |
  | 16384 | 0.958–0.975 | round 1, final module |

  | d128, L | JTPR / CK (min–max over 4 inputs, 2 jobs) [computed] |
  |---|---|
  | 1024 | 1.272–1.290 |
  | 1280 | 1.187–1.190 |
  | 1600 | 1.249–1.286 |
  | 2048 | 1.177–1.201 |
  | 2240 | 1.156–1.173 |
  | 2560 | 1.098–1.107 |
  | 3072 | 1.068–1.079 |
  | 3584 | 1.042–1.056 |
  | 4096 | 1.027–1.043 |
  | 4608 | 1.019–1.037 |
  | 5120 | 1.015–1.031 |
  | 6144 | 1.002–1.019 |
  | 7168 | 0.995–1.013 |
  | 8192 | 0.988–1.011 |
  | 9216 | 0.984–1.005 |
  | 10240 | 0.981–1.003 |
  | 11264 | 0.978–0.998 |
  | 12288 | 0.975–0.995 |
  | 13312 | 0.974–0.995 |
  | 14336 | 0.972–0.993 |
  | 15360 | 0.971–0.992 |
  | 16384 | 0.970–0.993 |

  - **JTPR over JTP [computed], jobs 23575843 and 23575844, noise 0.36% and 0.63% (identical-code spread).** 1.005–1.020 in all 176 cells, every cell above 1 + that job's noise, at every timed length. The P cast rounds toward zero, so error rises from JTP's 1.20e-4 to 2.60e-4 to at most 4.73e-4 against fp64 on randn0, big, ar and ramp, CK's level (open decision 9 Nov).
  - **Form M4HLR (32×32×8 MFMA with the same cast) is built and fp64-checked but not dispatched.** At L 8192–16384 it reads 0.982–0.991 of JTPR on `ar` (beyond noise) and 1.006–1.022 on randn0 and big; mean gain over JTPR -0.14% to +0.92%. The input rule read against the kernel it would replace (JTPR) fails at all 9 lengths; that reference was fixed in the frozen rule before the races, and the per-cell `ar` cost against JTPR at 10240–16384 is 0.86–1.60%. Against the previous JTP alone it passes at 10240–16384 (`ar` 0.997–1.003, mean +1.5–2.0%), fails at 8192 and 9216 (`ar` 0.991–0.996).
  - **Standing against CK at 8192 and up [computed, min–max over 4 inputs, 2 jobs]:** JTPR 0.970–1.011; M4HLR would read 0.979–1.010 (randn0 1.002–1.010, ramp 0.995–1.005, big 0.982–1.004, `ar` 0.979–0.992). Neither form beats CK on all inputs from 8192 up.
  - **Kept old kernel, by rule.** At 1088 JTP lost to the old kernel (0.959–1.021); at 1536 and 3200 it read above the old kernel in every cell (1.013–1.034, 1.017–1.032) but not by more than the identical-code spread of the forward job (1.8–2.3%), and the rule fixed in advance keeps a length only if it wins by more than that spread in every cell. The old kernels there read 1.216–1.299 (1088), 1.177–1.198 (1536) and 1.013–1.031 (3200, form J) of CK. Why JTP loses at 1088 only is not traced.
  - **d256 `vt`.** Round 1, final module, four inputs: 1024 1.169–1.188, 4096 1.172–1.187, 8192 1.128–1.158, 16384 1.099–1.147 of CK (was 1.10–1.11 at 8k). At the 13 lengths kept in round 3: 1.115–1.581 (1.581 at 1568, where CK is weak). At 1056 and 5152 the old tile reads 1.229–1.262 and 1.145–1.181.
  - **Correctness [computed].** fp64 check at every dispatched length (job 23574147): d128, all 241 lengths, inputs randn0 and big, overall relative error 1.20e-4–2.60e-4, worst row at most 0.65 of CK's worst row on the same input, 0 non-finite, 3 of 3 repeats identical. d256 `vt`, 17 lengths (15 with L % 64 != 0): at most 2.58e-4. CK reaches 4.72e-4. JTPR (rounds 4–5, fp64, 238 lengths): at most 4.73e-4 overall relative error on randn0, big, ar and ramp; per-row maxima up to 1.45× CK's on single cases at the same input; JTPR's worst row across inputs exceeds CK's at 54 of 129 lengths (8192–16384) and 45 of 109 (1024–8128), by at most 1.15× (L 11200, ramp, 1.12e-3 against CK's 9.75e-4; CK's own largest, 1.21e-3, is a different case, L 16320, ramp, JTPR 1.07e-3); 0 non-finite, 3 of 3 repeats identical.
  - **Throughput [computed].** % of the nominal 191.5 TF, per cell: d128 at 16320 51.9–60.3% (CK at 16384 52.8–62.8%); d256 at 16384 53.5–61.3%. Throughput spreads about 10 points from input to input at one length, so no single mean is quoted. The "% at clock" figures in the round 1 and 2 records used amdsmi's `current_gfxclk`, which is the 1650 MHz target, and are **unreliable**; they are not a hardware standing.
  - **Counters, d128 8192, JTP against CK** (rocprofv3, one process, diagnostic): MFMA busy cycles equal (1.62e9 against 1.64e9), while ours issues 2.0× the MFMA instructions, 2.3× VALU, 2.1× LDS instructions and 14× the LDS wait; ours has no LDS bank conflicts, CK 5.1e7. The gap is issue overhead and LDS wait [inferred].
  - **Not dispatched.** Form M4HL/M4HLR (32×32×8 MFMA; round 2 read 0.916–0.964 at 2048 against JTP, M8HLZ 108.9 against M4HL's 108.2 TF mean at 16384); available only by an explicit `form=` call. The `pkrtz` cast on d256 `vt` (`pk`): 0.970–1.003 of `vt` in 72 cells, 3 above 1, none above noise; removed from the module. JTPX and JTPS are closed (ties or losses).
  - **Noise.** Each round-3 job raced identical code twice; the forward-order jobs read 1.6–2.3% between identical arms against 0.35–0.64% in reverse (d256 pair 2: 1.67%), cause not traced. Lengths between timed points (for example 2112–2752 and 3648–4544) are interpolated for speed, though all are fp64-checked.
  - **Not tested.** Clang 18, B > 1, d128 L % 64 != 0, L outside 1024–16384, real captured inputs and in-server use. GPU use 1.16 GCD-hours over rounds 1–3, 0.72 over rounds 4–5 (including the 0.15 GCD-hour d256 `pk` check cancelled once `pk` was dropped).

## 4. Attention decode
- **Design:** flash-decoding (cache split into slices, combine kernel); vLLM-style paged cache read through 16-key tiles;
  6 query heads padded to 32 rows so two wavefronts share each K/V tile; variable lengths.
- **Steps that mattered:** occupancy (16-key tiles, 16 KB LDS) and wavefronts per tile (padded rows):
  8–13% → 34% → 56–59% of HBM.
- **Negative results:** SIMT kernel (10–50× slower); larger key tiles; simulated FP8 KV (1.9–2.2× slower, 3.8% error).
- **Baselines and frame.** Early results assumed page 16 and FP16; they are superseded. Live vLLM 0.30 starts with real 27B weights show the default ROCM_ATTN on layout LBHNC, physical shape [blocks, 2, 784, Hkv·D] with K then V inside each page, a generic kernel (the native one rejects head size 256), **page 784, BF16**. Mixed head shapes appear to drop LHBNC [inferred from source]. `--attention-backend TRITON_ATTN` gives LBNHC at the same page. One GCD at utilisation 0.9 and a 9000-token model limit holds 111 pages, about 87k tokens per layer, so 1 GB caches are kernel races only there. Tensor parallel 2 gives Hkv 2 and 1269 pages.
- **Label on every Triton ratio in this section outside §4.2:** our lab called `unified_attention` without `seq_threshold_3D` or segment buffers, so it timed Triton's 2D kernel. Serving passes them and runs the 3D 16-segment kernel for B up to 64 at Hkv 2 and 32 at Hkv 4 (read from vLLM's source). Against that kernel these ratios do not hold; §4.2 replaces them for B at or below the threshold. ROCM_ATTN ratios are a different kernel and stand.
- **Results** (vLLM ms over ours, higher is better; same job, base lengths, BF16, ms; two jobs, ROCM_ATTN on one seed; random data):

  | B × L | ours, LBNHC (with the TRITON flag) | Triton | ROCM_ATTN | ours, `rocm_hybrid`, no vLLM config change |
  |---|---|---|---|---|
  | 8 × 8k | 0.206–0.209 | 0.630–0.632 | 1.600–1.605 | 0.415 |
  | 16 × 4k | 0.216–0.217 | 0.330–0.332 | 0.804–0.808 | 0.416–0.417 |
  | 32 × 2k | 0.215 | 0.246–0.248 | 0.801–0.804 | 0.402 |
  | 32 × 1k | 0.117 | 0.129–0.131 | 0.406–0.407 | 0.206–0.207 |
  | 64 × 2k | 0.404–0.405 | 0.472–0.473 | 1.204–1.209 | 0.793–0.796 |
  | 64 × 1k | 0.212 | 0.242–0.244 | 0.610–0.612 | 0.392–0.394 |

- **Across six length patterns on the four shapes a GCD can serve, final bytes:** 1.10–5.33× 2D Triton (seeds 988–991, 72 draws per job), 1.07–3.98× on the bytes before the last gate edit (seeds 980–987, 136 draws per job), none below 1; 2.79–13.47× ROCM_ATTN. Lab Triton alone is 2.09–3.33× ROCM_ATTN, so the flag accounts for much of that second range. `rocm_hybrid` is 1.33–4.76× ROCM_ATTN and 0.54–1.87× Triton, at 29–39% of HBM (not re-measured on final bytes), so it is a way to run with no setting change rather than the fast path.
- **Near parity:** dead-split skip gains about 5.5% same-job at 64 × 1k skew, gated off where it loses (base and full lengths, small batches). Pooled over 28 seeds that cell reads median 1.070×, minimum 0.994×, about 3% of draws below 1, on mixed bytes. The lowest cell is now 64 × 1k mid, medians 1.050–1.060×, minimum 1.014×, where skip gains about 0. An earlier FP16 result of 8/8 wins did not hold on BF16.
- **HBM:** at Hkv 4 the lowest cell is 51.75–51.95% (32 × 1k skew) and base lengths read 63–73%. Computed from final-byte timings by formula, Hkv 4 cell medians read 50.7–52.1% with single draws down to 47.5%, so the target is met by median with little margin. Hkv 2 short skew reads 39–41% on those seeds; an earlier 8 × 8k median of 48.0% has draws down to 32.7%, so it is seed-sensitive. FP16 32 × 1k at base and full reads 1.097–1.147× with S5 on one seed through `paged_uniform`, not the picker; FP16 is not the served dtype.
- **Hkv 2, per GCD, no communication cost:** 1.335–7.97× 2D Triton on final bytes, six cells; 1.095–8.47× on earlier bytes, ten cells; 8 × 32k 0.40–0.44 ms against 2.47.
- **What moved it:** all 8 K+V loads issued in one round (k1 1.004 to 0.803 ms at batch 32), one division per row, split counts per shape (uniform 5 at 32 × 1k gains 11–13% over 4), and branchless BF16 conversion: the hip_bfloat16 cast adds an exec-mask branch per element (30 against 22 saveexec), and removing it gains a median 2.4%. Why the best split count moves with active blocks (320, 640, 1280 over 384, 512, 768, up to 1.32×, with one exception) is a hypothesis.
- **Correctness:** at most 2.24e-3 in BF16 (vLLM 2.30e-3), 2.7e-4 in FP16 (vLLM 2.9e-4), one seed per cell; no NaN. Dispatched S3, S6 and S8 skip forms have no error row on any seed. On real KV from 3 layers, speed is within about 1% of random data on LBNHC.
- **Other:** K-then-V staging helps on LBHNC by up to 13.2% and removes a 10% flip there, cause unknown. Triton's cache writer costs about 47 µs, about 30% of a short step. A 32-key tile gains 8–18% at page 800 and cannot deploy at 784.
- **Refuted earlier:** the model's 1.73× forecast (measured 1.12–1.32×), random-page traffic as the limit, and the 64-thread TileLang form (555 GB/s).
- **Limits:** CK's paged decode needs page sizes divisible by 128 (784 is not), so it has not been raced at page 784.
- **4.1 In a running vLLM** (arm over stock, `win_tok_s`, higher is better, median of rounds, two opposite-order jobs, TP=2 on 2 GCDs, real 27B weights, BF16):

  | Pair | short ×1 | short ×8 | short ×32 | 2k ×1 | 2k ×8 | 2k ×32 | 8k ×1 | 8k ×8 |
  |---|---|---|---|---|---|---|---|---|
  | ours in the FULL graph / stock ROCM_ATTN | 1.012, 1.012 | 1.010, 1.013 | 1.011, 1.009 | 1.160, 1.168 | 1.103, 1.101 | 1.075, 1.073 | 1.771, 1.777 | 1.485, 1.468 |
  | stock TRITON_ATTN / stock ROCM_ATTN (flag only) | 1.022, 1.050 | 1.015, 1.025 | 1.014, 1.025 | 1.219, 1.242 | 1.126, 1.138 | 1.098, 1.114 | 1.834, 1.852 | 1.520, 1.517 |
  | ours in the FULL graph / stock TRITON_ATTN | 0.991, 1.009 | 0.995, 0.991 | 0.994, 0.992 | 0.966, 0.976 | 0.983, 0.985 | 0.994, 0.994 | 0.981, 0.991 | 0.996, 0.991 |
  | ours through a forced PIECEWISE hook / stock TRITON_ATTN | 0.973, 0.973 | 0.981, 0.977 | 0.983, 0.987 | 0.976, 0.976 | 0.984, 0.981 | 0.977, 0.980 | 0.966, 0.975 | 0.979, 0.986 |

  - **Closed:** no token-rate gain over stock Triton in the forms tested. The single read above 1 reverses in the other order. Under vLLM's default FULL-graph mode a Python hook is inert, which a first job showed only by reading counters; forcing PIECEWISE or recording the kernels into the graph was required.
  - **Why the kernel ratio does not carry over:** the lab's kernel was the wrong one, raced in §4.2. Second, a step takes 30–51 ms, context length adds 1–2 ms, and 16 of 64 layers reach this hook. The 27B has no MoE; what fills the rest of the step is unprofiled.
  - **Correctness:** eager verify at most 1.28e-3, no NaN, including padded rows; FULL-graph arms cannot verify in a graph, and prefix logprob differences match stock against stock (p99 0.053–0.054 against 0.055). Temperature-0 text differs from stock on about as many rows as stock differs across jobs, with no same-job stock control, and 64 GSM questions cannot resolve a small quality effect.
  - **Cost:** 9.36 GCD-hours.
- **4.2 Against the 3D kernel** (vLLM ms over ours, higher is better; 2 opposite-order jobs; same frame; 0.107 GCD-hours). Eager `tri3` floors at about 0.13 ms whatever the shape, against 0.012 ms on replay at short ×1, and serving replays a FULL graph on decode steps, so the graph column is the relevant one. It covers seed 988, base, mid and skew. Our replay arm is a kernel fitted to one length set, which FULL-graph serving cannot deploy, so it probably flatters ours.

  | Hkv, B × L | 2D Triton / ours | 3D Triton / ours, graph |
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

  - **Replacement ranges:** on the 72 draws per job behind "1.10–5.33×", eager 3D reads 0.870–1.841 and 0.876–1.830, 4 draws below 1 per job, all at 8 × 8k; so Hkv 4 is at or below about 0.87 up to 1.37. Hkv 2 is 0.881–1.300, and V1's cells on replay go down to 0.862 (short ×1) with the length-fitted `pickg`; the unfitted `kkg`, the form behind the headroom, reads below 1 in both jobs at 5 of 8 V1 cells, lowest 0.856, 0.857 at 8 × 8k, and eager `tri3/kk` at 4 KV heads reads 0.910–1.767 with 9 and 10 of 72 cell medians below 1. Hkv 2 B = 64 is not raced. Hkv 4 B = 64 stays 2D in vLLM, 1.06–1.27× eager only.
  - **Closure against §4.1:** 16 layers × kernel ms lands within 0.8–2.2 ms of observed ROCM minus Triton on the 3D kernel; the 2D kernel misses 8k by 9.2–10.7 ms. That serving runs 3D is inferred from source and this closure; no running server was instrumented. Headroom of ours over 3D is −0.31 to +0.26 ms per step, at most 0.8%; Triton 3D attention is at most 1.84–1.98 ms of 46–50 ms, so a zero-time kernel gains at most about 4%. All of this holds in V1's eight cells only; at 8 × 32k and 32 × 8k no step was measured.
  - **Not explained:** the forced-PIECEWISE plugin's loss of 0.64–1.22 ms per step, after launch and PIECEWISE costs, leaves −0.04 to +1.25 ms; at 8k ×1 the model predicts a small gain. The in-graph arm has neither cost and still reads +0.18 to +1.06 ms outside short ×1; its S = 128 form may be why (hypothesis, not raced).
  - **Correctness:** no NaN; `tri3` at most 2.23e-3, `tri` 2.30e-3, ours 2.24e-3.

- **4.3 Kernel-max rounds, 9 Nov** (`evidence/DECODE-MAX/`; frame: vLLM 0.30 image, BF16, page 784, LBNHC, G 6, D 256, seed 988, base, mid and skew lengths, every arm replayed from a CUDA graph; vendor ms over ours, higher is better). Landed `gqa_decode.py` md5 b4cfec43 (was 0603b766). Supersedes the 3D-Triton ranges in §4.2 for these 14 shapes; §4.1 (no token-rate gain in a running vLLM) was not rerun on the new bytes.
  - **Dispatch.** Six BF16 page-784 shapes get new forms; every other call runs the 0603b766 code. `decode_bal` (stream-K over a fixed grid of 330 blocks, 3 resident × 110 CUs) at Hkv 2 (16,256), (16,512), (32,512); `decode_seg` (length-aware splits per sequence on a fixed grid) at Hkv 4 (8,512), (16,256); uniform splits of 5 (`u5`, 320 blocks) at Hkv 2 (32,64), +14.3 to +15.2% base, +8.6 to +10.0% mid and +0.7 to +2.1% skew over `bal` across three jobs. Six round-1 `v4` entries (loads issued before the LDS barrier) led in all 36 cell-job readings by +0.3 to +2.2% but were reverted: with 9 cycles per cell and an identical-code control arm, no entry beat the control spread (1.75% and 0.49%) on base, mid and skew in both jobs [observed]. v4 is probably a real gain of about 1% that this harness cannot certify [inferred].

  | Hkv | vs Triton 3D as served | median | readings below 1 | vs ROCM_ATTN | HBM share, lowest / median / highest |
  |---|---|---|---|---|---|
  | 4 | 0.953–1.264 | 1.138 | 5 of 36, in 32×2k skew and 64×1k mid and skew | 2.39–9.52 | 43.0 / 60.5 / 71.8% |
  | 2 | 0.870–1.308 | 1.104 | 6 of 48, in 8×8k base and 8×32k base and mid | 2.51–23.9 | 37.7 / 53.7 / 74.8% |

  - Readings are cell-job readings over two opposite-order jobs (23574143/144 for the 7 re-raced shapes, 23569886/888 for the 7 unchanged ones); HBM share takes the lower job per cell. Before these rounds the same frame read Hkv 4 median 57.9% and Hkv 2 median 52.0% of HBM.
  - **Ceilings [observed].** Plain streaming on this GCD reaches 80.2% (`sum`) and 78.7% (`copy`) of 1.6 TB/s, so the 80% goal equals the streaming ceiling. The kernel's memory skeleton without attention math reaches 72.7–75.1% on long shapes and 62.5–67.7% at 32×1k; the full kernel's long base cells read 67–75%, so the remaining gap sits in the access pattern, not the math [computed]. Every way tried to put more bytes in flight per CU lost (cross-tile prefetch −0.2 to −14%, paired tile loads −5.3% mean, a fourth block per CU mixed); a combine fused into the first kernel lost 9.0% on average.
  - **Correctness.** Error against fp64 in the final jobs: ours 2.05e-3–2.19e-3, Triton 1.97e-3–2.24e-3, ROCM_ATTN 2.10e-3–2.28e-3; 0 NaN on live rows. Padding rows of length 0 give zeros on `bal`, `seg` and `u5`, and NaN on every other shape, as the 0603b766 code always did; the vLLM overlay copies only live rows eagerly and applies `nan_to_num` when capturing.
  - **Clock.** A probe kernel run right after each arm reads 414–1406 MHz, median about 935 MHz; it never runs during an arm, and amdsmi's `current_gfxclk` reads the 1650 MHz target. Decode is bandwidth-bound, so the MFMA ceiling does not apply.
  - **Caveats.** One seed and three length patterns; even 5- to 9-cycle medians keep identical-code tails near 1%. The landed module installs a process-wide compile hook (`_install_regvec_hook`) that edits only kernels declaring `bfloat16_t Kr[`. GPU use 0.90 GCD-hours over the three rounds.

## 5. Fused MoE
- **Design:** expert-sorted grouped GEMM; gate/up fused with SiLU·up; down projection with the routing weight;
  GPU alignment using LDS atomics with a fixed upper bound on rows (no host sync).
- **Shape:** read from the Qwen3.6-35B-A3B checkpoint: H 2048, I 512, E 256, top-8. Real weights, inputs and routing sampled at layers 0, 19, 39 over 100 conversations; top-1 expert share mean 1.9%, max 6.05%, so zipf2 overstates skew. Captured through a transformers BF16 forward, not vLLM.
- **Kernel-max rounds, 9 Nov** (`evidence/MOE-MAX/RESULTS.md`, rounds 1–6; ours over vLLM `fused_experts`, **time, lower is better**, same process, vLLM 0.30 image, clang 22, one GCD per job, two opposite-order jobs, synthetic routing seeds 7 and 8). Landed `moe_fused.py` md5 8faee1ee (was 9ad0c345 before these rounds; the vLLM rows below were measured on 25a1b44a, which has the same dispatch at every T in the table). These rows replace the T ≤ 4096 standing below; T ≥ 6144 was not re-raced. **Baseline: vLLM 0.30 ships no fused-MoE config for MI200/MI250X (none of its 335 config files), so vLLM runs its untuned default Triton config here (16×64×128 blocks at T ≤ 24, 128×128×64 at T ≥ 1024); a tuned config could change every ratio in this section [observed; not measured].

  | T | Under graph replay |
  |---|---|
  | 1 | 0.849–0.900 |
  | 2 | 0.766–0.855 |
  | 3 | 0.793–0.842 |
  | 4 | 0.706–0.859 |
  | 24 | 0.863–0.925 |
  | 1024 | 0.753–0.956 |
  | 2048–4096, uniform | 0.892–1.004; T = 2700 0.994–1.004, parity |
  | 2048–4096, zipf1 | 0.883–0.935 |
  | 2048–4096, zipf2 | 0.933–0.960 |

  - **What landed.** Decode-size alignment and FullCol tiles at small T (round 1); a GEMV decode path at T = 1, 3, 4 (round 2); down projection and sum fused at T = 1 and a de-duplicating GEMV at T = 2 (round 3); a fused down+sum GEMV at T = 2 and 3 (round 4).
  - **Against the pre-workflow module itself** (round 6: 9ad0c345 imported in the same process, graph replay, 2 opposite-order jobs, seeds 7 and 8, identical-code spread 1.34% and 0.97%; a T is kept only if it wins beyond that spread in every cell of both jobs). Kept: T = 1 0.731–0.744, 2 0.797–0.835, 3 0.753–0.943, 4 0.798–0.973 (zipf2 0.907–0.973), 5–8 0.825–0.976, 20 0.932–0.968, 23 0.932–0.950, 24 0.938–0.962; with fp32 output T = 1–4 0.747–0.975. **Reverted to the pre-workflow tile at T = 13, 15, 16, 17, 19 and 21:** zipf2 lost there (up to 1.011) or tied, although uniform and zipf1 won by 3–10%. The reverted T run kernels whose sources equal the pre-workflow module's (md5 of all five). T = 9–12, 14, 18 and 22 keep the decode tile from round 5 (39 of 42 cells won, 0.83–0.99), which used a looser rule; under the round-6 rule four seed-7 cells there (T = 9 and 14 zipf2, T = 11 and 12 zipf1) would fail, so 38 of the 42 cells pass and these T are open.
  - **Reverted after review.** Round 3's band lever, two tile shapes in one gate/up launch at T = 2200–4096, read 0.887–0.949 of vLLM at T = 2700 but lost to the 64-row kernels beyond identical-code noise (0.996–1.007) in both jobs at four cells of skewed or uniform routing (up to 1.011); dispatch cannot see routing, so it stays opt-in (`cfg={"gu_mix": 96}`) and the band is back at parity on uniform routing. Round 4's untimed fp32-output fallback at T = 2 and 3 was also reverted.
  - **Graph replay only.** Eager ratios (0.16–0.30 at T = 1–2) mostly measure vLLM's host overhead (about 345 µs eager against 69 µs replayed) and are not headlined.
  - **Correctness** (23574603, fp64 per-expert reference, 240 cases per output type): fp16 output worst 3.874e-4, bitwise equal to the previous module; fp32 output worst 3.303e-4; no non-finite values. Round 6 (23576076, T = 1–26, 156 cases per output type): the same worst values, 0 non-finite, output bitwise equal to 25a1b44a in every case including the 6 reverted T.
  - **Throughput [computed].** T = 2700 uniform 48.3–48.6 TF and T = 4096 uniform 59.6–59.8 TF, about 31% and 38% of an MFMA ceiling of about 157.7 TF scaled to a probe-inferred clock of about 1400 MHz (25.2–31.2% of the nominal 191.5 TF). Weight bandwidth at decode size under graph replay, seed 7 only: T = 1 53.1–53.6% of 1.6 TB/s (vLLM 45–47%), T = 2 58–66%, T = 3 59–70%. The clock comes from a separate 1-CU probe kernel on a side stream, not from inside the timed kernel, and it is bimodal: 38 of 144 band reads in the final jobs look idle (about 1630–1640 MHz). Treat the clock-scaled shares as inferred.
  - **Open.** A band lever that does not lose on skewed routing, and the same at the 6 reverted small T (for example a routing-aware choice on the GPU after alignment); real routing captures; a tuned vLLM config as the baseline; T = 9–12, 14, 18, 22 under the round-6 rule. GPU use 4.21 GCD-hours over the six rounds (15145 s).
- **Final rounds, quiet jobs, same convention, landed import** (23525750, 23525751, p90 spread 0.26%, 0.41%; real 23525899): T = 2049–4096 uniform 0.892–0.998 at 14 T; T = 2800 0.995, 0.998, fastest call 1.001, 1.003; zipf1 and zipf2 0.892–0.947. Real T = 1280–4096 0.858–0.966, T ≥ 6144 0.976–0.991; noisy jobs read up to 1.034. Losses above the goal: T = 7168 uniform 1.009, 1.013; 12288 1.001, 1.005; 16384 1.003, 1.007; zipf1 there 0.968–0.997. Square beats FullRow in all 50 cell-job readings by 0.52–2.99%. These replace the noisy rows in the table below.
- **Fine sweep, T = 2600–3000 in steps of 25** (MOESWEEP; 2 opposite-order pairs, seeds 7 and 8; p90 spread 0.39–0.68%; one routing draw per T, adjacent T jump 1–2%): near parity, not a clean loss or win. A median loss in both jobs of a pair appears only at T = 2625–2800 on seed 7, largest T = 2700 at 1.0140, 1.0090, and in a second seed 7 pair at T = 2700 (1.0252, 1.0052) and 2800 (1.0128, 1.0001). Seed 8 shows none. 4-job mean at most 1.0060; T = 2750 reads 1.0090, 1.0019 on seed 7 and 0.9897, 0.9836 on seed 8, so the routing draw moves it about 2 points. T = 2900–3000 mean 0.985–0.989. The lower edge is not found: 2600 is the first T swept, and 2400–2600 is unmeasured apart from older reads below 1. Ours loses this band on k1: 1.68–1.72 ms against about 1.50 ms at T = 2700, while its k2 is faster (0.99 ms against activation 0.07 + k2 1.09 ms) [observed]. (Kernel-max round 1 read the landed k1 at 1.63–1.70 ms against vLLM's about 1.50 there; the band standing above, parity on uniform routing and 0.883–0.960 on zipf, supersedes this sweep for T ≤ 4096.)
- **Correctness across all T:** the grid kernels, every T from 1 to 4096 on 3 routings in 2 jobs, 24,576 evaluations: worst rel 3.988e-4, per-token 4.543e-4, none nonfinite, none over 1e-3; vLLM on 81 sampled cells per job 5.81e-4, 8.51e-4. Landed per-T builds were read on 17 more T (3.76e-4 against vLLM 5.38e-4); all T for those is inferred from the shared generator.
- **The grid's k1 gain is the value in the padding row [observed].** A zero padding row against real data is −3.0 to −4.0% k1 at T = 7168, while padding, X placement and copies change k1 within about 0.3%. Why is unknown [hypothesis]. A spare zero row with no copy reads 1.0016, 1.0004 whole-call at T = 7168 (`ship` 1.0224, 1.0170) and 1.0094, 0.9941 at T = 2700: parity, not a win, and it needs the caller to provide the row. Not landed.
- **Earlier results vs vLLM 0.30 `fused_experts`, same process on vLLM's stack, clang 22, landed per-compiler table** (ours over vLLM, **time, lower is better**; compiler output):

  | T | Result | Evidence |
  |---|---|---|
  | 16 | newest jobs 0.926–0.928, 0.884–0.896, 0.519–0.521; earlier 0.930, 0.844, 0.474 (uniform, zipf1, zipf2); real decode 0.920–0.944 | one synthetic job, one real job; same-tile harness 0.910–0.926, 0.915–0.929, 0.50 in 3 jobs |
  | 2–2049 | nothing over 1.0, worst 0.961–0.964; 1025–2048 0.827–0.942 with Square | 79-T scan on 3 routings, 4 jobs |
  | 2800–3072 uniform | 0.997–1.01; zipf1 0.92, real 0.94 | |
  | 4096, real | 0.943–0.944 | one job, override arm |
  | 6144–8192, real | 0.966–0.993 quiet (override arm); 0.967–1.044 by median of rounds, landed import, 2 noisy jobs | single paired rounds 0.938–1.053 |
  | 7168 uniform | 1.019–1.030, fastest call 1.012–1.022 | noisy jobs only |
  | ≥ 12288 | 1.00–1.02, fastest call 0.992–1.008; 16384 zipf1 0.983–0.997 | noisy jobs only; real routing untimed |

  - On clang 18 (ROCm 6.2.4), crossing processes and compilers: T up to 2049 no value over 1.0, real up to 4096 at most 0.972, real 8192 0.990–1.005, T = 6144–16384 0.96–1.04. vLLM cannot run there. Error at most 3.89e-4 synthetic, 6.73e-4 real, vLLM 6.33e-4, 7.65e-4.
- **What moved it:** a new down tile 64×64×128 (k2 1.616 to 1.284 ms), then two clang 22 levers: `-misched=gcn-max-ilp` on gate/up at T ≤ 512 and above 1024, which took T = 16 from 1.07–1.11× under 1.0, with load runs going from 1×20 to 20×1 although occupancy fell from 7 to 4; and a Square gate/up partition above T = 1024, which took 4.5–6.3% off k1 at T = 8192 in 3 jobs. Square does not carry to clang 18 (−0.9 to −2.2%, closed). Occupancy does not explain it: Square without the flag has higher occupancy and a slower k1.
- **Noise:** in two jobs an arm's round medians move up to 8.1%, and per-round fastest calls up to 7.5%; the job-wide fastest call flatters ours (median 3.9% above fastest against 1.6% for vLLM on real cells). Quote the median-of-rounds range.
- **vLLM stages at 4k tokens:** align 0.038, k1 2.078, `silu_and_mul` 0.095, k2 1.501, `moe_sum` 0.153 ms. vLLM's k1 still leads at T = 8192 on real data, 3.23 against 3.35 ms. It uses 32×32×8, ours 16×16×16; that this is the cause is a hypothesis.
- **Negative results:** larger tiles; pre-gathered rows; bM 128 (1.42–1.55× at 4k, 1.24–1.32× at T ≥ 6144 under Square, with k2 also worse); Square on down (k2 +4 to +12%); mid tile past 1024 (loses zipf2 and real T ≥ 1536); the flag on clang 18 (mixed); tiles do not carry between compilers.
- **Prototype, not landed:** one kernel set per power-of-two G serves T in (G/2, G]. T = 17 on 32: 0.941–0.965 against per-T 0.935–0.936; T = 4097 on 8192: 0.941–0.989 against 0.913–0.935. With the sum over T rows, T = 8193 on 16384 reads 0.977, 0.974 uniform and 0.983, 0.983 zipf1 against per-T 0.979, 0.975 and 0.975, 0.973; without it 1.015–1.022. That holds at G/2 + 1 only; the worst landed points were not timed on the grid.
- **An unexplained k1 gain:** the grid arms' k1 at T = 8193 uniform is 3.348, 3.315 ms against per-T 3.462, 3.427, −3.3% in every round, same tile and generator; vLLM 3.317, 3.295. k2 +2.0 to +2.2% and align +0.06 ms hide it in the total. So the 32×32×8 hypothesis above needs a qualifier: on uniform routing most of the k1 gap closes without it [inferred]; zipf1 keeps about 3.3%. Cause not established. The X path shipped was chosen on clang 18 and not retested on this tile on clang 22.
- **Decode-size weight bandwidth** (unique expert weight bytes over whole-call time, against 1.6 TB/s HBM): 53–77% synthetic and 62–77% real at T ≥ 4, thin at T = 4 (52.9–54.0%); 24–25% at T = 1, 32–47% at T = 2. vLLM reads 8.6–9% and 12–17% there. Whether the criterion should have a batch floor is an open decision.
- **Simulated FP8 and FP4 weights** (against our own fp16; no vendor FP kernel exists in the harness): time 1.70–3.22× and 1.68–6.32× at 7 sampled T; output relative error on real data 2.6–3.9% and 9.3–14.7%, cosine at least 0.9992 and 0.9903. An untuned prototype, and negative on speed.
- **Warp count closed:** `th` 512 and 128 on the Square tile raise k1 by 1.9–3.9% and 36–40%.
- **Open:** every new T costs 20–30 s of hipcc, excluded from all ratios, so none of this is a serving result. Landed kernels correct on sampled T only. Goal status at about T = 2600–2875 is open. Multi-GPU use is measured in §8.

## 6. GEMM
- **Overlap:** the direct global→LDS load runs on gfx90a. With overlap on, register staging beats it by 19–26% at one block per CU and ties at two. Our register prefetch was 3–4× slower because its staging arrays stayed in scratch memory, so it did not test register staging; with scalars, rr12864 reached 102.56 TF (one run), and rr12864w3 103.67–106.05 TF across jobs at 4096³. Double buffering helps only with bK 32.
- **Same process as hipBLASLt, direct C++ call, fixed-exponent data** (TF; hand-compiled C from TileLang output with occupancy and staging applied, not yet through TileLang):

  | Shape | hipBLASLt | base tile | best ours | ratio |
  |---|---|---|---|---|
  | 4096³ | 111.65 | 102.71 | 108.77 | 0.97× |
  | 8192³ | 118.06 | 83.11 | 106.37 | 0.90× |
  | 32768×1024×2048 | 102.76 | 107.07 | 108.72 | 1.06× |
  | 4096×17408×5120 | 117.52 | 89.03 | 109.65 | 0.93× |
  | 16384³ | 121.72 | 62.99 | 94.23 | 0.77× |

- That table used fixed-exponent data, which speeds hipBLASLt by 3.8–4.4% and these compiled kernels by 1.1–2.6%, so it understates them. On randn in one job the same kernels read 0.994, 0.93, 1.076, 0.97 and 0.826 (b25632w2). The earlier torch-versus-direct gap is under 0.03 ms on randn on both sides. Occupancy now goes through TileLang's JIT with no fork, by a compile hook or our patch 0001.
- **Placement, the current result** (hand asm; ours over hipBLASLt, throughput, higher is better; same process, randn; all 8 GCDs at the 500 W cap, 2 jobs on 2 nodes, 16 reads each, 0 of 256 reads below 1):

  | Shape | B at A − B = 128 mod 256 | same jobs, unshifted | through the script, placed |
  |---|---|---|---|
  | 16384³ | 1.047, 1.045 | 1.016, 1.016 (to 0.966) | 1.043, 1.044 |
  | 6144³ | 1.016, 1.023 | 0.998, 1.005 | 1.024, 1.022 |
  | 8192³ | 1.070, 1.068 | 1.036, 1.030 | 1.072 (1 job) |
  | 12288³ | 1.057, 1.057 | 1.042, 1.044 | 1.051 (1 job) |
  | 4096³ | 1.089, 1.078 | 1.055, 1.055 | not run |
  | 4096×17408×5120 | 1.057, 1.057 | 1.049, 1.045 | not run |
  | 32768×1024×2048 | 1.089, 1.093 | 1.059, 1.057 | not run |
  | 16384×4096×4096 | 1.056, 1.060 | 1.035, 1.038 | not run |

  - **Mechanism:** the low mode depends on the A-B address relationship, including higher bits, and exactly 128 mod 256 lifts it at every read so far; other offsets with bit 7 set do not. Default allocators return 2 MiB-aligned addresses, so unplaced runs sit in the unfavourable phase [inferred]. hipBLASLt is flat. The hardware reason is open.
  - **Cost:** a copy of B at 16384³ is 0.92 ms. Copying on every call loses at small M (net 0.986× at 6144³, 1.029× at 4096³, partly extrapolated), so only a long-lived B keeps these numbers.
  - **Unplaced tensors:** 0.99–1.10× by shape and condition. At the cap 16384³ 1.015× [0.978–1.022], 6144³ 1.003× [0.996–1.008]; solo script 0.999–1.000× and 0.988–0.990× at those two, after the rule moved to the cap-first tile. 3072³ reads 0.967–0.982× unplaced and 0.996–1.004× placed, since hipBLASLt peaks there.
  - **Floors:** below 200 blocks or short side 1024 the rule returns to hipBLASLt, after real losses (1024³ 0.56–0.60×, 2048³ 0.79–0.83×). Output is correct at all these shapes. On a placed B at low K, 128×256 reads under unplaced 256×128.
  - **Compilers:** placement does not make a compiler result. Post-processed rr25632w2 at 8192³ reads 0.972–1.023× over 4 jobs on one GCD; pure TileLang 128×256×32 at 16384³ reads 0.774–0.818×, worse than unplaced. A pass (patch 0009) recognises the plain kernel and emits the hand asm verbatim, medians 1.01–1.10×; that is hand asm from a pass, not a compiler schedule. Placement does not transfer to prefill or MoE.
- **Unplaced record, hand-written schedule, launched through TileLang** (same convention; inline asm, not generated by TileLang; correct to 3.5e-4–6.9e-4 relative to hipBLASLt). Later rule changes moved 12288³ and 4096×17408×5120 solo to 1.031–1.045×:

  | Shape | through `gemm_prefetch.py auto`, 2 jobs, one GCD | C++ harness, all reads | all 8 GCDs at the power cap, 2–3 jobs, median |
  |---|---|---|---|
  | 4096³ | 1.063, 1.065 | 1.050–1.073 | not run |
  | 8192³ | 1.075, 1.072 | 1.029–1.074 | 1.035–1.043 |
  | 12288³ | 1.059, 1.068 | 1.004–1.069 | not run |
  | 32768×1024×2048 | 1.107, 1.102 | 1.078–1.105 | not run |
  | 4096×17408×5120 | 1.068, 1.067 | 1.012–1.066 | not run |
  | 5632×3200×4096 | 1.077, 1.077 | 1.050–1.073 | not run |
  | 16384×4096×4096 | 1.067, 1.066 | 1.030–1.064 | not run |
  | 6144³ | 1.016, 1.021 | 0.996–1.020 | 1.006–1.008 (128×256) |
  | 16384³ | 1.041, 1.038 | 0.976–1.035 | 1.014 (128×256), 0.969–1.024 |

  - The rule then picked 128×256 at 6144³ and 16384³, which read 0.995 and 1.008 in one wiring job; unplaced, 6144³ is a tie and 16384³ is thin, with low-mode reads near 0.97×. Overall this table is "1.00–1.11× by condition; 1.06–1.11× on a quiet node".
  - **What moved it:** LDS row stride (128 B 107 TF, 144 B 117, 160 B 120), memory operations placed between MFMAs (+13.8%, two jobs), block swizzle on some shapes. Mechanism untraced; LDS banking is a hypothesis.
  - **Clock:** hipBLASLt runs at 1028–1145 MHz against our 1094–1251 MHz and wins on efficiency, about 91–93% against 85% at 8192³. (Superseded 9 Nov: with the 256×256 tile ours matches hipBLASLt per clock at 12288³ and reaches 90.2–90.5% at 8192³.) No lower-draw win was found. That clock is what limits MFMA-only throughput is shown for that micro only.
  - **Compiled kernels, through the JIT (C1):** still mostly behind hipBLASLt: on fixed-exponent data 0.97× at 4096³, about 0.90× at 8192³, 0.94× at 17408, 1.06× at 32768, about 0.79× at 16384³. A gap between global loads gives b12864 +12.9% at 8192³ through one LLVM flag, and still loses to staging. Contiguous allocation and xnack- do not help (hand ratios fall about 4%; one +1.2% read not repeated). Compiler support for the schedule is open.
- **Earlier best ratios through torch, base tiles:** 0.94× at 4096³; 1.06× on the MoE expert shape. hipBLASLt at 16384³ uses `MT128x256x64 MI32x32 MIWT2_4 MO40`, group_seg 55296.
- **Final rounds, landed rule** (hand asm, same convention). A new 256×192 tile at the cap through the real script, 8 GCDs, median of 16 reads: placed 3072³ 1.016, 6144³ 1.063, 12288³ 1.075; unplaced 1.005 [0.997–1.018], 1.054, 1.069. Solo, 2 jobs on 2 nodes: 1536³ 1.03–1.04, 2048³ 1.04–1.07, 2304³ 1.12–1.14, 4608³ 1.09–1.11, 6144³ 1.06–1.09, 12288³ 1.08–1.11, 1280×8448×2048 1.19–1.20, where the rule previously used hipBLASLt at the three smallest. A rule of factor × fill^0.7, fill = B / (110 ceil(B/110)), with factors fitted to about 30 shapes, picked the tile on 6 shapes tested; the cause is a hypothesis. Still below 1 in 2 jobs: 7168×3584×4096 0.951–0.987, 2816³ 0.968–0.998, 1024×7680×2048 0.979–0.997; 16384³ unplaced 0.993 and 1.028 by job. (Superseded 9 Nov: all three now lead, see the kernel-max block at the end of this section.) These supersede the 6144³ and 12288³ cells above.
- **Small M.** Earlier split-K reads of 0.80–1.08× do not stand. Compiled alone at 16×17408×5120 reads 0.786–0.798×, under 0.8. Zero-padding N to 17440 lets hand asm use hipBLASLt's own 109-block tile. Reads on one allocation, 2 jobs: 64×17408×5120 1.023–1.037×, 32×17408×5120 1.006–1.016×, 16×17408×5120 0.988–0.991×; own B 64×5120×5120 0.979–0.982×, 32×5120×5120 1.034–1.037×. Allocations matter: one B allocation read 5.7–8.4% faster across all arms using it in one job, and not on the padded allocation, so ratios across allocations mislead. These entries live in an explicit small-M path, not the automatic rule, and were never read at A − B = 128 mod 256.
  - **Reread on one allocation, not landed (2 jobs, shared nodes, randn, seeds repeat):** landed 16× 0.992, 0.994 paired and 64×5120 0.995, 0.992. A buffer resource limited to the true size of A and B (`rng`) needs no padded copy: 16× 1.015, 1.015. 32x96 at 64×5120×5120: `rng` N 5184 1.137, 1.141; zero-padded N 5280 1.124, 1.131; tied when compared directly; +14–15% over the landed tile. In the allocation's fast mode, on one job and 2 draws, `rng` reads 1.018 against hipBLASLt on the same B, while landed reads 0.933 on that B. Output matches hipBLASLt's error to all printed decimals. No production path reaches these forms.

- **Kernel-max rounds, 9 Nov** (`evidence/GEMM-MAX/RESULTS.md`, rounds 1–6; ours over hipBLASLt (hb) through torch, throughput, higher is better, same process, arm order rotated, true 2MNK; one GCD on shared small-g nodes, unplaced B; module CLI on the final bytes in two opposite-order jobs). **Hand asm launched through TileLang, not a compiler result.** Landed `gemm_prefetch.py` md5 04eece68 (was ecb1cb9d; round 6 changed only the output layout, from 2a1dfb8f). Every shape now returns a contiguous (M, N) tensor; the rows below that changed in round 6 are the final module file's reads (jobs 23579805 fwd, 23579806 rev, and 23579942 fwd at the three shapes disturbed in 23579805; 7 rounds each). Every arm's error against fp64 equals hb's to 3 digits (2.02e-4–2.09e-4). This block supersedes "Still below 1" above and the small-M reads of §6 for these shapes.

  | Shape | Tile | Ours / hb |
  |---|---|---|
  | 32768×1024×2048 | 256×256 | 1.143–1.163 |
  | 4096×17408×5120 | 256×256 | 1.126–1.140 |
  | 12288³ | 256×256 | 1.118–1.133 |
  | 8192³ | 256×224, contiguous output | 1.129–1.137 |
  | 16384³ | 256×256 | 1.094–1.112 |
  | 16384×4096×4096 | 224×256 | 1.093–1.108 |
  | 6144³ | 256×192, 8 waves | 1.085–1.101 |
  | 5632×3200×4096 | 256×224, contiguous output | 1.093–1.103 |
  | 4096³ | 224×256 | 1.084–1.096 |
  | 2816³ | 128×192, 8 waves, contiguous output | 1.086–1.091 |
  | 7168×3584×4096 | 256×192, 8 waves, contiguous output | 1.077–1.085 |
  | 2048³ | 192×256, 8 waves | 1.162–1.189 |
  | 3072³ | 192×256, 8 waves | 1.037–1.054 |
  | 1024×7680×2048 | 128×192, partial LDS waits | 1.109–1.122 |
  | 64×5120×5120 | small M | 1.142–1.155 |
  | 64×5120×17408 | small M | 1.123–1.148 |
  | 32×5120×5120 | small M | 1.075–1.096 |
  | 16, 32, 64 × 17408 × 5120 | small M | 1.007–1.064 |
  | 1×5120×5120 | small M (was hb) | 1.012–1.052 |

  - **What moved it.** (1) Full-fill tiles: M and N round up to the tile and the buffer resources end at the true sizes, so edge rows load zeros with no copy; this flipped the three shapes that were below hb (2816³, 7168×3584×4096, 1024×7680×2048). (2) Range-checked and PGR2-style small-M tiles, and partial LDS waits at 1024×7680×2048. (3) 8 waves on a range-checked 256×192 tile (+2.0 to +11.7% at 7 large shapes). (4) An XOR swizzle of 16-byte LDS chunks (chunk c of row r at c ^ (r & 7)) that frees the row padding, so a 4-wave 256×256 tile fits 64 KiB (204 VGPR + 256 AGPR, occupancy 1, no spills): +1.1 to +7.2% over round 3 solo, +3.2 to +8.8% per GCD at the 500 W cap with 8 GCDs racing. (5) Contiguous output everywhere: where N rounds up to the tile, C is allocated with the true N and stores past it are skipped (5632×3200×4096 in round 5; 8192³, 7168×3584×4096, 2816³, the small-M rows forms and the padded-B 32/64×17408 in round 6). Tiles and schedules are unchanged and outputs are bitwise equal to the earlier views. Against those non-contiguous views it costs 0.3–0.5% at 7168×3584×4096, 0.5–0.7% at 8192³, 0.2–0.3% at 5632×3200, and is at noise at the median at 2816³ (one round 0.15% down, beyond that job's 0.11% noise). At the other small-M shapes it is at noise at the median (at 64×17408, per-round minima 0.9985 and 0.9975 dip beyond noise of 0.07% and 0.10% in both jobs, medians 0.9996), and it gains 0.4–0.6% at 64×5120×5120. Contiguous output is a decision, not a measured win: only 64×5120×5120 meets the win rule (new over old above 1 + noise in every round of both jobs), and the costs at 8192³ and 7168×3584×4096 are beyond noise. It beats the view plus `.contiguous()` by 2.2–12.6% and an exact-tile form by 2.5–4.7%.
  - **Against the ceiling [computed].** Clock is `average_gfxclk` of our own GCD from `gpu_metrics` (a measured average, not the `current_gfxclk` target); the package includes the twin GCD, 19–100% active in other jobs. 7 of 9 established large shapes reach 85.1–94.6% of the clock-scaled MFMA peak (191.5 TF × clk / 1700) at 1182–1248 MHz; 12288³ 128.2–129.1 TF = 94.5–94.6%, equal to hb's 94.6–94.7% per clock while running about 140 MHz faster. Best solo read 129.1 TF = 67.4% of the nominal 191.5 TF; at the cap 12288³ reaches 132.9 TF = 69.4% of nominal (95.8–96.1% clock-scaled, hb 94.3–95.5%). hb still leads per clock at 16384³, 6144³, 5632×3200, 4096³ and 2048³ (2048³ ours 67.2%, hb 74.1%) but runs 110–350 MHz lower. Small M is bandwidth-bound: 45–59% of the nominal 1.6 TB/s, 57–79% of the measured 1.19–1.28 TB/s single-GCD streaming ceiling. Per-card clock evidence stays on LUMI (it carries hardware identifiers).
  - **Negative.** 32×32×8 MFMA in the hand loop (0.84–1.00 of hb, below the landed loop at all 14 shapes); compiled TileLang forms 0.66–0.96 of hb in round 3; knobs on the 256×256 tile; xs at small M. The direct global-to-LDS load remains untried only inside the current hand-asm loop; K1's standalone result (register staging ahead by 19–26% at one block per CU) stands.
  - **Caveats.** Ratios within about 2% are not separable from a same-process identical-kernel spread of up to 2.4% seen in round 1 at 8192³; placed B and the cap were not read for most new entries (one cap job, without 3072³). An earlier job with the identical kernels read 8192³ 1.088–1.108 (hb 111 TF against 107–108 here) and the padded-B ratio at 64×17408×5120 is 1.007–1.008 in one job and 1.060–1.064 in the other, so hb's and the padded B's per-process state moves these rows by a few percent [inferred]. GPU use 5.06 GCD-hours over the six rounds.
## 7. Pitfalls on gfx90a with TileLang 0.1.15
1. `__builtin_amdgcn_wavefrontsize` and the OCP FP8 types are missing in ROCm 6.2 headers; shim them.
2. fp16 reference drift.
3. Direct global→LDS bad offsets are silent: dropped or wrapped, EXEC and TRAPSTS unchanged. A VALU-written SGPR read by the load needs 4+ wait states (ISA asks for 5), and the template is exposed.
4. FullCol / 128–256-thread layouts fail on 16-row tiles.
5. Loop extents read from shared memory are rejected.
6. `T.Pipelined(num_stages > 1)` is stripped to a sequential loop on HIP, except gfx950. Explicit `order`/`stage` are not stripped (compile only).
7. 10 upstream unit tests fail only because int8/FP8 instructions are missing on gfx90a.
8. `compile_flags` has no effect on the HIP path in 0.1.15; `HIPCC_COMPILE_FLAGS_APPEND` works, but the kernel cache cannot see it or the post-process hook, so arms silently share a binary without their own key.
9. Occupancy decides tiles and is invisible at source level: 256×128×32 at fp16 runs Occupancy 1, 52–54 TF, and one attribute lifts it above 100 TF at 8192³. Spills near the register cap move TF by −9% to +39% on small source edits.
10. Fixed-exponent test data speeds hipBLASLt more than compiled kernels, by 3.8–4.4% against 1.1–2.6%. Benchmark on random data.
11. Compilers flip results: MoE tiles and load grouping do not carry between clang 18 and clang 22. Scheduler flags with a wrong spelling are silently ignored on clang 22.
12. A register promotion that is safe on clang 18 can kill clang 22: promoting a query fragment read as an MFMA operand left the K loop not unrolled, every read a chain of selects, 1198 to 4290 instructions, decode 8.6–10.3× slower with identical output. Race every patch across workloads and compilers.
13. Operand placement moves results: GEMM's A − B address relationship changes standing by several percent, and default allocators always land in one phase.
14. A cast can hide a branch: `hip_bfloat16` costs one exec-mask branch per element (about 2.4% on decode).
15. A lab baseline can be the wrong kernel: our Triton call omitted `seq_threshold_3D`, so it raced a 2D path serving does not use at these batches. Only running the server exposed that; racing the 3D kernel took 1.10–5.33× down to about 0.87–1.37×. A Python hook is also silently inert under vLLM's FULL-graph mode.
16. Comparing across allocations can invent a vendor mode: an allocation's speed moved all its arms by 5.7–8.4%.
17. Benchmarks can grade killed answers: a kill-time `finish=abort` stream is recorded as an answer, and a restart-from-zero timeout flatters the arm that loops. Seen in our Scores harness (§9).
18. A vendor-library clock counter can report a target: amdsmi `current_gfxclk` read exactly 1650.0 in 1812 of 2550 prefill readings, while a clock64 probe read about 935 MHz in decode and a GRBM estimate about 1.1 GHz in prefill. Clock-scaled shares built on it are unreliable.
19. Two opposite-order jobs do not separate gains of about 1% from noise: identical code in two graphs or two code objects moved by up to 1.75–2.4% in one job. Race an identical-code control arm and require the gain to beat its spread (decode and prefill kernel-max reviews reverted 6 decode entries and 5 prefill lengths on this rule).
20. Never replace a module file while a job compiles from it: TileLang re-reads source at compile time, and one landing race lost every opposite-order pair that way.

## 8. Scaling
No vendor comparison is made here; nothing in this section changes a kernel ratio.

- **RCCL on 1/2/4 nodes (torch.distributed, fp16, busbw):** all-reduce 1 GB keeps 134 → 98 → 93 GB/s; all-to-all 1 GB falls
  67 → 18.5 → 12.1 GB/s once traffic leaves the node.
- **Expert-parallel MoE layer, landed kernels** (256 experts in total, top-8, weak scaling at 2,048 tokens per rank, clang 18, synthetic routing, 1.3× padding added by `ep_moe.py`; job 23516765, other jobs in the ranges). Efficiency is tok/s at W over W × tok/s at W = 1, same arm, same job. Share is (dispatch + combine) over the three phases.

  | GCDs (nodes) | 1 (1) | 8 (1) | 16 (2) | 32 (4) |
  |---|---|---|---|---|
  | old kernels, tok/s total | 0.341 M | 2.706 M | 3.053 M | 4.114 M |
  | old, efficiency | – | 99.0–99.2% | 55.1–56.0% | 37.7–38.0% |
  | landed, tok/s total | 0.527 M | 3.219 M | 3.342 M | 4.345 M |
  | landed, efficiency | – | 76.4–76.7% | 39.4–39.7% | 25.8–26.6% |
  | landed, dispatch + combine share | 28.3% | 65.2% | 82.6% | 88.6% |

- **Reading.** "99% inside a node" does not hold for landed kernels. Throughput rises 19% at 8 GCDs while efficiency falls, because W = 1 is faster and dispatch and combine are unchanged. Neither number is pure communication: with W = 1 compute held at its W = 8 value, efficiency would read about 74% for old and 64% for landed [inferred]. 8 to 32 GCDs gives 1.35× throughput for 4× the GCDs. At 32 GCDs compute is under 12% of the step. `land` is landed kernel code in a non-landed call form: one row per routing pair, no landed sum. An earlier share of 87% at 32 GCDs on old kernels is mostly a formula difference: 87.2% over the logged step, 84.6% over the phase sum, 83.9% here.
- **Conclusion:** keep expert parallelism inside a node on LUMI-G. No skewed routing, real model or opposite-order full ladder was run; clang 22 not tested.
- **vLLM 0.30, TP=4, across two nodes** (public 27B, BF16, ROCm 7.2.3 image, outside LUMI's supported range, 128 output tokens, no speculative decoding). Two nodes over one, higher is better:

  | Measure | Image's TCP sockets, across jobs | Hand-written OFI adapter, same job against a cold one-node arm | Hand-written OFI adapter, across jobs |
  |---|---|---|---|
  | Batch 1 | 0.56–0.65 | 0.83 | 0.75–0.78 |
  | Batch 32 | 0.36–0.40 | 0.83 | 0.67–0.79 |
  | Prefill 7,211 and 28,812 tokens | 0.19–0.23 | 0.86–0.87 | 0.65–0.81 |

  - **Transport.** The image's RCCL falls back to sockets across Slingshot, 7.0–7.8 GB/s on all-to-all 256 MB against 190 GB/s in a node. LUMI's aws-ofi-rccl 1.4.0 exports only plugin API v5, so as shipped it does not load. A hand-written v6 adapter plus a `libamdhip64` shim runs it, 61.8–124.4 GB/s; **not a compiler or vendor result**, with an ABI risk not excluded. OFI over sockets on prefill: about 3–4× across jobs, one same-job pair at 4.3–4.6× on a low socket read. Every OFI prefill ran at NCCL INFO; OFI prefill at WARN is unmeasured.
  - **It hangs.** 12 two-node attempts: sockets 8 (2 finished, 5 hung, 1 died at startup), OFI 4 (3 finished, 1 hung); one-node arms 2 of 2 finished. Per burst of 32 requests, socket 5 of 12, OFI 1 of 10, p about 0.16, so the transports cannot be told apart. The stall sits in the first forward pass of a burst [inferred]. Mixed compile state, sockets, logging level and allocation shape are not necessary conditions. Cause open. sacct reports COMPLETED on such jobs.
  - **Output.** Batch 1 text hash matches across arms and jobs, except one cold-compile arm; needle, tool and thinking checks pass. Batch 8, 32 and prefill carry no usable equality check.
  - **Trap.** The compile cache keys on rank but stores device ordinals, so a cache from one layout breaks on another.
  - **Replicas** remain the likely route for serving [hypothesis, not measured].


## 9. Reproducibility
Every number comes from a logged run, with job IDs, arm order and raw JSON. The code, harness and results log are published with this paper (`amd-mi250x/` in the horizon-kernels repository). TileLang compiler fixes are prepared as a patch series that applies to upstream main 6e8f599e, with gates passed on the final tree: no fuzz, 47 new tests on clang 18 and clang 22, the same 27 regression failures as clean main on clang 18, and CUDA and Ascend builds. No pull request is open yet; CUDA was checked at compile level only.

## Acknowledgement
We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LUMI, hosted
by CSC (Finland) and the LUMI consortium, through a EuroHPC Development Access call (EHPC-DEV-2026D11-419).
