# K4 segment results — 28-09-2026

Correctness gate passed; no persistent-segment speed win in the tested buckets. GPU run completed in 244.9 seconds. This is a partial model segment, not the full megakernel.

| Token rows | Phase-1 baseline (us/layer) | Phase-1 policy | Best persistent segment | Slower than best phase-1 |
|---:|---:|---:|---:|---:|
| 4 | 165.27 | 161.02 | 166.53 | 3.42% |
| 8 | 169.40 | 163.74 | 171.92 | 5.00% |
| 16 | 170.66 | 170.01 | 180.88 | 6.39% |
| 32 | 193.18 | 194.44 | 217.99 | 12.84% |

The test rotates four layers beyond L2 capacity, uses CUDA graphs and 150 interleaved iterations per bucket. Median totals above are descriptive; paired timing distributions are in the raw JSONL.

Coverage: 19 main correctness rows across five variants; six-stage run-ahead at M32 is excluded by the shared-memory co-residency constraint. The selected run-ahead variant has complete main correctness, attention-next-layer correctness, timing and HKT records at all four sizes. Timeout reporting, host-visible guard, raised step check and bit-identical recovery pass. Local seg_gate_v2 independently accepts the captured results.

HKT run-ahead task-busy fractions are 0.648/0.653/0.523/0.520 at M4/8/16/32. These are instrumented CTA-time fractions, not hardware GPU utilization. Gate/up GEMM median task runtime is 63–67 us, down projection 31–37 us and next-layer QKV 20–31 us. This identifies expensive stages, but does not prove waiting alone explains the gap. Increased prefetch and six stages did not produce a winning tested segment.

Evidence: results/k4_segment_28-09-2026/seg4.jsonl and seggate.txt. The earlier partial snapshot remains historical evidence only.
