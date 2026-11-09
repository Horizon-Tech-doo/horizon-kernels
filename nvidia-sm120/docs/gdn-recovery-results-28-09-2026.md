# GDN recovery standalone evidence

28-09-2026. Experimental; full vLLM integration gate remains pending.

| Requests | vLLM μs/layer | G1 μs/layer | G2 verify + commit μs/layer | Reduction vs vLLM |
|---:|---:|---:|---:|---:|
| 1 | 13.42 | 11.28 | 9.28 | 30.8% |
| 8 | 91.65 | 84.33 | 41.47 | 54.8% |
| 32 | 370.9 | 369.7 | 229.62 | 38.1% |

Raw data: `results/gdn_recovery_28-09-2026/gdn_g2.jsonl`.

Three chained cases passed against G1: 300 steps at two requests/two layers, 100 steps at eight requests/two layers, and 50 steps at three requests/one layer with vr=128. Four-token windows and randomized acceptance lengths 1–4. Outputs compared every step; recurrent states checked every 25 steps and the final step. Equality ignores signed-zero differences. This is not full-model quality evidence.

The standalone commit uses zero boundary indices/lengths. It does not validate the integration’s convolution compaction, aligned-prefix boundary handling, mixed prefill/decode scheduling or real runner sampling lifecycle. G2 total sums separately measured verify and commit medians; these are not end-to-end latency or throughput gains.
