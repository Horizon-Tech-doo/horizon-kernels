
## Flags `d` (ops arm po_d, job 23621041)

**Cross-check [observed]: client time to first token (s) or decode step (ms), and the server-side step time summed over the prefill's steps (ms) or per decode step, worker 0**

| Cell | prompt tokens | steps (tokens) | s_d client | ps_d client | py_d client | pyn_d client | po_d client | ps_d server | py_d server | pyn_d server | po_d server | ps period, ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pf115 | [1413] | [784, 629] | 0.4547 | 0.4572 | 0.4646 | 0.4645 | 0.6869 | 262.7 | 452.9 | 453.0 | 679.0 | - |
| pf330 | [3993] | [1568, 1568, 784, 73] | 1.2405 | 1.2479 | 1.2635 | 1.3196 | 1.6873 | 465.8 | 1243.0 | 1299.8 | 1674.5 | - |
| pf640 | [7713] | [1568, 1568, 1568, 1568, 784, 657] | 2.3688 | 2.3798 | 2.4038 | 2.4015 | 3.0142 | 790.1 | 2371.2 | 2370.7 | 2994.7 | - |
| pf1340 | [16115] | [1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 435] | 5.3387 | 5.3585 | 5.4069 | 5.4013 | 6.4872 | 1392.4 | 5351.4 | 5346.6 | 6447.6 | - |
| dc1 | [7709] | 94 steps | 55.78 | 56.336 | 58.883 | 70.813 | 135.237 | 5.9 | 58.7 | 70.6 | 135.0 | 56.42 |
| dc8 | [7709] | 59 steps | 68.507 | 69.991 | 72.426 | 93.667 | 167.828 | 6.4 | 71.9 | 93.2 | 166.9 | 69.799 |
| dc32 | [7711, 7713] | - | -999159.363 | -998691.605 | -1012718.155 | -1010265.312 | -1215915.779 | - | - | - | - | - |

### `d` pf115: prompt [1413] tokens, per prefill, all its steps, steps [784, 629]

Ops arm 679.0 ms, timed parts sum 678.3 ms, same flags with one synchronise per step 453.0 ms, 2476 synchronises, c = 91 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 256 | 241.6 | 35.6% | 218.2 | 48.3% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 192 | 78.8 | 11.6% | 61.3 | 13.5% |
| attention projection GEMM (qkv_proj, o_proj) | 160 | 68.3 | 10.1% | 53.7 | 11.9% |
| attention kernel | 32 | 19.0 | 2.8% | 16.1 | 3.6% |
| KV-cache write | 32 | 4.6 | 0.7% | 1.7 | 0.4% |
| GDN core (conv + delta rule) | 96 | 119.8 | 17.7% | 76.0 | 16.8% |
| tensor-parallel all-reduce | 258 | 47.9 | 7.1% | 24.4 | 5.4% |
| sampling incl. lm_head GEMM and all-gather | 2 | 3.9 | 0.6% | 3.0 | 0.7% |
| runner: input prep, request bookkeeping, Python | 14 | 3.5 | 0.5% | 2.2 | 0.5% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 1026 | 84.0 | 12.4% | -9.7 | -2.1% |
| runner glue | 17 | 6.8 | 1.0% | 5.4 | 1.2% |
| sum | | 678.3 | | 452.3 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 784 x 17408 x 5120 | gate_up_proj | 64 | 1.2892 | 82.5 | 108.4 | 76.7 |
| linear 629 x 17408 x 5120 | gate_up_proj | 64 | 1.1224 | 71.8 | 99.9 | 66.0 |
| linear 784 x 5120 x 8704 | down_proj | 64 | 0.7695 | 49.2 | 90.8 | 43.4 |
| linear 629 x 5120 x 8704 | down_proj | 64 | 0.5938 | 38.0 | 94.4 | 32.2 |
| linear 784 x 8192 x 5120 | in_proj_qkvz | 48 | 0.7450 | 35.8 | 88.3 | 31.4 |
| linear 629 x 8192 x 5120 | in_proj_qkvz | 48 | 0.6213 | 29.8 | 84.9 | 25.4 |
| linear 784 x 5120 x 3072 | o_proj/out_proj | 64 | 0.4040 | 25.9 | 61.0 | 20.0 |
| linear 629 x 5120 x 3072 | o_proj/out_proj | 64 | 0.3352 | 21.5 | 59.0 | 15.6 |
| linear 784 x 7168 x 5120 | qkv_proj | 16 | 0.7143 | 11.4 | 80.6 | 10.0 |
| linear 629 x 7168 x 5120 | qkv_proj | 16 | 0.5972 | 9.6 | 77.3 | 8.1 |
| linear 629 x 48 x 5120 | in_proj_ba | 48 | 0.1382 | 6.6 | 2.2 | 2.3 |
| linear 784 x 48 x 5120 | in_proj_ba | 48 | 0.1373 | 6.6 | 2.8 | 2.2 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 96, 66.4; causal_conv1d_fn 96, 15.4; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 2, 2.4; all_gather 2, 0.4.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 14.7; gate_up_proj>down_proj 11.7; in_proj_ba>gdn 10.6; comm>gate_up_proj 10.1; comm>in_proj_qkvz 9.5; run>comm 5.4; qkv_proj>kvw 5.3; o_proj/out_proj>comm 4.9; down_proj>comm 4.6; in_proj_qkvz>in_proj_ba 4.3; attn>o_proj/out_proj 3.8; comm>qkv_proj 3.2; kvw>attn 1.3; run>run 0.9.

### `d` pf330: prompt [3993] tokens, per prefill, all its steps, steps [1568, 1568, 784, 73]

Ops arm 1674.5 ms, timed parts sum 1672.4 ms, same flags with one synchronise per step 1299.8 ms, 4952 synchronises, c = 76 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 512 | 632.4 | 37.8% | 593.6 | 45.7% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 384 | 200.9 | 12.0% | 171.9 | 13.2% |
| attention projection GEMM (qkv_proj, o_proj) | 320 | 162.8 | 9.7% | 138.6 | 10.7% |
| attention kernel | 64 | 92.9 | 5.6% | 88.1 | 6.8% |
| KV-cache write | 64 | 9.7 | 0.6% | 4.9 | 0.4% |
| GDN core (conv + delta rule) | 192 | 254.4 | 15.2% | 181.7 | 14.0% |
| tensor-parallel all-reduce | 516 | 109.3 | 6.5% | 70.3 | 5.4% |
| sampling incl. lm_head GEMM and all-gather | 4 | 7.8 | 0.5% | 6.3 | 0.5% |
| runner: input prep, request bookkeeping, Python | 28 | 6.8 | 0.4% | 4.7 | 0.4% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 2052 | 182.0 | 10.9% | 26.7 | 2.1% |
| runner glue | 33 | 13.5 | 0.8% | 11.1 | 0.9% |
| sum | | 1672.4 | | 1297.7 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 1568 x 17408 x 5120 | gate_up_proj | 128 | 2.4135 | 308.9 | 115.8 | 299.2 |
| linear 1568 x 5120 x 8704 | down_proj | 128 | 1.2212 | 156.3 | 114.4 | 146.6 |
| linear 1568 x 8192 x 5120 | in_proj_qkvz | 96 | 1.3362 | 128.3 | 98.4 | 121.0 |
| linear 784 x 17408 x 5120 | gate_up_proj | 64 | 1.2906 | 82.6 | 108.3 | 77.8 |
| linear 1568 x 5120 x 3072 | o_proj/out_proj | 128 | 0.5927 | 75.9 | 83.2 | 66.2 |
| linear 784 x 5120 x 8704 | down_proj | 64 | 0.7711 | 49.3 | 90.6 | 44.5 |
| linear 784 x 8192 x 5120 | in_proj_qkvz | 48 | 0.7464 | 35.8 | 88.1 | 32.2 |
| linear 1568 x 7168 x 5120 | qkv_proj | 32 | 1.0951 | 35.0 | 105.1 | 32.6 |
| linear 784 x 5120 x 3072 | o_proj/out_proj | 64 | 0.4052 | 25.9 | 60.9 | 21.1 |
| linear 73 x 17408 x 5120 | gate_up_proj | 64 | 0.3222 | 20.6 | 40.4 | 15.8 |
| linear 73 x 5120 x 8704 | down_proj | 64 | 0.2275 | 14.6 | 28.6 | 9.7 |
| linear 1568 x 48 x 5120 | in_proj_ba | 96 | 0.1444 | 13.9 | 5.3 | 6.6 |
| linear 784 x 7168 x 5120 | qkv_proj | 16 | 0.7154 | 11.4 | 80.4 | 10.2 |
| linear 73 x 5120 x 3072 | o_proj/out_proj | 64 | 0.1737 | 11.1 | 13.2 | 6.3 |
| linear 73 x 8192 x 5120 | in_proj_qkvz | 48 | 0.2154 | 10.3 | 28.4 | 6.7 |
| linear 784 x 48 x 5120 | in_proj_ba | 48 | 0.1377 | 6.6 | 2.8 | 3.0 |
| linear 73 x 48 x 5120 | in_proj_ba | 48 | 0.1257 | 6.0 | 0.3 | 2.4 |
| linear 73 x 7168 x 5120 | qkv_proj | 16 | 0.2115 | 3.4 | 25.3 | 2.2 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 192, 144.7; causal_conv1d_fn 192, 33.8; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 4, 4.8; all_gather 4, 0.7.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 29.1; gate_up_proj>down_proj 28.1; comm>in_proj_qkvz 24.0; comm>gate_up_proj 22.9; in_proj_ba>gdn 21.2; qkv_proj>kvw 10.7; run>comm 10.5; o_proj/out_proj>comm 9.7; down_proj>comm 9.2; in_proj_qkvz>in_proj_ba 8.7; comm>qkv_proj 8.1; attn>o_proj/out_proj 7.6; kvw>attn 2.6; run>run 1.8.

### `d` pf640: prompt [7713] tokens, per prefill, all its steps, steps [1568, 1568, 1568, 1568, 784, 657]

Ops arm 2994.7 ms, timed parts sum 2991.8 ms, same flags with one synchronise per step 2370.7 ms, 7428 synchronises, c = 84 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 768 | 1176.6 | 39.3% | 1112.1 | 47.0% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 576 | 366.9 | 12.3% | 318.5 | 13.5% |
| attention projection GEMM (qkv_proj, o_proj) | 480 | 291.0 | 9.7% | 250.7 | 10.6% |
| attention kernel | 96 | 236.5 | 7.9% | 228.5 | 9.6% |
| KV-cache write | 96 | 14.8 | 0.5% | 6.7 | 0.3% |
| GDN core (conv + delta rule) | 288 | 395.6 | 13.2% | 274.6 | 11.6% |
| tensor-parallel all-reduce | 774 | 186.0 | 6.2% | 121.0 | 5.1% |
| sampling incl. lm_head GEMM and all-gather | 6 | 11.7 | 0.4% | 9.2 | 0.4% |
| runner: input prep, request bookkeeping, Python | 42 | 10.3 | 0.3% | 6.8 | 0.3% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 3078 | 281.6 | 9.4% | 23.0 | 1.0% |
| runner glue | 49 | 20.7 | 0.7% | 16.7 | 0.7% |
| sum | | 2991.8 | | 2367.8 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 1568 x 17408 x 5120 | gate_up_proj | 256 | 2.4141 | 618.0 | 115.8 | 596.5 |
| linear 1568 x 5120 x 8704 | down_proj | 256 | 1.2237 | 313.3 | 114.2 | 291.8 |
| linear 1568 x 8192 x 5120 | in_proj_qkvz | 192 | 1.3377 | 256.8 | 98.3 | 240.7 |
| linear 1568 x 5120 x 3072 | o_proj/out_proj | 256 | 0.5926 | 151.7 | 83.2 | 130.2 |
| linear 784 x 17408 x 5120 | gate_up_proj | 64 | 1.2902 | 82.6 | 108.3 | 77.2 |
| linear 657 x 17408 x 5120 | gate_up_proj | 64 | 1.1249 | 72.0 | 104.1 | 66.6 |
| linear 1568 x 7168 x 5120 | qkv_proj | 64 | 1.0960 | 70.1 | 105.0 | 64.8 |
| linear 784 x 5120 x 8704 | down_proj | 64 | 0.7694 | 49.2 | 90.8 | 43.9 |
| linear 657 x 5120 x 8704 | down_proj | 64 | 0.6491 | 41.5 | 90.2 | 36.2 |
| linear 784 x 8192 x 5120 | in_proj_qkvz | 48 | 0.7465 | 35.8 | 88.1 | 31.8 |
| linear 657 x 8192 x 5120 | in_proj_qkvz | 48 | 0.6906 | 33.1 | 79.8 | 29.1 |
| linear 1568 x 48 x 5120 | in_proj_ba | 192 | 0.1451 | 27.9 | 5.3 | 11.7 |
| linear 784 x 5120 x 3072 | o_proj/out_proj | 64 | 0.4058 | 26.0 | 60.8 | 20.6 |
| linear 657 x 5120 x 3072 | o_proj/out_proj | 64 | 0.3429 | 21.9 | 60.3 | 16.6 |
| linear 784 x 7168 x 5120 | qkv_proj | 16 | 0.7183 | 11.5 | 80.1 | 10.1 |
| linear 657 x 7168 x 5120 | qkv_proj | 16 | 0.6098 | 9.8 | 79.1 | 8.4 |
| linear 784 x 48 x 5120 | in_proj_ba | 48 | 0.1377 | 6.6 | 2.8 | 2.6 |
| linear 657 x 48 x 5120 | in_proj_ba | 48 | 0.1375 | 6.6 | 2.3 | 2.6 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 288, 227.2; causal_conv1d_fn 288, 54.2; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 6, 7.2; all_gather 6, 1.1.
Glue by neighbours, raw ms (previous operator > next): gate_up_proj>down_proj 44.4; gdn>o_proj/out_proj 43.5; comm>in_proj_qkvz 39.2; comm>gate_up_proj 35.7; in_proj_ba>gdn 32.1; qkv_proj>kvw 16.2; run>comm 16.1; o_proj/out_proj>comm 14.7; down_proj>comm 13.9; in_proj_qkvz>in_proj_ba 13.2; comm>qkv_proj 13.2; attn>o_proj/out_proj 11.6; kvw>attn 4.0; run>run 2.7.

### `d` pf1340: prompt [16115] tokens, per prefill, all its steps, steps [1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 1568, 435]

Ops arm 6447.6 ms, timed parts sum 6443.5 ms, same flags with one synchronise per step 5346.6 ms, 13618 synchronises, c = 81 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 1408 | 2404.2 | 37.3% | 2290.3 | 42.9% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 1056 | 739.1 | 11.5% | 653.7 | 12.2% |
| attention projection GEMM (qkv_proj, o_proj) | 880 | 578.0 | 9.0% | 506.8 | 9.5% |
| attention kernel | 176 | 951.1 | 14.8% | 936.9 | 17.5% |
| KV-cache write | 176 | 27.6 | 0.4% | 13.4 | 0.3% |
| GDN core (conv + delta rule) | 528 | 741.5 | 11.5% | 528.1 | 9.9% |
| tensor-parallel all-reduce | 1419 | 388.8 | 6.0% | 274.1 | 5.1% |
| sampling incl. lm_head GEMM and all-gather | 11 | 21.4 | 0.3% | 16.9 | 0.3% |
| runner: input prep, request bookkeeping, Python | 77 | 19.4 | 0.3% | 13.2 | 0.2% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 5643 | 534.3 | 8.3% | 78.1 | 1.5% |
| runner glue | 89 | 38.2 | 0.6% | 31.1 | 0.6% |
| sum | | 6443.5 | | 5342.5 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 1568 x 17408 x 5120 | gate_up_proj | 640 | 2.4078 | 1541.0 | 116.1 | 1489.2 |
| linear 1568 x 5120 x 8704 | down_proj | 640 | 1.2202 | 780.9 | 114.5 | 729.2 |
| linear 1568 x 8192 x 5120 | in_proj_qkvz | 480 | 1.3352 | 640.9 | 98.5 | 602.1 |
| linear 1568 x 5120 x 3072 | o_proj/out_proj | 640 | 0.5913 | 378.4 | 83.4 | 326.7 |
| linear 1568 x 7168 x 5120 | qkv_proj | 160 | 1.0931 | 174.9 | 105.3 | 162.0 |
| linear 1568 x 48 x 5120 | in_proj_ba | 480 | 0.1449 | 69.5 | 5.3 | 30.7 |
| linear 435 x 17408 x 5120 | gate_up_proj | 64 | 0.8287 | 53.0 | 93.6 | 47.9 |
| linear 435 x 5120 x 8704 | down_proj | 64 | 0.4568 | 29.2 | 84.9 | 24.1 |
| linear 435 x 8192 x 5120 | in_proj_qkvz | 48 | 0.4592 | 22.0 | 79.5 | 18.2 |
| linear 435 x 5120 x 3072 | o_proj/out_proj | 64 | 0.2699 | 17.3 | 50.7 | 12.1 |
| linear 435 x 7168 x 5120 | qkv_proj | 16 | 0.4594 | 7.3 | 69.5 | 6.1 |
| linear 435 x 48 x 5120 | in_proj_ba | 48 | 0.1378 | 6.6 | 1.6 | 2.7 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 528, 429.3; causal_conv1d_fn 528, 103.6; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 11, 13.2; all_gather 11, 1.9.
Glue by neighbours, raw ms (previous operator > next): gate_up_proj>down_proj 87.4; gdn>o_proj/out_proj 79.7; comm>in_proj_qkvz 78.2; comm>gate_up_proj 69.3; in_proj_ba>gdn 58.6; run>comm 29.7; qkv_proj>kvw 29.6; o_proj/out_proj>comm 27.0; comm>qkv_proj 26.3; down_proj>comm 25.6; in_proj_qkvz>in_proj_ba 24.3; attn>o_proj/out_proj 21.1; kvw>attn 7.3; run>run 5.0.

### `d` dc1: prompt [7709] tokens, per decode step, 94 steps

Ops arm 135.0 ms, timed parts sum 135.0 ms, same flags with one synchronise per step 70.6 ms, 1334 synchronises, c = 48 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 128 | 20.8 | 15.4% | 14.6 | 20.7% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 96 | 8.2 | 6.0% | 3.5 | 5.0% |
| attention projection GEMM (qkv_proj, o_proj) | 80 | 6.8 | 5.0% | 2.9 | 4.1% |
| attention kernel | 16 | 28.7 | 21.3% | 28.0 | 39.6% |
| KV-cache write | 16 | 2.2 | 1.6% | 1.4 | 2.0% |
| GDN core (conv + delta rule) | 48 | 15.7 | 11.6% | -0.6 | -0.8% |
| tensor-parallel all-reduce | 129 | 11.8 | 8.8% | 5.6 | 7.9% |
| sampling incl. lm_head GEMM and all-gather | 1 | 1.9 | 1.4% | 1.6 | 2.3% |
| runner: input prep, request bookkeeping, Python | 7 | 1.1 | 0.8% | 0.7 | 1.1% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 513 | 36.2 | 26.8% | 11.4 | 16.2% |
| runner glue | 9 | 1.7 | 1.2% | 1.3 | 1.8% |
| sum | | 135.0 | | 70.5 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| wvSplitK 1 x 17408 x 5120 | gate_up_proj | 64 | 0.2073 | 13.3 | 0.9 | 10.2 |
| wvSplitK 1 x 5120 x 8704 | down_proj | 64 | 0.1175 | 7.5 | 0.8 | 4.4 |
| wvSplitK 1 x 8192 x 5120 | in_proj_qkvz | 48 | 0.1172 | 5.6 | 0.7 | 3.3 |
| wvSplitK 1 x 5120 x 3072 | o_proj/out_proj | 64 | 0.0792 | 5.1 | 0.4 | 2.0 |
| wvSplitK 1 x 48 x 5120 | in_proj_ba | 48 | 0.0529 | 2.5 | 0.0 | 0.2 |
| wvSplitK 1 x 7168 x 5120 | qkv_proj | 16 | 0.1075 | 1.7 | 0.7 | 0.9 |

Nested timers, raw (calls, ms): _forward_core_decode_non_spec 48, 14.8; causal_conv1d_update 48, 5.6; fused_recurrent_gated_delta_rule_packed_decode 48, 5.4; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 1, 1.2; all_gather 1, 0.2.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 6.6; gate_up_proj>down_proj 4.9; comm>gate_up_proj 4.8; in_proj_ba>gdn 4.6; comm>in_proj_qkvz 3.9; qkv_proj>kvw 2.3; o_proj/out_proj>comm 1.9; attn>o_proj/out_proj 1.8; down_proj>comm 1.8; in_proj_qkvz>in_proj_ba 1.7; comm>qkv_proj 1.3; run>comm 1.2; kvw>attn 0.6; run>run 0.3.

### `d` dc8: prompt [7709] tokens, per decode step, 59 steps

Ops arm 166.9 ms, timed parts sum 166.9 ms, same flags with one synchronise per step 93.2 ms, 1334 synchronises, c = 55 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 128 | 31.5 | 18.9% | 24.4 | 26.2% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 96 | 15.1 | 9.0% | 9.8 | 10.5% |
| attention projection GEMM (qkv_proj, o_proj) | 80 | 12.5 | 7.5% | 8.1 | 8.7% |
| attention kernel | 16 | 28.1 | 16.8% | 27.2 | 29.2% |
| KV-cache write | 16 | 2.3 | 1.4% | 1.4 | 1.5% |
| GDN core (conv + delta rule) | 48 | 17.0 | 10.2% | -1.5 | -1.6% |
| tensor-parallel all-reduce | 129 | 15.2 | 9.1% | 8.1 | 8.7% |
| sampling incl. lm_head GEMM and all-gather | 1 | 2.3 | 1.4% | 2.0 | 2.1% |
| runner: input prep, request bookkeeping, Python | 7 | 1.1 | 0.7% | 0.7 | 0.8% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 513 | 39.9 | 23.9% | 11.6 | 12.5% |
| runner glue | 9 | 1.9 | 1.1% | 1.4 | 1.5% |
| sum | | 166.9 | | 93.2 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 8 x 17408 x 5120 | gate_up_proj | 64 | 0.2919 | 18.7 | 4.9 | 15.1 |
| linear 8 x 5120 x 8704 | down_proj | 64 | 0.2000 | 12.8 | 3.6 | 9.3 |
| linear 8 x 5120 x 3072 | o_proj/out_proj | 64 | 0.1490 | 9.5 | 1.7 | 6.0 |
| linear 8 x 8192 x 5120 | in_proj_qkvz | 48 | 0.1938 | 9.3 | 3.5 | 6.7 |
| linear 8 x 48 x 5120 | in_proj_ba | 48 | 0.1207 | 5.8 | 0.0 | 3.1 |
| linear 8 x 7168 x 5120 | qkv_proj | 16 | 0.1857 | 3.0 | 3.2 | 2.1 |

Nested timers, raw (calls, ms): _forward_core_decode_non_spec 48, 16.2; fused_recurrent_gated_delta_rule_packed_decode 48, 6.5; causal_conv1d_update 48, 5.8; gemm|linear|8|124160|5120|bfloat16|bfloat16|b0 1, 1.5; all_gather 1, 0.2.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 6.8; gate_up_proj>down_proj 5.6; in_proj_ba>gdn 5.3; comm>gate_up_proj 5.0; comm>in_proj_qkvz 4.1; qkv_proj>kvw 2.6; o_proj/out_proj>comm 2.4; down_proj>comm 2.3; in_proj_qkvz>in_proj_ba 2.2; attn>o_proj/out_proj 1.8; comm>qkv_proj 1.4; run>comm 1.3; kvw>attn 0.6; run>run 0.3.

## Flags `m16` (ops arm po_m16, job 23624390)

**Cross-check [observed]: client time to first token (s) or decode step (ms), and the server-side step time summed over the prefill's steps (ms) or per decode step, worker 0**

| Cell | prompt tokens | steps (tokens) | s_m16 client | py_m16 client | pyn_m16 client | po_m16 client | py_m16 server | pyn_m16 server | po_m16 server | ps period, ms |
|---|---|---|---|---|---|---|---|---|---|---|
| pf115 | [1413] | [784, 629] | 0.4566 | 0.4628 | 0.5135 | 0.7634 | 451.5 | 505.0 | 755.5 | - |
| pf330 | [3993] | [3920, 73] | 1.22 | 1.2286 | 1.4109 | 1.611 | 1208.5 | 1397.3 | 1597.5 | - |
| pf640 | [7713] | [7056, 657] | 2.2373 | 2.2499 | 2.5397 | 2.7591 | 2218.9 | 2522.8 | 2737.3 | - |
| pf1340 | [16115] | [15680, 435] | 4.7254 | 4.7409 | 5.3303 | 5.5472 | 4687.2 | 5292.4 | 5507.5 | - |
| dc1 | [7709] | 94 steps | 55.828 | 58.163 | 63.975 | 144.561 | 58.0 | 63.8 | 144.1 | - |
| dc8 | [7709] | 89 steps | 69.499 | 71.499 | 84.993 | 173.467 | 71.4 | 84.9 | 173.9 | - |
| dc32 | [7711, 7713] | 78 steps | 72.948 | 75.37 | 86.529 | 180.845 | 75.3 | 86.2 | 180.0 | - |

### `m16` pf115: prompt [1413] tokens, per prefill, all its steps, steps [784, 629]

Ops arm 755.5 ms, timed parts sum 755.0 ms, same flags with one synchronise per step 505.0 ms, 2476 synchronises, c = 101 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 256 | 242.7 | 32.1% | 216.8 | 43.0% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 192 | 79.3 | 10.5% | 59.8 | 11.9% |
| attention projection GEMM (qkv_proj, o_proj) | 160 | 68.8 | 9.1% | 52.6 | 10.4% |
| attention kernel | 32 | 19.6 | 2.6% | 16.4 | 3.2% |
| KV-cache write | 32 | 5.0 | 0.7% | 1.7 | 0.3% |
| GDN core (conv + delta rule) | 96 | 126.5 | 16.7% | 77.9 | 15.4% |
| tensor-parallel all-reduce | 258 | 107.0 | 14.2% | 80.9 | 16.0% |
| sampling incl. lm_head GEMM and all-gather | 2 | 4.0 | 0.5% | 3.0 | 0.6% |
| runner: input prep, request bookkeeping, Python | 14 | 3.9 | 0.5% | 2.5 | 0.5% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 1026 | 90.8 | 12.0% | -13.1 | -2.6% |
| runner glue | 17 | 7.5 | 1.0% | 5.9 | 1.2% |
| sum | | 755.0 | | 504.5 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 784 x 17408 x 5120 | gate_up_proj | 64 | 1.2945 | 82.8 | 108.0 | 76.4 |
| linear 629 x 17408 x 5120 | gate_up_proj | 64 | 1.1265 | 72.1 | 99.5 | 65.6 |
| linear 784 x 5120 x 8704 | down_proj | 64 | 0.7744 | 49.6 | 90.2 | 43.1 |
| linear 629 x 5120 x 8704 | down_proj | 64 | 0.5967 | 38.2 | 94.0 | 31.7 |
| linear 784 x 8192 x 5120 | in_proj_qkvz | 48 | 0.7452 | 35.8 | 88.3 | 30.9 |
| linear 629 x 8192 x 5120 | in_proj_qkvz | 48 | 0.6189 | 29.7 | 85.3 | 24.9 |
| linear 784 x 5120 x 3072 | o_proj/out_proj | 64 | 0.4081 | 26.1 | 60.4 | 19.6 |
| linear 629 x 5120 x 3072 | o_proj/out_proj | 64 | 0.3383 | 21.7 | 58.5 | 15.2 |
| linear 784 x 7168 x 5120 | qkv_proj | 16 | 0.7176 | 11.5 | 80.2 | 9.9 |
| linear 629 x 7168 x 5120 | qkv_proj | 16 | 0.5942 | 9.5 | 77.7 | 7.9 |
| linear 784 x 48 x 5120 | in_proj_ba | 48 | 0.1438 | 6.9 | 2.7 | 2.0 |
| linear 629 x 48 x 5120 | in_proj_ba | 48 | 0.1433 | 6.9 | 2.2 | 2.0 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 96, 70.2; causal_conv1d_fn 96, 16.3; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 2, 2.4; all_gather 2, 0.4.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 14.2; gate_up_proj>down_proj 12.8; in_proj_ba>gdn 12.6; comm>gate_up_proj 10.7; comm>in_proj_qkvz 9.8; qkv_proj>kvw 6.5; run>comm 5.9; o_proj/out_proj>comm 5.4; down_proj>comm 5.1; in_proj_qkvz>in_proj_ba 4.8; attn>o_proj/out_proj 4.1; comm>qkv_proj 3.3; kvw>attn 1.4; run>run 0.9.

### `m16` pf330: prompt [3993] tokens, per prefill, all its steps, steps [3920, 73]

Ops arm 1597.5 ms, timed parts sum 1596.6 ms, same flags with one synchronise per step 1397.3 ms, 2476 synchronises, c = 81 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 256 | 647.2 | 40.5% | 626.5 | 44.9% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 192 | 162.4 | 10.2% | 146.9 | 10.5% |
| attention projection GEMM (qkv_proj, o_proj) | 160 | 142.1 | 8.9% | 129.2 | 9.3% |
| attention kernel | 32 | 67.6 | 4.2% | 65.0 | 4.7% |
| KV-cache write | 32 | 5.7 | 0.4% | 3.1 | 0.2% |
| GDN core (conv + delta rule) | 96 | 161.3 | 10.1% | 122.5 | 8.8% |
| tensor-parallel all-reduce | 258 | 273.6 | 17.1% | 252.7 | 18.1% |
| sampling incl. lm_head GEMM and all-gather | 2 | 4.0 | 0.3% | 3.2 | 0.2% |
| runner: input prep, request bookkeeping, Python | 14 | 4.5 | 0.3% | 3.4 | 0.2% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 1026 | 120.7 | 7.6% | 37.8 | 2.7% |
| runner glue | 17 | 7.5 | 0.5% | 6.2 | 0.4% |
| sum | | 1596.6 | | 1396.4 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 3920 x 17408 x 5120 | gate_up_proj | 64 | 6.3678 | 407.5 | 109.7 | 402.4 |
| linear 3920 x 5120 x 8704 | down_proj | 64 | 3.1868 | 204.0 | 109.6 | 198.8 |
| linear 3920 x 8192 x 5120 | in_proj_qkvz | 48 | 2.8615 | 137.4 | 114.9 | 133.5 |
| linear 3920 x 5120 x 3072 | o_proj/out_proj | 64 | 1.3316 | 85.2 | 92.6 | 80.0 |
| linear 3920 x 7168 x 5120 | qkv_proj | 16 | 2.6234 | 42.0 | 109.7 | 40.7 |
| linear 73 x 17408 x 5120 | gate_up_proj | 64 | 0.3256 | 20.8 | 40.0 | 15.7 |
| linear 73 x 5120 x 8704 | down_proj | 64 | 0.2321 | 14.9 | 28.0 | 9.7 |
| linear 73 x 5120 x 3072 | o_proj/out_proj | 64 | 0.1793 | 11.5 | 12.8 | 6.3 |
| linear 73 x 8192 x 5120 | in_proj_qkvz | 48 | 0.2210 | 10.6 | 27.7 | 6.7 |
| linear 3920 x 48 x 5120 | in_proj_ba | 48 | 0.1692 | 8.1 | 11.4 | 4.2 |
| linear 73 x 48 x 5120 | in_proj_ba | 48 | 0.1321 | 6.3 | 0.3 | 2.5 |
| linear 73 x 7168 x 5120 | qkv_proj | 16 | 0.2157 | 3.5 | 24.8 | 2.2 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 96, 98.7; causal_conv1d_fn 96, 22.1; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 2, 2.4; all_gather 2, 0.4.
Glue by neighbours, raw ms (previous operator > next): gate_up_proj>down_proj 21.4; comm>in_proj_qkvz 19.2; comm>gate_up_proj 16.5; gdn>o_proj/out_proj 16.4; in_proj_ba>gdn 12.6; qkv_proj>kvw 6.8; comm>qkv_proj 6.4; run>comm 5.7; o_proj/out_proj>comm 5.4; down_proj>comm 5.1; in_proj_qkvz>in_proj_ba 4.8; attn>o_proj/out_proj 4.7; kvw>attn 1.4; run>run 1.0.

### `m16` pf640: prompt [7713] tokens, per prefill, all its steps, steps [7056, 657]

Ops arm 2737.3 ms, timed parts sum 2738.2 ms, same flags with one synchronise per step 2522.8 ms, 2476 synchronises, c = 87 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 256 | 1170.7 | 42.8% | 1148.5 | 45.5% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 192 | 305.1 | 11.1% | 288.4 | 11.4% |
| attention projection GEMM (qkv_proj, o_proj) | 160 | 246.3 | 9.0% | 232.4 | 9.2% |
| attention kernel | 32 | 158.1 | 5.8% | 155.3 | 6.2% |
| KV-cache write | 32 | 6.5 | 0.2% | 3.7 | 0.1% |
| GDN core (conv + delta rule) | 96 | 216.8 | 7.9% | 175.2 | 6.9% |
| tensor-parallel all-reduce | 258 | 462.9 | 16.9% | 440.6 | 17.5% |
| sampling incl. lm_head GEMM and all-gather | 2 | 4.0 | 0.1% | 3.1 | 0.1% |
| runner: input prep, request bookkeeping, Python | 14 | 5.4 | 0.2% | 4.2 | 0.2% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 1026 | 154.7 | 5.6% | 65.8 | 2.6% |
| runner glue | 17 | 7.8 | 0.3% | 6.5 | 0.3% |
| sum | | 2738.2 | | 2523.6 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 7056 x 17408 x 5120 | gate_up_proj | 64 | 11.0753 | 708.8 | 113.6 | 703.3 |
| linear 7056 x 5120 x 8704 | down_proj | 64 | 5.4423 | 348.3 | 115.6 | 342.8 |
| linear 7056 x 8192 x 5120 | in_proj_qkvz | 48 | 5.3097 | 254.9 | 111.5 | 250.7 |
| linear 7056 x 5120 x 3072 | o_proj/out_proj | 64 | 2.1768 | 139.3 | 102.0 | 133.8 |
| linear 7056 x 7168 x 5120 | qkv_proj | 16 | 4.6724 | 74.8 | 110.8 | 73.4 |
| linear 657 x 17408 x 5120 | gate_up_proj | 64 | 1.1280 | 72.2 | 103.8 | 66.6 |
| linear 657 x 5120 x 8704 | down_proj | 64 | 0.6471 | 41.4 | 90.5 | 35.9 |
| linear 657 x 8192 x 5120 | in_proj_qkvz | 48 | 0.6982 | 33.5 | 78.9 | 29.4 |
| linear 657 x 5120 x 3072 | o_proj/out_proj | 64 | 0.3500 | 22.4 | 59.0 | 16.9 |
| linear 657 x 7168 x 5120 | qkv_proj | 16 | 0.6108 | 9.8 | 79.0 | 8.4 |
| linear 7056 x 48 x 5120 | in_proj_ba | 48 | 0.2033 | 9.8 | 17.1 | 5.6 |
| linear 657 x 48 x 5120 | in_proj_ba | 48 | 0.1445 | 6.9 | 2.2 | 2.8 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 96, 141.7; causal_conv1d_fn 96, 29.6; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 2, 2.4; all_gather 2, 0.4.
Glue by neighbours, raw ms (previous operator > next): gate_up_proj>down_proj 30.1; comm>in_proj_qkvz 28.6; comm>gate_up_proj 22.2; gdn>o_proj/out_proj 20.2; in_proj_ba>gdn 12.7; comm>qkv_proj 9.7; qkv_proj>kvw 8.4; run>comm 5.9; attn>o_proj/out_proj 5.7; o_proj/out_proj>comm 5.5; down_proj>comm 5.3; in_proj_qkvz>in_proj_ba 4.9; kvw>attn 1.4; run>run 0.9.

### `m16` pf1340: prompt [16115] tokens, per prefill, all its steps, steps [15680, 435]

Ops arm 5507.5 ms, timed parts sum 5507.4 ms, same flags with one synchronise per step 5292.4 ms, 2476 synchronises, c = 87 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 256 | 2328.2 | 42.3% | 2305.9 | 43.6% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 192 | 602.4 | 10.9% | 585.8 | 11.1% |
| attention projection GEMM (qkv_proj, o_proj) | 160 | 456.1 | 8.3% | 442.2 | 8.4% |
| attention kernel | 32 | 574.6 | 10.4% | 571.8 | 10.8% |
| KV-cache write | 32 | 8.5 | 0.2% | 5.7 | 0.1% |
| GDN core (conv + delta rule) | 96 | 361.0 | 6.6% | 319.3 | 6.0% |
| tensor-parallel all-reduce | 258 | 913.6 | 16.6% | 891.2 | 16.8% |
| sampling incl. lm_head GEMM and all-gather | 2 | 4.0 | 0.1% | 3.1 | 0.1% |
| runner: input prep, request bookkeeping, Python | 14 | 8.1 | 0.1% | 6.9 | 0.1% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 1026 | 242.1 | 4.4% | 152.9 | 2.9% |
| runner glue | 17 | 8.9 | 0.2% | 7.5 | 0.1% |
| sum | | 5507.4 | | 5292.3 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 15680 x 17408 x 5120 | gate_up_proj | 64 | 24.0502 | 1539.2 | 116.2 | 1533.7 |
| linear 15680 x 5120 x 8704 | down_proj | 64 | 11.0326 | 706.1 | 126.7 | 700.5 |
| linear 15680 x 8192 x 5120 | in_proj_qkvz | 48 | 11.6368 | 558.6 | 113.0 | 554.4 |
| linear 15680 x 5120 x 3072 | o_proj/out_proj | 64 | 4.2155 | 269.8 | 117.0 | 264.2 |
| linear 15680 x 7168 x 5120 | qkv_proj | 16 | 10.0706 | 161.1 | 114.3 | 159.7 |
| linear 435 x 17408 x 5120 | gate_up_proj | 64 | 0.8301 | 53.1 | 93.4 | 47.6 |
| linear 435 x 5120 x 8704 | down_proj | 64 | 0.4647 | 29.7 | 83.4 | 24.2 |
| linear 435 x 8192 x 5120 | in_proj_qkvz | 48 | 0.4615 | 22.2 | 79.1 | 18.0 |
| linear 435 x 5120 x 3072 | o_proj/out_proj | 64 | 0.2772 | 17.7 | 49.4 | 12.2 |
| linear 15680 x 48 x 5120 | in_proj_ba | 48 | 0.3085 | 14.8 | 25.0 | 10.6 |
| linear 435 x 7168 x 5120 | qkv_proj | 16 | 0.4641 | 7.4 | 68.8 | 6.0 |
| linear 435 x 48 x 5120 | in_proj_ba | 48 | 0.1441 | 6.9 | 1.5 | 2.7 |

Nested timers, raw (calls, ms): fla_chunk_gated_delta_rule 96, 251.3; causal_conv1d_fn 96, 47.0; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 2, 2.4; all_gather 2, 0.4.
Glue by neighbours, raw ms (previous operator > next): gate_up_proj>down_proj 53.2; comm>in_proj_qkvz 51.2; comm>gate_up_proj 38.4; gdn>o_proj/out_proj 30.7; comm>qkv_proj 17.2; in_proj_ba>gdn 12.9; qkv_proj>kvw 12.8; attn>o_proj/out_proj 8.4; run>comm 6.4; o_proj/out_proj>comm 5.5; down_proj>comm 5.3; in_proj_qkvz>in_proj_ba 4.9; kvw>attn 1.4; comm>exec_end 1.0.

### `m16` dc1: prompt [7709] tokens, per decode step, 94 steps

Ops arm 144.1 ms, timed parts sum 144.0 ms, same flags with one synchronise per step 63.8 ms, 1334 synchronises, c = 60 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 128 | 21.0 | 14.6% | 13.3 | 20.9% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 96 | 8.4 | 5.8% | 2.6 | 4.1% |
| attention projection GEMM (qkv_proj, o_proj) | 80 | 7.0 | 4.8% | 2.1 | 3.4% |
| attention kernel | 16 | 29.2 | 20.3% | 28.3 | 44.3% |
| KV-cache write | 16 | 2.3 | 1.6% | 1.4 | 2.1% |
| GDN core (conv + delta rule) | 48 | 17.0 | 11.8% | -3.2 | -5.0% |
| tensor-parallel all-reduce | 129 | 15.9 | 11.1% | 8.2 | 12.8% |
| sampling incl. lm_head GEMM and all-gather | 1 | 1.9 | 1.3% | 1.5 | 2.4% |
| runner: input prep, request bookkeeping, Python | 7 | 1.1 | 0.8% | 0.7 | 1.1% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 513 | 38.6 | 26.8% | 7.7 | 12.1% |
| runner glue | 9 | 1.7 | 1.2% | 1.2 | 1.9% |
| sum | | 144.0 | | 63.7 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| wvSplitK 1 x 17408 x 5120 | gate_up_proj | 64 | 0.2086 | 13.4 | 0.9 | 9.5 |
| wvSplitK 1 x 5120 x 8704 | down_proj | 64 | 0.1200 | 7.7 | 0.7 | 3.8 |
| wvSplitK 1 x 8192 x 5120 | in_proj_qkvz | 48 | 0.1193 | 5.7 | 0.7 | 2.8 |
| wvSplitK 1 x 5120 x 3072 | o_proj/out_proj | 64 | 0.0814 | 5.2 | 0.4 | 1.4 |
| wvSplitK 1 x 48 x 5120 | in_proj_ba | 48 | 0.0553 | 2.7 | 0.0 | -0.2 |
| wvSplitK 1 x 7168 x 5120 | qkv_proj | 16 | 0.1095 | 1.8 | 0.7 | 0.8 |

Nested timers, raw (calls, ms): _forward_core_decode_non_spec 48, 16.1; fused_recurrent_gated_delta_rule_packed_decode 48, 6.1; causal_conv1d_update 48, 5.9; gemm|wvSplitK|1|124160|5120|bfloat16|bfloat16|b0 1, 1.2; all_gather 1, 0.2.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 6.3; in_proj_ba>gdn 5.3; gate_up_proj>down_proj 5.1; comm>gate_up_proj 5.1; comm>in_proj_qkvz 4.2; qkv_proj>kvw 2.8; o_proj/out_proj>comm 2.1; down_proj>comm 1.9; attn>o_proj/out_proj 1.9; in_proj_qkvz>in_proj_ba 1.8; comm>qkv_proj 1.4; run>comm 1.2; kvw>attn 0.7; run>run 0.3.

### `m16` dc8: prompt [7709] tokens, per decode step, 89 steps

Ops arm 173.9 ms, timed parts sum 173.9 ms, same flags with one synchronise per step 84.9 ms, 1334 synchronises, c = 67 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 128 | 31.8 | 18.3% | 23.3 | 27.4% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 96 | 15.5 | 8.9% | 9.1 | 10.8% |
| attention projection GEMM (qkv_proj, o_proj) | 80 | 12.9 | 7.4% | 7.5 | 8.9% |
| attention kernel | 16 | 28.4 | 16.4% | 27.4 | 32.3% |
| KV-cache write | 16 | 2.4 | 1.4% | 1.3 | 1.6% |
| GDN core (conv + delta rule) | 48 | 18.0 | 10.3% | -4.5 | -5.3% |
| tensor-parallel all-reduce | 129 | 17.3 | 10.0% | 8.7 | 10.3% |
| sampling incl. lm_head GEMM and all-gather | 1 | 2.3 | 1.3% | 2.0 | 2.3% |
| runner: input prep, request bookkeeping, Python | 7 | 1.1 | 0.7% | 0.7 | 0.8% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 513 | 42.3 | 24.3% | 8.0 | 9.5% |
| runner glue | 9 | 1.8 | 1.1% | 1.3 | 1.5% |
| sum | | 173.9 | | 84.9 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 8 x 17408 x 5120 | gate_up_proj | 64 | 0.2956 | 18.9 | 4.8 | 14.6 |
| linear 8 x 5120 x 8704 | down_proj | 64 | 0.2012 | 12.9 | 3.5 | 8.6 |
| linear 8 x 5120 x 3072 | o_proj/out_proj | 64 | 0.1535 | 9.8 | 1.6 | 5.6 |
| linear 8 x 8192 x 5120 | in_proj_qkvz | 48 | 0.1985 | 9.5 | 3.4 | 6.3 |
| linear 8 x 48 x 5120 | in_proj_ba | 48 | 0.1253 | 6.0 | 0.0 | 2.8 |
| linear 8 x 7168 x 5120 | qkv_proj | 16 | 0.1904 | 3.0 | 3.1 | 2.0 |

Nested timers, raw (calls, ms): _forward_core_decode_non_spec 48, 17.0; fused_recurrent_gated_delta_rule_packed_decode 48, 6.8; causal_conv1d_update 48, 6.1; gemm|linear|8|124160|5120|bfloat16|bfloat16|b0 1, 1.5; all_gather 1, 0.3.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 6.4; in_proj_ba>gdn 6.0; gate_up_proj>down_proj 5.9; comm>gate_up_proj 5.2; comm>in_proj_qkvz 4.3; qkv_proj>kvw 3.0; o_proj/out_proj>comm 2.6; down_proj>comm 2.5; in_proj_qkvz>in_proj_ba 2.3; attn>o_proj/out_proj 1.9; comm>qkv_proj 1.4; run>comm 1.3; kvw>attn 0.7; run>run 0.3.

### `m16` dc32: prompt [7711, 7713] tokens, per decode step, 78 steps

Ops arm 180.0 ms, timed parts sum 180.0 ms, same flags with one synchronise per step 86.2 ms, 1334 synchronises, c = 70 us [computed].

| Operator class | calls | raw ms [observed] | raw share | corrected ms [computed] | corrected share |
|---|---|---|---|---|---|
| MLP GEMM (gate_up_proj, down_proj) | 128 | 32.0 | 17.8% | 23.0 | 26.6% |
| GDN projection GEMM (in_proj_qkvz, in_proj_ba, out_proj) | 96 | 15.5 | 8.6% | 8.8 | 10.2% |
| attention projection GEMM (qkv_proj, o_proj) | 80 | 12.7 | 7.1% | 7.1 | 8.2% |
| attention kernel | 16 | 28.8 | 16.0% | 27.7 | 32.2% |
| KV-cache write | 16 | 2.5 | 1.4% | 1.3 | 1.5% |
| GDN core (conv + delta rule) | 48 | 21.4 | 11.9% | -2.3 | -2.6% |
| tensor-parallel all-reduce | 129 | 18.9 | 10.5% | 9.8 | 11.4% |
| sampling incl. lm_head GEMM and all-gather | 1 | 2.7 | 1.5% | 2.3 | 2.7% |
| runner: input prep, request bookkeeping, Python | 7 | 1.2 | 0.7% | 0.7 | 0.8% |
| compiled glue in the forward: norms, activations, residual adds, RoPE, embedding, reshapes | 513 | 42.4 | 23.6% | 6.4 | 7.4% |
| runner glue | 9 | 1.9 | 1.0% | 1.3 | 1.5% |
| sum | | 180.0 | | 86.2 | |

| GEMM: path, M x N x K (bf16 in, bf16 weight, no bias) | layer | calls | raw ms per call | raw ms | TFLOP/s raw | corrected ms |
|---|---|---|---|---|---|---|
| linear 32 x 17408 x 5120 | gate_up_proj | 64 | 0.2951 | 18.9 | 19.3 | 14.4 |
| linear 32 x 5120 x 8704 | down_proj | 64 | 0.2042 | 13.1 | 14.0 | 8.6 |
| linear 32 x 5120 x 3072 | o_proj/out_proj | 64 | 0.1511 | 9.7 | 6.7 | 5.2 |
| linear 32 x 8192 x 5120 | in_proj_qkvz | 48 | 0.1988 | 9.5 | 13.5 | 6.2 |
| linear 32 x 48 x 5120 | in_proj_ba | 48 | 0.1248 | 6.0 | 0.1 | 2.6 |
| linear 32 x 7168 x 5120 | qkv_proj | 16 | 0.1912 | 3.1 | 12.3 | 1.9 |

Nested timers, raw (calls, ms): _forward_core_decode_non_spec 48, 20.4; fused_recurrent_gated_delta_rule_packed_decode 48, 10.1; causal_conv1d_update 48, 6.2; gemm|linear|32|124160|5120|bfloat16|bfloat16|b0 1, 1.6; all_gather 1, 0.6.
Glue by neighbours, raw ms (previous operator > next): gdn>o_proj/out_proj 6.4; in_proj_ba>gdn 6.0; gate_up_proj>down_proj 5.9; comm>gate_up_proj 5.2; comm>in_proj_qkvz 4.3; qkv_proj>kvw 3.0; o_proj/out_proj>comm 2.6; down_proj>comm 2.5; in_proj_qkvz>in_proj_ba 2.3; attn>o_proj/out_proj 1.9; comm>qkv_proj 1.4; run>comm 1.3; kvw>attn 0.7; run>run 0.3.
