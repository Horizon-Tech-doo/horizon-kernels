
## Job 23619582

- ## job 23619582 start 22:08:19 arms fv_rocm s_rocm f_rocm f_rocm s_rocm
- Result rows (timed cells of 6, client finished): f_rocm-r0 6 True, f_rocm-r1 6 True, fv_rocm-r0 6 True, s_rocm-r0 6 True, s_rocm-r1 6 True

**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)

| Cell (prompt tokens x streams) | f_rocm-r0 | f_rocm-r1 | fv_rocm-r0 | s_rocm-r0 | s_rocm-r1 |
|---|---|---|---|---|---|
| 1419 x1 | 0.5148 [0.5146-0.5159] | 0.5169 [0.5143-0.5175] | 0.5162 [0.5149-0.5165] | 0.5138 [0.5132-0.5149] | 0.5158 [0.5134-0.5162] |
| 2859 x1 | 1.0004 [1.0000-1.0261] | 1.0001 [0.9997-1.0223] | 1.0040 [1.0033-1.0229] | 0.9978 [0.9976-0.9984] | 0.9971 [0.9966-0.9973] |
| 5679 x1 | 1.9734 [1.9727-1.9771] | 1.9715 [1.9712-2.0028] | 1.9796 [1.9791-1.9904] | 1.9692 [1.9689-1.9695] | 1.9668 [1.9660-1.9679] |
| 7719 x1 | 2.6890 [2.6886-2.6978] | 2.6859 [2.6855-2.7069] | 2.6973 [2.6967-2.7002] | 2.6846 [2.6831-2.6853] | 2.6817 [2.6796-2.6820] |
| 14201 x1 | 5.2676 [5.2588-5.2942] | 5.2506 [5.2477-5.2995] | 5.2762 [5.2723-5.3075] | 5.2514 [5.2477-5.2656] | 5.2432 [5.2394-5.2440] |
| 2859 x4 | 2.7112 [1.2290-4.0048] | 2.6873 [1.2273-3.9588] | 2.7115 [1.2288-4.0048] | 2.6828 [1.2258-3.9467] | 2.6785 [1.2234-3.9427] |

**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**

| Cell | s reps | f reps | f / s | s spread | f spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.5138, 0.5158 | 0.5148, 0.5169 | 1.0020 | 0.39% | 0.41% | ~ inside noise |
| 2859 x1 | 0.9978, 0.9971 | 1.0004, 1.0001 | 1.0028 | 0.07% | 0.03% | plugin slower |
| 5679 x1 | 1.9692, 1.9668 | 1.9734, 1.9715 | 1.0023 | 0.12% | 0.10% | plugin slower |
| 7719 x1 | 2.6846, 2.6817 | 2.6890, 2.6859 | 1.0016 | 0.11% | 0.12% | plugin slower |
| 14201 x1 | 5.2514, 5.2432 | 5.2676, 5.2506 | 1.0022 | 0.16% | 0.32% | ~ inside noise |
| 2859 x4 | 2.6828, 2.6785 | 2.7112, 2.6873 | 1.0069 | 0.16% | 0.89% | ~ inside noise |

**Plugin counters per worker (attention-layer calls; 16 per step)**

| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_rocm-r0 | 53140 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r0 | 53141 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 56587 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 56588 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| fv_rocm-r0 | 29917 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| fv_rocm-r0 | 29918 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |

**HLP_VERIFY in fv_rocm-r0: ours against vLLM's kernel on the same step inputs, per bucket (both workers, all 16 layers)**

| bucket | lengths | compared calls | rel (Frobenius) min-max | max abs difference | largest |reference| | NaN |
|---|---|---|---|---|---|---|
| (1600,) | [(1568,)] | 32 | 1.52e-03-2.26e-03 | 0.5 | 101 | 0 |

## Job 23620886

- ## job 23620886 start 22:35:18 arms f_rocm s_rocm s_rocm f_rocm
- Result rows (timed cells of 6, client finished): f_rocm-r0 6 True, f_rocm-r1 6 True, s_rocm-r0 6 True, s_rocm-r1 6 True

**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)

| Cell (prompt tokens x streams) | f_rocm-r0 | f_rocm-r1 | s_rocm-r0 | s_rocm-r1 |
|---|---|---|---|---|
| 1419 x1 | 0.5031 [0.5028-0.5036] | 0.5021 [0.5016-0.5022] | 0.5306 [0.5259-0.5340] | 0.5338 [0.5288-0.5368] |
| 2859 x1 | 0.9757 [0.9754-1.0088] | 0.9753 [0.9747-1.0011] | 1.0368 [1.0351-1.0398] | 1.0368 [1.0356-1.0403] |
| 5679 x1 | 1.9253 [1.9244-1.9301] | 1.9254 [1.9249-2.1375] | 2.0616 [2.0584-2.0657] | 2.0628 [2.0601-2.0686] |
| 7719 x1 | 2.6242 [2.6214-2.6370] | 2.6243 [2.6241-2.6314] | 2.8104 [2.8057-2.8161] | 2.8119 [2.8062-2.8151] |
| 14201 x1 | 5.3897 [5.1314-5.5859] | 5.1414 [5.1394-5.1556] | 5.4656 [5.4530-5.4704] | 5.4641 [5.4524-5.4765] |
| 2859 x4 | 2.8362 [1.1991-4.2759] | 2.6279 [1.1971-3.8603] | 2.8193 [1.2828-4.1447] | 2.8587 [1.2817-4.2154] |

**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**

| Cell | s reps | f reps | f / s | s spread | f spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.5306, 0.5338 | 0.5031, 0.5021 | 0.9444 | 0.60% | 0.20% | plugin faster |
| 2859 x1 | 1.0368, 1.0368 | 0.9757, 0.9753 | 0.9409 | 0.00% | 0.04% | plugin faster |
| 5679 x1 | 2.0616, 2.0628 | 1.9253, 1.9254 | 0.9336 | 0.06% | 0.01% | plugin faster |
| 7719 x1 | 2.8104, 2.8119 | 2.6242, 2.6243 | 0.9335 | 0.05% | 0.00% | plugin faster |
| 14201 x1 | 5.4656, 5.4641 | 5.3897, 5.1414 | 0.9635 | 0.03% | 4.72% | ~ inside noise |
| 2859 x4 | 2.8193, 2.8587 | 2.8362, 2.6279 | 0.9623 | 1.39% | 7.62% | ~ inside noise |

**Plugin counters per worker (attention-layer calls; 16 per step)**

| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_rocm-r0 | 50796 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r0 | 50797 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 59540 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 59541 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |

## Job 23624337

- ## job 23624337 start 00:59:54 arms f_rocm s_rocm s_rocm f_rocm
- Result rows (timed cells of 6, client finished): f_rocm-r0 6 True, f_rocm-r1 6 True, s_rocm-r0 6 True, s_rocm-r1 6 True

**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)

| Cell (prompt tokens x streams) | f_rocm-r0 | f_rocm-r1 | s_rocm-r0 | s_rocm-r1 |
|---|---|---|---|---|
| 1419 x1 | 0.5042 [0.5030-0.5053] | 0.5139 [0.5054-0.5157] | 0.5047 [0.5037-0.5060] | 0.5123 [0.5093-0.5835] |
| 2859 x1 | 0.9776 [0.9769-0.9968] | 0.9925 [0.9820-1.0050] | 0.9782 [0.9772-0.9796] | 0.9895 [0.9878-0.9898] |
| 5679 x1 | 1.9291 [1.9277-2.2335] | 1.9595 [1.9591-1.9702] | 1.9301 [1.9294-1.9317] | 1.9542 [1.9522-1.9553] |
| 7719 x1 | 2.6274 [2.6266-2.6318] | 2.6702 [2.6676-2.7272] | 2.6289 [2.6268-2.6323] | 2.6621 [2.6609-2.6632] |
| 14201 x1 | 5.1371 [5.1353-5.1428] | 5.2268 [5.1897-5.2538] | 5.1443 [5.1433-5.1671] | 5.1847 [5.1378-5.1986] |
| 2859 x4 | 2.6535 [1.2009-3.9177] | 2.6663 [1.2182-3.9247] | 2.6652 [1.2024-3.9704] | 2.6607 [1.2148-3.9190] |

**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**

| Cell | s reps | f reps | f / s | s spread | f spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.5047, 0.5123 | 0.5042, 0.5139 | 1.0011 | 1.49% | 1.91% | ~ inside noise |
| 2859 x1 | 0.9782, 0.9895 | 0.9776, 0.9925 | 1.0012 | 1.15% | 1.51% | ~ inside noise |
| 5679 x1 | 1.9301, 1.9542 | 1.9291, 1.9595 | 1.0011 | 1.24% | 1.56% | ~ inside noise |
| 7719 x1 | 2.6289, 2.6621 | 2.6274, 2.6702 | 1.0012 | 1.25% | 1.62% | ~ inside noise |
| 14201 x1 | 5.1443, 5.1847 | 5.1371, 5.2268 | 1.0034 | 0.78% | 1.73% | ~ inside noise |
| 2859 x4 | 2.6652, 2.6607 | 2.6535, 2.6663 | 0.9989 | 0.17% | 0.48% | ~ inside noise |

**Plugin counters per worker (attention-layer calls; 16 per step)**

| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_rocm-r0 | 75095 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r0 | 75096 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 116426 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |
| f_rocm-r1 | 116427 | 800 | 50 | 1568 | 2.04% | 2080 | 480 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1600:800 |

## Job 23620887

- ## job 23620887 start 22:56:22 arms fv_rocm s_rocm f_rocm f_rocm s_rocm
- Result rows (timed cells of 6, client finished): f_rocm-r0 6 True, f_rocm-r1 6 True, fv_rocm-r0 6 True, s_rocm-r0 6 True, s_rocm-r1 6 True

**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)

| Cell (prompt tokens x streams) | f_rocm-r0 | f_rocm-r1 | fv_rocm-r0 | s_rocm-r0 | s_rocm-r1 |
|---|---|---|---|---|---|
| 1419 x1 | 0.5019 [0.5015-0.5023] | 0.5040 [0.5015-0.5047] | 0.5019 [0.5017-0.5022] | 0.5013 [0.5009-0.5018] | 0.5034 [0.5005-0.5036] |
| 2859 x1 | 0.9850 [0.9841-0.9869] | 0.9848 [0.9838-1.0310] | 0.9778 [0.9776-0.9978] | 0.9902 [0.9896-0.9909] | 0.9802 [0.9799-0.9803] |
| 5679 x1 | 1.8572 [1.8570-1.8581] | 1.8570 [1.8556-1.8581] | 1.8468 [1.8462-1.8479] | 1.8922 [1.8916-1.8929] | 1.8737 [1.8731-1.8745] |
| 7719 x1 | 2.4787 [2.4782-2.6219] | 2.4775 [2.4768-2.4787] | 2.4707 [2.4703-2.4711] | 2.5412 [2.5404-2.6045] | 2.5236 [2.5223-2.5248] |
| 14201 x1 | 4.5758 [4.5742-4.5767] | 4.5693 [4.5667-4.5708] | 4.5656 [4.5647-4.5662] | 4.7759 [4.7742-4.7788] | 4.7686 [4.7676-4.7691] |
| 2859 x4 | 3.7246 [0.9869-3.7261] | 3.7165 [0.9840-3.7589] | 3.7098 [0.9804-3.7118] | 3.7441 [0.9936-3.7455] | 3.7106 [0.9808-3.7155] |

**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**

| Cell | s reps | f reps | f / s | s spread | f spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.5013, 0.5034 | 0.5019, 0.5040 | 1.0012 | 0.42% | 0.42% | ~ inside noise |
| 2859 x1 | 0.9902, 0.9802 | 0.9850, 0.9848 | 0.9997 | 1.02% | 0.02% | ~ inside noise |
| 5679 x1 | 1.8922, 1.8737 | 1.8572, 1.8570 | 0.9863 | 0.98% | 0.01% | plugin faster |
| 7719 x1 | 2.5412, 2.5236 | 2.4787, 2.4775 | 0.9786 | 0.69% | 0.05% | plugin faster |
| 14201 x1 | 4.7759, 4.7686 | 4.5758, 4.5693 | 0.9582 | 0.15% | 0.14% | plugin faster |
| 2859 x4 | 3.7441, 3.7106 | 3.7246, 3.7165 | 0.9982 | 0.90% | 0.22% | ~ inside noise |

**Plugin counters per worker (attention-layer calls; 16 per step)**

| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_rocm-r0 | 35911 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r0 | 35912 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r1 | 55149 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r1 | 55150 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| fv_rocm-r0 | 28060 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| fv_rocm-r0 | 28061 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |

**HLP_VERIFY in fv_rocm-r0: ours against vLLM's kernel on the same step inputs, per bucket (both workers, all 16 layers)**

| bucket | lengths | compared calls | rel (Frobenius) min-max | max abs difference | largest |reference| | NaN |
|---|---|---|---|---|---|---|
| (2432,) | [(2352,)] | 32 | 1.52e-03-2.97e-03 | 0.5 | 102 | 0 |
| (5760,) | [(5488,)] | 32 | 1.52e-03-6.35e-03 | 0.75 | 106 | 0 |
| (7200,) | [(7056,)] | 32 | 1.52e-03-7.91e-03 | 0.75 | 106.5 | 0 |
| (14240,) | [(14112,)] | 32 | 1.57e-03-1.29e-02 | 1.5 | 103 | 0 |

## Job 23624336

- ## job 23624336 start 00:59:54 arms f_rocm s_rocm s_rocm f_rocm
- Result rows (timed cells of 6, client finished): f_rocm-r0 6 True, f_rocm-r1 6 True, s_rocm-r0 6 True, s_rocm-r1 6 True

**Time to first token, s: median of the cell's timed requests [min-max of them]** (8 requests per cell at 1 stream, 32 at 4)

| Cell (prompt tokens x streams) | f_rocm-r0 | f_rocm-r1 | s_rocm-r0 | s_rocm-r1 |
|---|---|---|---|---|
| 1419 x1 | 0.4853 [0.4850-0.4862] | 0.4852 [0.4850-0.4854] | 0.4849 [0.4847-0.4855] | 0.4883 [0.4855-0.4885] |
| 2859 x1 | 0.9377 [0.9375-0.9380] | 0.9377 [0.9374-0.9529] | 0.9397 [0.9391-0.9400] | 0.9404 [0.9399-0.9432] |
| 5679 x1 | 1.7620 [1.7614-1.7625] | 1.7631 [1.7627-1.7634] | 1.7894 [1.7887-1.7897] | 1.7895 [1.7887-1.7960] |
| 7719 x1 | 2.3693 [2.3596-2.3701] | 2.3595 [2.3591-2.3604] | 2.4114 [2.4111-2.4119] | 2.4103 [2.4099-2.4124] |
| 14201 x1 | 4.3555 [4.3547-4.3718] | 4.3535 [4.3530-4.3545] | 4.5595 [4.5587-4.5598] | 4.5582 [4.5538-4.5593] |
| 2859 x4 | 3.5443 [0.9398-3.6582] | 3.5422 [0.9391-3.7307] | 3.5488 [0.9424-3.5500] | 3.5467 [0.9406-3.5486] |

**f_rocm / s_rocm, ratio of the arms' medians over repetitions (below 1: plugin faster); spread = (max - min) / median of one arm's repetition medians; `~` when |ratio - 1| <= the larger spread**

| Cell | s reps | f reps | f / s | s spread | f spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4849, 0.4883 | 0.4853, 0.4852 | 0.9972 | 0.70% | 0.02% | ~ inside noise |
| 2859 x1 | 0.9397, 0.9404 | 0.9377, 0.9377 | 0.9975 | 0.07% | 0.00% | plugin faster |
| 5679 x1 | 1.7894, 1.7895 | 1.7620, 1.7631 | 0.9850 | 0.01% | 0.06% | plugin faster |
| 7719 x1 | 2.4114, 2.4103 | 2.3693, 2.3595 | 0.9807 | 0.05% | 0.41% | plugin faster |
| 14201 x1 | 4.5595, 4.5582 | 4.3555, 4.3535 | 0.9552 | 0.03% | 0.05% | plugin faster |
| 2859 x4 | 3.5488, 3.5467 | 3.5443, 3.5422 | 0.9987 | 0.06% | 0.06% | plugin faster |

**Plugin counters per worker (attention-layer calls; 16 per step)**

| Arm | worker | hit | steps taken (hit / 16) | tokens / hit | pad share | stock_context | stock_mixed | stock_short | stock_long | stock_miss | stock_guard | stock_shape | stock_capture | error | buckets used (bucket: calls) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f_rocm-r0 | 100142 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r0 | 100143 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r1 | 9186 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
| f_rocm-r1 | 9187 | 800 | 50 | 6272 | 2.24% | 0 | 160 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2432:320 5760:160 7200:160 14240:160 |
