
## Job 23627000-nodefail

- ## job 23627000 start 03:32:44 arms g_tri s_tri cg_rocm g_rocm s_rocm s_rocm g_rocm cg_rocm s_tri g_tri MNBT=default HLG_MS= HLG_MIN=512 HLG_MAXPAD=0
- Run order and result rows (timed cells of 6, client finished; equivalence rows of 12): g_tri-r0 6 True 12, s_tri-r0 6 True 12, g_rocm-r0 6 True 12, g_rocm-r1 0 False 0, cg_rocm-r0 6 True 12, s_rocm-r1 6 True 12, s_rocm-r0 6 True 12

**Time to first token, s: median of the cell's timed requests [min-max]**

| Cell (prompt tokens x streams) | g_tri-r0 | s_tri-r0 | g_rocm-r0 | g_rocm-r1 | cg_rocm-r0 | s_rocm-r1 | s_rocm-r0 |
|---|---|---|---|---|---|---|---|
| 1419 x1 | 0.4482 [0.4477-0.5017] | 0.4576 [0.4573-0.4588] | 0.4457 [0.4450-0.4504] | - | 0.4446 [0.4445-0.4483] | 0.4555 [0.4549-0.4560] | 0.4554 [0.4550-0.4566] |
| 2859 x1 | 0.8811 [0.8806-0.9057] | 0.9083 [0.9077-0.9085] | 0.8638 [0.8630-0.8656] | - | 0.8609 [0.8599-0.8629] | 0.8905 [0.8901-0.8917] | 0.8900 [0.8889-0.8907] |
| 5679 x1 | 1.7798 [1.7782-1.7807] | 1.8388 [1.8375-1.8402] | 1.7054 [1.7050-1.7067] | - | 1.6831 [1.6815-1.6860] | 1.7672 [1.7662-1.8343] | 1.7661 [1.7652-1.7678] |
| 7719 x1 | 2.5008 [2.4991-2.5033] | 2.5762 [2.5742-2.5770] | 2.3232 [2.3213-2.3253] | - | 2.2881 [2.2868-2.2903] | 2.4021 [2.4010-2.4033] | 2.4019 [2.4003-2.4035] |
| 14201 x1 | 5.0861 [5.0762-5.1134] | 5.2382 [5.2357-5.2405] | 4.5668 [4.5644-4.5684] | - | 4.5224 [4.5212-4.5253] | 4.7167 [4.7154-4.7184] | 4.7215 [4.7180-4.7232] |
| 2859 x4 | 2.3787 [1.0814-3.4797] | 2.4004 [1.1092-3.4958] | 2.3602 [1.0671-3.4898] | - | 2.3556 [1.0644-3.4812] | 2.3812 [1.0861-3.5170] | 2.3896 [1.0924-3.5231] |

**Time to first token: cg_rocm / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm s | cg_rocm s | ratio | s_rocm spread | cg_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4555, 0.4554 | 0.4446 | 0.9762 | 0.02% | nan% | faster |
| 2859 x1 | 0.8905, 0.8900 | 0.8609 | 0.9670 | 0.06% | nan% | faster |
| 5679 x1 | 1.7672, 1.7661 | 1.6831 | 0.9527 | 0.06% | nan% | faster |
| 7719 x1 | 2.4021, 2.4019 | 2.2881 | 0.9526 | 0.01% | nan% | faster |
| 14201 x1 | 4.7167, 4.7215 | 4.5224 | 0.9583 | 0.10% | nan% | faster |
| 2859 x4 | 2.3812, 2.3896 | 2.3556 | 0.9875 | 0.35% | nan% | faster |

**Time to first token: g_rocm / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm s | g_rocm s | ratio | s_rocm spread | g_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4555, 0.4554 | 0.4457 | 0.9786 | 0.02% | nan% | faster |
| 2859 x1 | 0.8905, 0.8900 | 0.8638 | 0.9703 | 0.06% | nan% | faster |
| 5679 x1 | 1.7672, 1.7661 | 1.7054 | 0.9653 | 0.06% | nan% | faster |
| 7719 x1 | 2.4021, 2.4019 | 2.3232 | 0.9672 | 0.01% | nan% | faster |
| 14201 x1 | 4.7167, 4.7215 | 4.5668 | 0.9677 | 0.10% | nan% | faster |
| 2859 x4 | 2.3812, 2.3896 | 2.3602 | 0.9894 | 0.35% | nan% | faster |

**Time to first token: g_tri / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm s | g_tri s | ratio | s_rocm spread | g_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4555, 0.4554 | 0.4482 | 0.9841 | 0.02% | nan% | faster |
| 2859 x1 | 0.8905, 0.8900 | 0.8811 | 0.9897 | 0.06% | nan% | faster |
| 5679 x1 | 1.7672, 1.7661 | 1.7798 | 1.0074 | 0.06% | nan% | slower |
| 7719 x1 | 2.4021, 2.4019 | 2.5008 | 1.0411 | 0.01% | nan% | slower |
| 14201 x1 | 4.7167, 4.7215 | 5.0861 | 1.0778 | 0.10% | nan% | slower |
| 2859 x4 | 2.3812, 2.3896 | 2.3787 | 0.9972 | 0.35% | nan% | ~ inside noise |

**Time to first token: s_tri / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm s | s_tri s | ratio | s_rocm spread | s_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4555, 0.4554 | 0.4576 | 1.0047 | 0.02% | nan% | slower |
| 2859 x1 | 0.8905, 0.8900 | 0.9083 | 1.0203 | 0.06% | nan% | slower |
| 5679 x1 | 1.7672, 1.7661 | 1.8388 | 1.0408 | 0.06% | nan% | slower |
| 7719 x1 | 2.4021, 2.4019 | 2.5762 | 1.0725 | 0.01% | nan% | slower |
| 14201 x1 | 4.7167, 4.7215 | 5.2382 | 1.1100 | 0.10% | nan% | slower |
| 2859 x4 | 2.3812, 2.3896 | 2.4004 | 1.0063 | 0.35% | nan% | slower |

**Time to first token: cg_rocm / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri s | cg_rocm s | ratio | s_tri spread | cg_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4576 | 0.4446 | 0.9716 | nan% | nan% | ~ inside noise |
| 2859 x1 | 0.9083 | 0.8609 | 0.9478 | nan% | nan% | ~ inside noise |
| 5679 x1 | 1.8388 | 1.6831 | 0.9153 | nan% | nan% | ~ inside noise |
| 7719 x1 | 2.5762 | 2.2881 | 0.8882 | nan% | nan% | ~ inside noise |
| 14201 x1 | 5.2382 | 4.5224 | 0.8634 | nan% | nan% | ~ inside noise |
| 2859 x4 | 2.4004 | 2.3556 | 0.9813 | nan% | nan% | ~ inside noise |

**Time to first token: g_rocm / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri s | g_rocm s | ratio | s_tri spread | g_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4576 | 0.4457 | 0.9740 | nan% | nan% | ~ inside noise |
| 2859 x1 | 0.9083 | 0.8638 | 0.9510 | nan% | nan% | ~ inside noise |
| 5679 x1 | 1.8388 | 1.7054 | 0.9275 | nan% | nan% | ~ inside noise |
| 7719 x1 | 2.5762 | 2.3232 | 0.9018 | nan% | nan% | ~ inside noise |
| 14201 x1 | 5.2382 | 4.5668 | 0.8718 | nan% | nan% | ~ inside noise |
| 2859 x4 | 2.4004 | 2.3602 | 0.9833 | nan% | nan% | ~ inside noise |

**Time to first token: g_tri / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri s | g_tri s | ratio | s_tri spread | g_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 0.4576 | 0.4482 | 0.9795 | nan% | nan% | ~ inside noise |
| 2859 x1 | 0.9083 | 0.8811 | 0.9701 | nan% | nan% | ~ inside noise |
| 5679 x1 | 1.8388 | 1.7798 | 0.9679 | nan% | nan% | ~ inside noise |
| 7719 x1 | 2.5762 | 2.5008 | 0.9707 | nan% | nan% | ~ inside noise |
| 14201 x1 | 5.2382 | 5.0861 | 0.9710 | nan% | nan% | ~ inside noise |
| 2859 x4 | 2.4004 | 2.3787 | 0.9910 | nan% | nan% | ~ inside noise |

**Plugin counters per worker** (GEMM plugin: linear calls; prefill-attention plugin: attention-layer calls, 16 per step)

| Arm | plugin | worker | hit | rows or tokens per hit | padded rows / pad share | left to vLLM | errors | kernels or buckets used |
|---|---|---|---|---|---|---|---|---|
| g_tri-r0 | gemm | 25832 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| g_tri-r0 | gemm | 25833 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| g_rocm-r0 | gemm | 36635 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| g_rocm-r0 | gemm | 36636 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| cg_rocm-r0 | gemm | 33123 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| cg_rocm-r0 | gemm | 33124 | 53088 | 1409 | 43200 | rows 38304 (635:2560 663:2560 1291:2560 1292:2560 1293:2560 512:1024), shape 17136, capture 0 | 0 | 9 kernels: 1568x5120x3072:13632 1568x17408x5120:13632 1568x8192x5120:10224 784x5120x3072:3584 784x5120x8704:3584 1568x7168x5120:3408 784x8192x5120:2688 1600x8192x5120:1440 |
| cg_rocm-r0 | prefill attention | 33123 | 2832 | 514 | 0.00% | context 0, mixed 480, short 0, long 0, miss 0, guard 0, shape 0 | 0 | 1568:928 ctx3136:560 ctx4704:560 ctx6272:400 ctx7840:192 ctx9408:192 |
| cg_rocm-r0 | prefill attention | 33124 | 2832 | 514 | 0.00% | context 0, mixed 480, short 0, long 0, miss 0, guard 0, shape 0 | 0 | 1568:928 ctx3136:560 ctx4704:560 ctx6272:400 ctx7840:192 ctx9408:192 |

**Energy per timed cell [observed]: Cray accel0 energy counter (mean W of the four counters over all timed cells: 256, 97, 100, 97; GCDs visible to the sampler 8). J per cell (mean W)**

| Cell | g_tri-r0 | s_tri-r0 | g_rocm-r0 | g_rocm-r1 | cg_rocm-r0 | s_rocm-r1 | s_rocm-r0 |
|---|---|---|---|---|---|---|---|
| 1419 x1 | 2932 (315 W) | 3091 (331 W) | 3202 (336 W) | - | 3119 (328 W) | 2946 (306 W) | 2797 (291 W) |
| 2859 x1 | 4827 (379 W) | 4947 (382 W) | 4673 (356 W) | - | 4833 (369 W) | 5174 (387 W) | 4993 (374 W) |
| 5679 x1 | 8603 (432 W) | 8529 (418 W) | 8392 (411 W) | - | 8387 (415 W) | 8601 (410 W) | 8495 (407 W) |
| 7719 x1 | 11688 (455 W) | 11572 (440 W) | 11066 (430 W) | - | 10776 (424 W) | 11545 (438 W) | 11082 (421 W) |
| 14201 x1 | 22333 (481 W) | 23052 (484 W) | 20227 (451 W) | - | 20070 (451 W) | 20901 (454 W) | 21015 (456 W) |
| 2859 x4 | 15473 (459 W) | 15918 (468 W) | 15371 (446 W) | - | 15749 (458 W) | 15821 (456 W) | 16036 (462 W) |

**Energy per cell: cg_rocm / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm J | cg_rocm J | ratio | s_rocm spread | cg_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 2946, 2797 | 3119 | 1.0864 | 5.19% | nan% | more energy |
| 2859 x1 | 5174, 4993 | 4833 | 0.9507 | 3.55% | nan% | less energy |
| 5679 x1 | 8601, 8495 | 8387 | 0.9811 | 1.25% | nan% | less energy |
| 7719 x1 | 11545, 11082 | 10776 | 0.9525 | 4.09% | nan% | less energy |
| 14201 x1 | 20901, 21015 | 20070 | 0.9576 | 0.54% | nan% | less energy |
| 2859 x4 | 15821, 16036 | 15749 | 0.9887 | 1.35% | nan% | ~ inside noise |

**Energy per cell: g_rocm / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm J | g_rocm J | ratio | s_rocm spread | g_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 2946, 2797 | 3202 | 1.1152 | 5.19% | nan% | more energy |
| 2859 x1 | 5174, 4993 | 4673 | 0.9193 | 3.55% | nan% | less energy |
| 5679 x1 | 8601, 8495 | 8392 | 0.9818 | 1.25% | nan% | less energy |
| 7719 x1 | 11545, 11082 | 11066 | 0.9782 | 4.09% | nan% | ~ inside noise |
| 14201 x1 | 20901, 21015 | 20227 | 0.9651 | 0.54% | nan% | less energy |
| 2859 x4 | 15821, 16036 | 15371 | 0.9650 | 1.35% | nan% | less energy |

**Energy per cell: g_tri / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm J | g_tri J | ratio | s_rocm spread | g_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 2946, 2797 | 2932 | 1.0212 | 5.19% | nan% | ~ inside noise |
| 2859 x1 | 5174, 4993 | 4827 | 0.9495 | 3.55% | nan% | less energy |
| 5679 x1 | 8601, 8495 | 8603 | 1.0064 | 1.25% | nan% | ~ inside noise |
| 7719 x1 | 11545, 11082 | 11688 | 1.0331 | 4.09% | nan% | ~ inside noise |
| 14201 x1 | 20901, 21015 | 22333 | 1.0656 | 0.54% | nan% | more energy |
| 2859 x4 | 15821, 16036 | 15473 | 0.9714 | 1.35% | nan% | less energy |

**Energy per cell: s_tri / s_rocm** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_rocm J | s_tri J | ratio | s_rocm spread | s_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 2946, 2797 | 3091 | 1.0766 | 5.19% | nan% | more energy |
| 2859 x1 | 5174, 4993 | 4947 | 0.9731 | 3.55% | nan% | ~ inside noise |
| 5679 x1 | 8601, 8495 | 8529 | 0.9978 | 1.25% | nan% | ~ inside noise |
| 7719 x1 | 11545, 11082 | 11572 | 1.0229 | 4.09% | nan% | ~ inside noise |
| 14201 x1 | 20901, 21015 | 23052 | 1.0999 | 0.54% | nan% | more energy |
| 2859 x4 | 15821, 16036 | 15918 | 0.9994 | 1.35% | nan% | ~ inside noise |

**Energy per cell: cg_rocm / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri J | cg_rocm J | ratio | s_tri spread | cg_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 3091 | 3119 | 1.0091 | nan% | nan% | ~ inside noise |
| 2859 x1 | 4947 | 4833 | 0.9770 | nan% | nan% | ~ inside noise |
| 5679 x1 | 8529 | 8387 | 0.9832 | nan% | nan% | ~ inside noise |
| 7719 x1 | 11572 | 10776 | 0.9312 | nan% | nan% | ~ inside noise |
| 14201 x1 | 23052 | 20070 | 0.8706 | nan% | nan% | ~ inside noise |
| 2859 x4 | 15918 | 15749 | 0.9894 | nan% | nan% | ~ inside noise |

**Energy per cell: g_rocm / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri J | g_rocm J | ratio | s_tri spread | g_rocm spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 3091 | 3202 | 1.0359 | nan% | nan% | ~ inside noise |
| 2859 x1 | 4947 | 4673 | 0.9447 | nan% | nan% | ~ inside noise |
| 5679 x1 | 8529 | 8392 | 0.9839 | nan% | nan% | ~ inside noise |
| 7719 x1 | 11572 | 11066 | 0.9563 | nan% | nan% | ~ inside noise |
| 14201 x1 | 23052 | 20227 | 0.8775 | nan% | nan% | ~ inside noise |
| 2859 x4 | 15918 | 15371 | 0.9656 | nan% | nan% | ~ inside noise |

**Energy per cell: g_tri / s_tri** (ratio of the arms' medians over repetitions; spread = (max - min) / median of an arm's repetition values; a verdict needs |ratio - 1| above the larger spread)

| Cell | s_tri J | g_tri J | ratio | s_tri spread | g_tri spread | verdict |
|---|---|---|---|---|---|---|
| 1419 x1 | 3091 | 2932 | 0.9485 | nan% | nan% | ~ inside noise |
| 2859 x1 | 4947 | 4827 | 0.9758 | nan% | nan% | ~ inside noise |
| 5679 x1 | 8529 | 8603 | 1.0086 | nan% | nan% | ~ inside noise |
| 7719 x1 | 11572 | 11688 | 1.0100 | nan% | nan% | ~ inside noise |
| 14201 x1 | 23052 | 22333 | 0.9688 | nan% | nan% | ~ inside noise |
| 2859 x4 | 15918 | 15473 | 0.9721 | nan% | nan% | ~ inside noise |

**Equivalence set [observed]: 12 fixed greedy prompts x 48 tokens. Identical token sequences of 12; largest |difference| of the chosen token's logprob over the tokens before the first difference (nats); differing prompts as (kind, index, first differing token, top-2 margin of the reference there, change of the reference token's logprob there in this run)**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_tri-r0 | s_rocm-r1 | 11 of 12 | 0.5674 | ('short', 1, 17, 0.125, -0.063) |
| s_tri-r0 | s_rocm-r1 | 8 of 12 | 0.2493 | ('short', 1, 41, 0.0, -0.095), ('short', 3, 20, 0.125, -0.031), ('2.9k', 1, 34, 0.0, -0.065), ('7.7k', 0, 35, 2.125, -0.781) |
| g_rocm-r0 | s_rocm-r1 | 9 of 12 | 0.3298 | ('2.9k', 1, 32, 0.25, -0.123), ('7.7k', 1, 19, 0.125, -0.083), ('14.2k', 0, 35, 0.125, -0.125) |
| cg_rocm-r0 | s_rocm-r1 | 10 of 12 | 0.3391 | ('2.9k', 1, 34, 0.0, -0.065), ('14.2k', 0, 35, 0.125, -0.755) |
| s_rocm-r0 | s_rocm-r1 | 12 of 12 | 0.0000 | - |
| g_tri-r0 | s_tri-r0 | 8 of 12 | 0.1487 | ('short', 1, 17, 0.125, -0.064), ('short', 3, 20, 0.0, -0.1), ('2.9k', 1, 34, 0.125, -0.061), ('7.7k', 0, 35, 0.25, -0.162) |
| g_rocm-r0 | s_tri-r0 | 6 of 12 | 0.1825 | ('short', 1, 41, 0.125, -0.03), ('short', 3, 20, 0.0, -0.094), ('2.9k', 1, 32, 0.125, -0.057), ('7.7k', 0, 35, 0.25, -0.674), ('7.7k', 1, 19, 0.125, -0.049), ('14.2k', 0, 35, 0.25, -0.183) |
| cg_rocm-r0 | s_tri-r0 | 8 of 12 | 0.1568 | ('short', 1, 41, 0.125, -0.03), ('short', 3, 20, 0.0, -0.094), ('7.7k', 0, 35, 0.25, -0.558), ('14.2k', 0, 35, 0.25, -0.813) |
