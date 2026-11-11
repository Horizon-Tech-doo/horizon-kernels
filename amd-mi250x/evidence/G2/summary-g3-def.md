
**Time to first token, vLLM defaults, scaled prefill kernel (jobs 23627540 / 23627541): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cgp_rocm / s_rocm | 0.984 faster / 0.981 faster | 0.974 faster / 0.969 faster | 0.959 faster / 0.954 faster | 0.958 faster / 0.952 faster | 0.964 faster / 0.958 faster | 0.991 faster / 0.986 faster | 0.53%, 0.50% |

**Card energy per prefill cell, vLLM defaults, scaled prefill kernel (jobs 23627540 / 23627541): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cgp_rocm / s_rocm | 0.970 ~ / 1.001 ~ | 1.048 ~ / 1.047 more | 0.956 ~ / 0.907 less | 0.985 less / 0.982 less | 0.964 ~ / 0.982 ~ | 0.994 ~ / 0.983 ~ | 7.85%, 7.35% |

**Equivalence set, vLLM defaults, scaled prefill kernel (jobs 23627540 / 23627541), job A**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| cgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.3300 | ('short', 0, 17, 0.125, -0.061), ('2.9k', 1, 32, 0.0, -0.055) |
| cgp_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.3300 | ('short', 0, 17, 0.125, -0.061), ('2.9k', 1, 32, 0.0, -0.055) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |


**Equivalence set, vLLM defaults, scaled prefill kernel (jobs 23627540 / 23627541), job B**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| cgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.2606 | ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -1.168) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| cgp_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.2606 | ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -1.168) |

