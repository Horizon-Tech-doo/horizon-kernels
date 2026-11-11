
**Time to first token, one-step on the scaled prefill kernel (jobs 23627538 rerun / 23627539): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fgp_rocm / s_rocm | 0.977 faster / 0.978 faster | 0.925 faster / 0.923 faster | 0.921 faster / 0.923 faster | 0.904 faster / 0.906 faster | 0.873 faster / 0.872 faster | 0.979 faster / 0.979 faster | 0.53%, 0.79% |

**Card energy per prefill cell, one-step on the scaled prefill kernel (jobs 23627538 rerun / 23627539): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fgp_rocm / s_rocm | 0.995 ~ / 0.894 ~ | 0.951 less / 0.950 ~ | 0.932 less / 0.953 less | 0.935 less / 0.918 less | 0.884 less / 0.890 less | 0.965 ~ / 0.968 ~ | 4.43%, 12.26% |

**Equivalence set, one-step on the scaled prefill kernel (jobs 23627538 rerun / 23627539), job A**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| fgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1989 | ('2.9k', 1, 32, 0.125, -0.066), ('14.2k', 0, 35, 1.75, -0.991) |
| fgp_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.1989 | ('2.9k', 1, 32, 0.125, -0.066), ('14.2k', 0, 35, 1.75, -0.991) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |


**Equivalence set, one-step on the scaled prefill kernel (jobs 23627538 rerun / 23627539), job B**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| fgp_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1989 | ('2.9k', 1, 32, 0.125, -0.066), ('14.2k', 0, 35, 1.75, -0.991) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| fgp_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.1989 | ('2.9k', 1, 32, 0.125, -0.066), ('14.2k', 0, 35, 1.75, -0.991) |

