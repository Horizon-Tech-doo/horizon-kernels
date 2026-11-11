
**Time to first token, vLLM defaults (jobs 23626999 / 23627000 partial): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cg_rocm / s_rocm | 0.979 faster / 0.976 faster | 0.970 faster / 0.967 faster | 0.956 faster / 0.953 faster | 0.955 faster / 0.953 faster | 0.961 faster / 0.958 faster | 0.987 faster / 0.988 faster | 0.26%, 0.35% |
| g_rocm / s_rocm | 0.980 faster / 0.979 faster | 0.969 faster / 0.970 faster | 0.965 faster / 0.965 faster | 0.967 faster / 0.967 faster | 0.969 faster / 0.968 faster | 0.989 faster / 0.989 faster | 0.66%, 0.35% |
| g_tri / s_rocm | 0.986 faster / 0.984 faster | 0.989 faster / 0.990 faster | 1.007 slower / 1.007 slower | 1.039 slower / 1.041 slower | 1.075 slower / 1.078 slower | 0.987 faster / 0.997 ~ | 0.63%, 0.35% |
| s_tri / s_rocm | 1.003 slower / 1.005 slower | 1.018 slower / 1.020 slower | 1.041 slower / 1.041 slower | 1.073 slower / 1.073 slower | 1.108 slower / 1.110 slower | 1.000 ~ / 1.006 slower | 0.27%, 0.35% |
| cg_rocm / s_tri | 0.976 faster / 0.972 ~ | 0.954 faster / 0.948 ~ | 0.918 faster / 0.915 ~ | 0.890 faster / 0.888 ~ | 0.868 faster / 0.863 ~ | 0.987 faster / 0.981 ~ | 0.27%, 0.00% |
| g_rocm / s_tri | 0.977 faster / 0.974 ~ | 0.953 faster / 0.951 ~ | 0.927 faster / 0.927 ~ | 0.901 faster / 0.902 ~ | 0.874 faster / 0.872 ~ | 0.988 faster / 0.983 ~ | 0.66%, 0.00% |
| g_tri / s_tri | 0.983 faster / 0.980 ~ | 0.972 faster / 0.970 ~ | 0.967 faster / 0.968 ~ | 0.969 faster / 0.971 ~ | 0.970 faster / 0.971 ~ | 0.987 faster / 0.991 ~ | 0.63%, 0.00% |

**Card energy per prefill cell, vLLM defaults (jobs 23626999 / 23627000 partial): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| cg_rocm / s_rocm | 0.918 ~ / 1.086 more | 0.916 less / 0.951 less | 0.968 ~ / 0.981 less | 0.977 less / 0.953 less | 0.968 less / 0.958 less | 0.982 ~ / 0.989 ~ | 13.50%, 5.19% |
| g_rocm / s_rocm | 0.970 ~ / 1.115 more | 0.923 less / 0.919 less | 0.984 ~ / 0.982 less | 1.009 ~ / 0.978 ~ | 0.966 less / 0.965 less | 0.981 less / 0.965 less | 13.50%, 5.19% |
| g_tri / s_rocm | 0.941 ~ / 1.021 ~ | 0.891 less / 0.950 less | 0.996 ~ / 1.006 ~ | 1.015 ~ / 1.033 ~ | 1.063 more / 1.066 more | 0.975 less / 0.971 less | 13.50%, 5.19% |
| s_tri / s_rocm | 0.947 ~ / 1.077 more | 0.914 less / 0.973 ~ | 1.037 ~ / 0.998 ~ | 1.060 more / 1.023 ~ | 1.091 more / 1.100 more | 0.980 ~ / 0.999 ~ | 13.50%, 5.19% |
| cg_rocm / s_tri | 0.970 ~ / 1.009 ~ | 1.002 ~ / 0.977 ~ | 0.933 less / 0.983 ~ | 0.922 less / 0.931 ~ | 0.887 less / 0.871 ~ | 1.002 ~ / 0.989 ~ | 11.25%, 0.00% |
| g_rocm / s_tri | 1.024 ~ / 1.036 ~ | 1.009 ~ / 0.945 ~ | 0.949 ~ / 0.984 ~ | 0.952 ~ / 0.956 ~ | 0.885 less / 0.877 ~ | 1.001 ~ / 0.966 ~ | 11.25%, 0.00% |
| g_tri / s_tri | 0.994 ~ / 0.949 ~ | 0.975 ~ / 0.976 ~ | 0.961 less / 1.009 ~ | 0.957 less / 1.010 ~ | 0.974 less / 0.969 ~ | 0.995 ~ / 0.972 ~ | 11.25%, 0.00% |

**Equivalence set, vLLM defaults (jobs 23626999 / 23627000 partial), job A**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_rocm-r0 | s_rocm-r0 | 9 of 12 | 0.3298 | ('2.9k', 1, 32, 0.25, -0.123), ('7.7k', 1, 19, 0.125, -0.083), ('14.2k', 0, 35, 0.125, -0.125) |
| cg_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.3391 | ('2.9k', 1, 34, 0.0, -0.065), ('14.2k', 0, 35, 0.125, -0.755) |
| s_tri-r0 | s_rocm-r0 | 9 of 12 | 0.2153 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -0.843) |
| g_tri-r0 | s_rocm-r0 | 9 of 12 | 0.3734 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.123), ('14.2k', 0, 35, 0.125, -0.749) |
| g_tri-r1 | s_rocm-r0 | 9 of 12 | 0.3734 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.123), ('14.2k', 0, 35, 0.125, -0.749) |
| s_tri-r1 | s_rocm-r0 | 9 of 12 | 0.2153 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.25, -0.118), ('14.2k', 0, 35, 0.125, -0.843) |
| cg_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.3391 | ('2.9k', 1, 34, 0.0, -0.065), ('14.2k', 0, 35, 0.125, -0.755) |
| g_rocm-r1 | s_rocm-r0 | 9 of 12 | 0.3298 | ('2.9k', 1, 32, 0.25, -0.123), ('7.7k', 1, 19, 0.125, -0.083), ('14.2k', 0, 35, 0.125, -0.125) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| g_rocm-r0 | s_tri-r0 | 10 of 12 | 0.4068 | ('short', 1, 17, 0.0, -0.062), ('7.7k', 1, 19, 0.25, -0.127) |
| cg_rocm-r0 | s_tri-r0 | 10 of 12 | 0.2086 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.059) |
| g_tri-r0 | s_tri-r0 | 12 of 12 | 0.2868 | - |
| g_tri-r1 | s_tri-r0 | 12 of 12 | 0.2868 | - |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |
| cg_rocm-r1 | s_tri-r0 | 10 of 12 | 0.2086 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.059) |
| g_rocm-r1 | s_tri-r0 | 10 of 12 | 0.4068 | ('short', 1, 17, 0.0, -0.062), ('7.7k', 1, 19, 0.25, -0.127) |


**Equivalence set, vLLM defaults (jobs 23626999 / 23627000 partial), job B**

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

