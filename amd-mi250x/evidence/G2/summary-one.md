
**Time to first token, one-step prefill (jobs 23626997 / 23626998): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fg_rocm / s_rocm | 0.983 faster / 0.975 faster | 0.925 faster / 0.921 faster | 0.921 faster / 0.923 faster | 0.904 faster / 0.905 faster | 0.872 faster / 0.870 faster | 0.980 faster / 0.977 faster | 0.66%, 0.57% |
| fg_tri / s_rocm | 0.989 faster / 0.985 faster | 0.929 faster / 0.928 faster | 0.913 faster / 0.914 faster | 0.917 faster / 0.919 faster | 0.863 faster / 0.861 faster | 0.994 faster / 0.995 faster | 0.54%, 0.57% |
| g_rocm / s_rocm | 0.981 faster / 0.976 faster | 0.929 faster / 0.925 faster | 0.938 faster / 0.937 faster | 0.929 faster / 0.928 faster | 0.919 faster / 0.918 faster | 0.982 faster / 0.979 faster | 0.54%, 0.57% |
| s_tri / s_rocm | 1.009 slower / 1.003 ~ | 1.026 slower / 1.025 slower | 1.056 slower / 1.053 slower | 1.091 slower / 1.090 slower | 1.160 slower / 1.157 slower | 1.021 slower / 1.020 slower | 0.53%, 0.57% |
| fg_rocm / s_tri | 0.974 faster / 0.972 faster | 0.901 faster / 0.898 faster | 0.872 faster / 0.876 faster | 0.828 faster / 0.831 faster | 0.751 faster / 0.752 faster | 0.959 faster / 0.958 faster | 0.66%, 0.53% |
| fg_tri / s_tri | 0.980 faster / 0.982 faster | 0.905 faster / 0.905 faster | 0.865 faster / 0.868 faster | 0.841 faster / 0.843 faster | 0.744 faster / 0.744 faster | 0.974 faster / 0.975 faster | 0.54%, 0.53% |
| g_rocm / s_tri | 0.972 faster / 0.973 faster | 0.905 faster / 0.902 faster | 0.887 faster / 0.890 faster | 0.851 faster / 0.851 faster | 0.792 faster / 0.794 faster | 0.961 faster / 0.959 faster | 0.54%, 0.53% |

**Card energy per prefill cell, one-step prefill (jobs 23626997 / 23626998): ratio (verdict) in job A / job B; `~` = inside the arms' spread**

| Comparison | 1419 x1 | 2859 x1 | 5679 x1 | 7719 x1 | 14201 x1 | 2859 x4 | largest spread A, B |
|---|---|---|---|---|---|---|---|
| fg_rocm / s_rocm | 1.006 more / 0.995 ~ | 0.916 less / 0.923 less | 0.971 less / 0.964 less | 0.890 less / 0.917 less | 0.871 less / 0.885 less | 0.977 less / 0.968 less | 5.56%, 10.85% |
| fg_tri / s_rocm | 1.036 more / 0.985 ~ | 0.907 less / 0.927 less | 0.925 less / 0.927 less | 0.931 less / 0.906 less | 0.835 less / 0.857 less | 0.981 less / 0.978 less | 5.56%, 6.80% |
| g_rocm / s_rocm | 1.017 more / 1.008 ~ | 0.920 less / 0.908 less | 0.963 less / 0.965 ~ | 0.943 less / 0.949 less | 0.916 less / 0.925 less | 0.945 less / 0.985 ~ | 5.56%, 6.80% |
| s_tri / s_rocm | 0.988 ~ / 1.054 ~ | 0.998 ~ / 0.967 less | 1.032 more / 1.055 more | 1.067 more / 1.060 more | 1.124 more / 1.131 more | 0.988 less / 1.001 ~ | 5.56%, 6.80% |
| fg_rocm / s_tri | 1.018 ~ / 0.944 ~ | 0.918 less / 0.954 less | 0.941 less / 0.914 less | 0.834 less / 0.865 less | 0.775 less / 0.782 less | 0.988 ~ / 0.967 less | 4.35%, 10.85% |
| fg_tri / s_tri | 1.048 more / 0.935 less | 0.909 less / 0.959 ~ | 0.897 less / 0.878 less | 0.873 less / 0.855 less | 0.743 less / 0.758 less | 0.992 ~ / 0.977 less | 5.31%, 5.17% |
| g_rocm / s_tri | 1.029 more / 0.957 less | 0.922 less / 0.939 less | 0.933 less / 0.914 less | 0.884 less / 0.895 less | 0.815 less / 0.817 less | 0.956 less / 0.984 less | 4.68%, 5.04% |

**Equivalence set, one-step prefill (jobs 23626997 / 23626998), job A**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| g_rocm-r0 | s_rocm-r0 | 8 of 12 | 0.1939 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.074), ('14.2k', 0, 14, 0.125, -0.279), ('14.2k', 1, 12, 0.125, -0.125) |
| fg_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1614 | ('short', 1, 17, 0.0, -0.062), ('14.2k', 0, 14, 0.125, -0.424) |
| s_tri-r0 | s_rocm-r0 | 10 of 12 | 0.1864 | ('short', 1, 36, 0.125, 0.045), ('14.2k', 0, 14, 0.125, -0.533) |
| fg_tri-r0 | s_rocm-r0 | 9 of 12 | 0.1760 | ('short', 1, 36, 0.125, 0.045), ('2.9k', 1, 32, 0.0, -0.055), ('14.2k', 0, 14, 0.125, -0.283) |
| fg_tri-r1 | s_rocm-r0 | 9 of 12 | 0.1760 | ('short', 1, 36, 0.125, 0.045), ('2.9k', 1, 32, 0.0, -0.055), ('14.2k', 0, 14, 0.125, -0.283) |
| s_tri-r1 | s_rocm-r0 | 10 of 12 | 0.1864 | ('short', 1, 36, 0.125, 0.045), ('14.2k', 0, 14, 0.125, -0.533) |
| fg_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.1614 | ('short', 1, 17, 0.0, -0.062), ('14.2k', 0, 14, 0.125, -0.424) |
| g_rocm-r1 | s_rocm-r0 | 8 of 12 | 0.1939 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.074), ('14.2k', 0, 14, 0.125, -0.279), ('14.2k', 1, 12, 0.125, -0.125) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| g_rocm-r0 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |
| fg_rocm-r0 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| fg_tri-r0 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |
| fg_tri-r1 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |
| fg_rocm-r1 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| g_rocm-r1 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |


**Equivalence set, one-step prefill (jobs 23626997 / 23626998), job B**

| Run | against | identical | max abs logprob difference | differing prompts |
|---|---|---|---|---|
| fg_tri-r0 | s_rocm-r0 | 10 of 12 | 0.1983 | ('short', 1, 17, 0.125, -0.063), ('14.2k', 0, 35, 1.75, -1.098) |
| s_tri-r0 | s_rocm-r0 | 10 of 12 | 0.2130 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.125, -0.052) |
| fg_rocm-r0 | s_rocm-r0 | 11 of 12 | 0.3686 | ('2.9k', 1, 32, 0.125, -0.07) |
| g_rocm-r0 | s_rocm-r0 | 10 of 12 | 0.1946 | ('14.2k', 0, 35, 1.75, -0.844), ('14.2k', 1, 12, 0.0, -0.065) |
| s_rocm-r1 | s_rocm-r0 | 12 of 12 | 0.0000 | - |
| g_rocm-r1 | s_rocm-r0 | 10 of 12 | 0.1946 | ('14.2k', 0, 35, 1.75, -0.844), ('14.2k', 1, 12, 0.0, -0.065) |
| fg_rocm-r1 | s_rocm-r0 | 11 of 12 | 0.3686 | ('2.9k', 1, 32, 0.125, -0.07) |
| s_tri-r1 | s_rocm-r0 | 10 of 12 | 0.2130 | ('short', 1, 17, 0.125, -0.063), ('2.9k', 1, 32, 0.125, -0.052) |
| fg_tri-r1 | s_rocm-r0 | 10 of 12 | 0.1983 | ('short', 1, 17, 0.125, -0.063), ('14.2k', 0, 35, 1.75, -1.098) |
| fg_tri-r0 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |
| fg_rocm-r0 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| g_rocm-r0 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |
| g_rocm-r1 | s_tri-r0 | 8 of 12 | 0.2823 | ('short', 1, 17, 0.0, -0.062), ('2.9k', 1, 32, 0.0, -0.092), ('14.2k', 0, 35, 1.0, -0.631), ('14.2k', 1, 12, 0.25, -0.182) |
| fg_rocm-r1 | s_tri-r0 | 11 of 12 | 0.2242 | ('short', 1, 17, 0.0, -0.062) |
| s_tri-r1 | s_tri-r0 | 12 of 12 | 0.0000 | - |
| fg_tri-r1 | s_tri-r0 | 10 of 12 | 0.1864 | ('2.9k', 1, 32, 0.0, -0.073), ('14.2k', 0, 35, 1.0, -0.885) |

