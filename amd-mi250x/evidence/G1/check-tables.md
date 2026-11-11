# G1 standalone checks: the plugin's own path against vLLM's dense-linear path, one GCD, two arm orders [observed]

Ratio = vLLM's path / ours in time (above 1: ours faster), forward job / reversed job. Verdict per cell: LEAD when the ratio exceeds 1 + noise in both jobs, LOSS when below 1 - noise in both, else TIE; noise = that cell's |stock_b / stock - 1| in that job, floored at 0.5%. Error against an fp32 product equals stock's in every cell (ratio of errors at most 1.0000002); output repeats bitwise; all six edge cases pass in all four jobs.

## One-step prefill row counts, 224x256 xs4 rows form (plugin version 1 forward job 23624793, version 2 reversed job 23626106)

| M | 17408x5120 | 5120x8704 | 8192x5120 | 7168x5120 | 5120x3072 |
|---|---|---|---|---|---|
| 1568 | 0.968 / 0.956 LOSS | 0.736 / 0.732 LOSS | 0.848 / 0.848 LOSS | 1.011 / 1.003 TIE | 0.777 / 0.767 LOSS |
| 2352 | 1.130 / 1.130 LEAD | 1.169 / 1.163 LEAD | 0.893 / 0.890 LOSS | 1.101 / 1.102 LEAD | 1.208 / 1.206 LEAD |
| 3136 | 1.054 / 1.042 LEAD | 0.948 / 0.941 LOSS | 0.967 / 0.970 LOSS | 0.995 / 0.985 LOSS | 0.986 / 0.978 LOSS |
| 3920 | 1.085 / 1.087 LEAD | 0.980 / 0.980 LOSS | 0.960 / 0.950 LOSS | 1.068 / 1.069 LEAD | 1.020 / 1.014 LEAD |
| 4704 | 1.160 / 1.159 LEAD | 1.119 / 1.116 LEAD | 0.977 / 0.970 LOSS | 1.065 / 1.062 LEAD | 1.152 / 1.144 LEAD |
| 5488 | 1.119 / 1.121 LEAD | 1.046 / 1.047 LEAD | 1.064 / 1.062 LEAD | 1.004 / 0.993 TIE | 1.083 / 1.075 LEAD |
| 6272 | 1.100 / 1.101 LEAD | 0.939 / 0.933 LOSS | 1.047 / 1.050 LEAD | 1.045 / 1.046 LEAD | 0.980 / 0.966 LOSS |
| 7056 | 1.128 / 1.130 LEAD | 1.103 / 1.103 LEAD | 1.067 / 1.068 LEAD | 1.061 / 1.059 LEAD | 1.130 / 1.125 LEAD |
| 7840 | 1.121 / 1.122 LEAD | 1.040 / 1.039 LEAD | 0.994 / 0.981 LOSS | 1.126 / 1.124 LEAD | 1.070 / 1.062 LEAD |
| 8624 | 1.103 / 1.105 LEAD | 0.998 / 1.002 TIE | 1.073 / 1.073 LEAD | 1.150 / 1.150 LEAD | 1.026 / 1.026 LEAD |
| 9408 | 1.123 / 1.122 LEAD | 1.106 / 1.106 LEAD | 1.074 / 1.074 LEAD | 1.103 / 1.100 LEAD | 1.127 / 1.122 LEAD |
| 10192 | 1.110 / 1.113 LEAD | 1.054 / 1.053 LEAD | 1.100 / 1.100 LEAD | 1.042 / 1.026 LEAD | 1.080 / 1.070 LEAD |
| 10976 | 1.107 / 1.111 LEAD | 1.125 / 1.125 LEAD | 1.096 / 1.100 LEAD | 1.105 / 1.105 LEAD | 1.146 / 1.141 LEAD |
| 11760 | 1.109 / 1.112 LEAD | 1.095 / 1.094 LEAD | 1.083 / 1.083 LEAD | 1.099 / 1.101 LEAD | 1.121 / 1.117 LEAD |
| 12544 | 1.109 / 1.112 LEAD | 0.986 / 0.972 LOSS | 1.081 / 1.084 LEAD | 1.065 / 1.072 LEAD | 1.010 / 0.992 TIE |
| 13328 | 1.108 / 1.108 LEAD | 1.146 / 1.148 LEAD | 1.092 / 1.099 LEAD | 1.085 / 1.089 LEAD | 1.170 / 1.161 LEAD |
| 14112 | 1.138 / 1.136 LEAD | 1.093 / 1.093 LEAD | 1.100 / 1.104 LEAD | 1.088 / 1.088 LEAD | 1.111 / 1.105 LEAD |
| 14896 | 1.113 / 1.113 LEAD | 1.079 / 1.079 LEAD | 1.109 / 1.111 LEAD | 1.096 / 1.098 LEAD | 1.101 / 1.096 LEAD |
| 15680 | 1.109 / 1.113 LEAD | 1.026 / 1.018 LEAD | 1.111 / 1.114 LEAD | 1.126 / 1.128 LEAD | 1.042 / 1.033 LEAD |

Cells: LEAD 75, TIE 4, LOSS 16.

## vLLM's default chunk, per-shape tiles of the sweep (plugin version 2; forward job 23625419, reversed job 23626105); `stock` = left to vLLM

| M | 17408x5120 | 5120x8704 | 8192x5120 | 7168x5120 | 5120x3072 |
|---|---|---|---|---|---|
| 784 | stock | 1.146 / 1.145 LEAD | 1.041 / 1.017 LEAD | 1.053 / 1.044 LEAD | stock |
| 1568 | 1.075 / 1.071 LEAD | stock | 1.131 / 1.136 LEAD | 1.032 / 1.027 LEAD | 1.029 / 1.020 LEAD |

Cells: LEAD 7, TIE 0, LOSS 0.

