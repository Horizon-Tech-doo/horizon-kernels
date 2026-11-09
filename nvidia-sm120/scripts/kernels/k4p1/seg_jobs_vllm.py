# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Print the JSON job list (for seg_compile.py) of every segment kernel k4seg.py builds inside vLLM (M 1..32, both
# next-layer kinds, HORIZON_K4_HW / STAGES / PF_KT defaults). usage: python seg_jobs_vllm.py [hw] [stages] [pf_kt]
import json, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from tilelang.contrib.horizon_k4 import segment as sg
hw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
st = int(sys.argv[2]) if len(sys.argv) > 2 else 4
pk = int(sys.argv[3]) if len(sys.argv) > 3 else 0
jobs = set()
for M in range(1, 33):
    for nq in (sg.NQ, sg.NQ_ATTN):
        WM, splits = sg.bucket(M, nq)
        jobs.add((376, WM, tuple(splits), st, 0, pk, nq, hw))
print(json.dumps([[a, b, list(c), d, e, f, g, h] for (a, b, c, d, e, f, g, h) in sorted(jobs)]))
