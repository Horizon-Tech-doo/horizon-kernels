# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU-only (GPU hidden): JIT-compile the segment kernel variants into TILELANG_CACHE_DIR so the GPU job does not
# compile while holding the GPU lock. usage: python seg_precompile.py M[,M..] NCTA[,NCTA..] [stamps 0/1] [silu_mode]
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
from tilelang.contrib.horizon_k4 import segment as sg
Ms = [int(x) for x in sys.argv[1].split(",")]; nctas = [int(x) for x in sys.argv[2].split(",")]
st = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0").split(",")]
mode = int(sys.argv[4]) if len(sys.argv) > 4 else 1
done = set()
for M in Ms:
    WM, splits = sg.bucket(M)
    for n in nctas:
        for s in st:
            key = (n, WM, splits, s)
            if key in done:
                continue
            done.add(key)
            t0 = time.time()
            try:
                sg.segment_kernel(n, WM, *splits, sg.SK_CFG[0], sg.SK_CFG[1], 4, mode, 1e-6, 1, s)
                print(f"compiled M={M} ncta={n} stamps={s} in {time.time()-t0:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"FAIL M={M} ncta={n} stamps={s}: {type(e).__name__}: {str(e)[:400]}", flush=True)
print("PRECOMPILE_DONE")
