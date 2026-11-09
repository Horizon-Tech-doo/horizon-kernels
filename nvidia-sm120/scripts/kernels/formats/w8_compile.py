# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Compile decode_w8a16 kernels WITHOUT touching the GPU (no CUDA context), dump source / SASS stats.
#   TVM_FFI_DISABLE_TORCH_C_DLPACK=1 python w8_compile.py NxK[,NxK] M[,M...] [OUT_DIR]
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang  # noqa: E402,F401
from tvm.target import Target  # noqa: E402
from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8  # noqa: E402

shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
ms = [int(m) for m in sys.argv[2].split(",")]
out = sys.argv[3] if len(sys.argv) > 3 else None
extra = [eval(c) for c in os.environ.get("CFGS", "").split(";") if c.strip()]
err = 0
with Target({"kind": "cuda", "arch": "sm_120a"}):
    for N, K in shapes:
        cfgs = [(M, w8.config(N, K, M)) for M in ms] + [(0, c) for c in extra]
        seen = set()
        for M, cfg in cfgs:
            if cfg in seen:
                continue
            seen.add(cfg)
            t = time.time()
            try:
                k = w8.fp8_w8a16_kernel(N, K, *cfg)
                src = k.get_kernel_source()
                print(f"OK {N}x{K} M={M} cfg={cfg} smem={w8.smem_bytes(cfg)} src={len(src)} {time.time() - t:.0f}s",
                      flush=True)
                if out:
                    os.makedirs(out, exist_ok=True)
                    with open(os.path.join(out, f"w8_{N}x{K}_{'_'.join(map(str, cfg))}.cu"), "w") as f:
                        f.write(src)
            except Exception as ex:  # noqa: BLE001
                err += 1
                print(f"ERR {N}x{K} M={M} cfg={cfg}: {type(ex).__name__}: {str(ex)[-1500:]}", flush=True)
print("W8_COMPILE errors=%d" % err)
sys.exit(1 if err else 0)
