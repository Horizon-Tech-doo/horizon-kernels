# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Calibrate horizon_kernels.nvfp4_linear (tilelang.adaptive) offline, e.g. after a driver or GPU change.
# New models do not need this: the vLLM plugin calibrates any uncalibrated layer shape at load (HORIZON_AUTO_CALIBRATE).
#   calibrate_nvfp4_linear.py [BUCKETS=1,2,4,...] [SHAPES=N1xK1,N2xK2,...]   (HORIZON_DROP_MARLIN=1 -> no-Marlin table)
import sys
from horizon_kernels.nvfp4_linear import make_op
from horizon_kernels.calibration import calibrate_shape

QWEN38 = {"gdn.in_proj_qkvz": (16384, 5120), "attn.qkv_proj": (14336, 5120), "o/out_proj": (5120, 6144),
          "mlp.gate_up": (34816, 5120), "mlp.down": (5120, 17408)}
BUCKETS = [int(b) for b in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1] else None
SHAPES = {s: tuple(int(v) for v in s.split("x")) for s in sys.argv[2].split(",")} if len(sys.argv) > 2 else QWEN38

op = make_op(allow_lossy=False)
for name, (N, K) in SHAPES.items():
    print(f"== {name} N={N} K={K}", flush=True)
    print(f"   ({calibrate_shape(op, N, K, BUCKETS)} weight copies)", flush=True)
print("table:", op.table_path)
