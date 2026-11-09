# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline lowering (no GPU, no CUDA context) of the ops-track kernels: writes the CUDA source, compiles it with nvcc
# the way TileLang does (+ --use_fast_math where the kernel asks for it) and prints ptxas registers / spills.
# usage: python ops_lower.py OUTDIR [case-substring]
import faulthandler, os, subprocess, sys, time, re
faulthandler.dump_traceback_later(1500, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.contrib.horizon_ops import gdn_decode as gd

out = sys.argv[1]
sel = sys.argv[2] if len(sys.argv) > 2 else ""
os.makedirs(out, exist_ok=True)
TG = Target({"kind": "cuda", "arch": "sm_120a"})
# production strides: mixed_qkv contiguous 10240, a/b views of ba (row 96), z view of qkvz (row 16384),
# state slot stride from the mamba page (checked at run time; any multiple of 4 works)
CASES = []
for vr in (32, 64, 128):
    CASES.append((f"gdn_vr{vr}", gd.gdn_decode_kernel, (48, 16, vr, 10240, 96, 96, 16384, 48 * 128 * 128 + 1024,
                                                         "float32", "float32", False), {}))
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
for name, jf, args, kw in CASES:
    if sel and sel not in name:
        continue
    t0 = time.time()
    try:
        tir = jf.get_tir(*args, **kw)
        with TG:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                art = tilelang.lower(tir, target=TG)
        src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"{name} LOWER-FAIL {type(e).__name__}: {str(e)[:1500]}", flush=True)
        continue
    cu = f"{out}/{name}.cu"; open(cu, "w").write(src)
    cub = cu[:-3] + ".cubin"
    r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w", "--use_fast_math",
                        "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cub, cu],
                       capture_output=True, text=True)
    regs = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "registers" in l or "spill" in l or "error" in l]
    print(f"{name}: lowered+compiled in {time.time() - t0:.1f} s rc={r.returncode}", flush=True)
    for l in regs[:6]:
        print("   ", l, flush=True)
print("OPS_LOWER_DONE", flush=True)
