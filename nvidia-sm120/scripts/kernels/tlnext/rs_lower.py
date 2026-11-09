# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline (no GPU, no lock) lowering of prefill_rs.nvfp4_gemm_rs_kernel -> CUDA source + ptxas -v (registers, spills).
#   rs_lower.py N K passes conv
import faulthandler, os, subprocess, sys, time
faulthandler.dump_traceback_later(900, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
import tilelang.language as T
from tvm.target import Target
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs
N, K, passes, conv = (int(v) for v in sys.argv[1:5])
extra = eval(sys.argv[5]) if len(sys.argv) > 5 else {}
TG = Target({"kind": "cuda", "arch": "sm_120a"})
t0 = time.time()
jf = prs.nvfp4_gemm_rs_kernel
tir = jf.get_tir(N, K, passes, T.bfloat16, conv, **extra)
with TG:
    with tilelang.transform.PassContext(config=jf.pass_configs or {}):
        art = tilelang.lower(tir, target=TG)
src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
od = os.path.expanduser("$HK_WORK/tlnext/offline"); os.makedirs(od, exist_ok=True)
cu = f"{od}/rs_{N}_{K}_{passes}_{conv}.cu"; open(cu, "w").write(src)
print("lowered", round(time.time() - t0, 1), cu, flush=True)
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
cub = cu[:-3] + ".cubin"
r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w", "-I" + TILELANG_TEMPLATE_PATH,
                    "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cub, cu], capture_output=True, text=True)
print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "Used" in l or "spill" in l or "rror" in l)[:3000], flush=True)
