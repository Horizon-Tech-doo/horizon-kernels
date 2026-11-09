# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU-only: lower + compile the K4 runtime self-test kernel (no GPU), print ptxas registers.
import os, re, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.contrib.horizon_k4 import selftest as st
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
TG = Target({"kind": "cuda", "arch": "sm_120a"})
for rd in (1, 0):
    tir = st.dag_kernel.get_tir(188, 8, rd)
    with TG:
        with tilelang.transform.PassContext(config=st.dag_kernel.pass_configs or {}):
            art = tilelang.lower(tir, target=TG)
    src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
    cu = f"{out}/dag_rd{rd}.cu"; open(cu, "w").write(src)
    r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w", "-I" + TILELANG_TEMPLATE_PATH,
                        "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cu[:-3] + ".cubin", cu], capture_output=True, text=True)
    print(f"rd={rd} rc={r.returncode} regs={re.findall(r'Used (\d+) registers', r.stderr)}", r.stderr[-1500:] if r.returncode else "", flush=True)
print("RT_LOWER_DONE")
