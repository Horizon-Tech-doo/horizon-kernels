# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline lower of a K5 TileLang kernel (no CUDA context, GPU not needed) -> CUDA source + nvcc -Xptxas -v.
# usage: k5_lower.py MODULE_PATH FUNC "args" ["kwargs"]
import faulthandler, os, subprocess, sys, time, importlib.util as u
faulthandler.dump_traceback_later(900, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
path, fname = sys.argv[1], sys.argv[2]
args = eval(sys.argv[3]); kw = eval(sys.argv[4]) if len(sys.argv) > 4 else {}
spec = u.spec_from_file_location("k5mod", path); m = u.module_from_spec(spec); spec.loader.exec_module(m)
TG = Target({"kind": "cuda", "arch": "sm_120a"})
t0 = time.time()
jf = getattr(m, fname)
tir = jf.get_tir(*args, **kw)
with TG:
    with tilelang.transform.PassContext(config=jf.pass_configs or {}):
        art = tilelang.lower(tir, target=TG)
src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
od = os.path.expanduser("$HK_WORK/k5/offline"); os.makedirs(od, exist_ok=True)
cu = f"{od}/{fname}_{'_'.join(map(str, args))}.cu"; open(cu, "w").write(src)
print("lowered", round(time.time() - t0, 1), cu, flush=True)
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
cub = cu[:-3] + ".cubin"
r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w", "-I" + TILELANG_TEMPLATE_PATH,
                    "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cub, cu], capture_output=True, text=True)
print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "Used" in l or "spill" in l or "rror" in l)[:3000], flush=True)
