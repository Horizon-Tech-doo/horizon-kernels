# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU-only: lower + compile the K4 segment kernel for the M buckets (no GPU), print smem, ptxas registers, spills.
import os, re, subprocess, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.contrib.horizon_k4 import segment as sg
from tilelang.contrib.horizon_fused import skinny_bf16 as sk
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
Ms = [int(m) for m in (sys.argv[2] if len(sys.argv) > 2 else "4,16,32").split(",")]
stamps = int(os.environ.get("STAMPS", "0"))
hw = int(os.environ.get("HW", "0"))
gdn = int(os.environ.get("GDN", "0"))
stages = int(os.environ.get("STAGES", "4"))
TG = Target({"kind": "cuda", "arch": "sm_120a"})
for M in Ms:
    WM, splits = sg.bucket(M)
    skc = sg.SK_CFG
    t0 = time.time()
    tir = sg.segment_kernel.get_tir(376, WM, *splits, skc[0], skc[1], stages, 1, 1e-6, 1, stamps, 2_000_000_000, 0, 0, sg.NQ, hw, gdn)
    with TG:
        with tilelang.transform.PassContext(config=sg.segment_kernel.pass_configs or {}):
            art = tilelang.lower(tir, target=TG)
    src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
    cu = f"{out}/seg_M{M}_s{stamps}_hw{hw}_st{stages}_g{gdn}.cu"; open(cu, "w").write(src)
    r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w", "-I" + TILELANG_TEMPLATE_PATH,
                        "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cu[:-3] + ".cubin", cu], capture_output=True, text=True)
    smem = re.findall(r"(\d+) bytes smem", r.stderr)
    dyn = re.findall(r"buf_dyn_shmem\[\]", src)
    print(f"M={M} WM={WM} splits={splits} sk={skc} lower={time.time()-t0:.0f}s rc={r.returncode} "
          f"regs={re.findall(r'Used (\d+) registers', r.stderr)} spill={re.findall(r'(\d+) bytes spill stores', r.stderr)} "
          f"static_smem={smem}", flush=True)
    m = re.search(r"dynamic_smem_size[^0-9]*(\d+)", str(getattr(art, 'params', '')) )
    if r.returncode:
        print(r.stderr[-3000:])
print("SEG_LOWER_DONE")
