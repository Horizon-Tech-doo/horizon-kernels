# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Offline SM120 compile of the NVFP4 KV-cache kernels WITHOUT a CUDA context (no module load, no execution).

usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<tl-pkgs> python kvfp4_lower.py PKG_PARENT OUT_DIR
  PKG_PARENT contains a package `hzkv/` = copies of horizon_ops/{__init__,attn_decode,kv_write_fp4,attn_decode_fp4,
  kv_fp4_ref}.py from the horizon/kv-fp4 branch. Lowers each variant with get_tir + tilelang.lower, then nvcc --cubin
  -Xptxas -v for registers / spills. Compile success says nothing about GPU correctness or speed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK'] = '1'
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler

sys.path.insert(0, sys.argv[1])
from hzkv import attn_decode_fp4 as AD, kv_write_fp4 as KW  # noqa: E402

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
KVH, G, D, PS = 4, 6, 256, 64
PB = 2 * KVH * PS * (D // 2 + D // 16)                   # 73,728-byte kernel page (vLLM NVFP4 geometry, HND)
DSTR = (PB, D // 2, PS * D // 2)
SSTR = (PB, D // 16, PS * D // 16)

variants = []
for had in (1, 0):
    for search in (1, 0):
        variants.append((f"write_had{had}_search{search}", KW.kv_write_fp4_kernel,
                         (KVH, D, PS, 12288, 256, 12288, 256, *DSTR, *SSTR, 128, had, search)))
variants.append(("rotate_q_hq24", AD.rotate_q_kernel, (KVH * G, D, 128)))
for QL in (1, 4):
    for pipe, bn in ((1, 32), (0, 64)):
        variants.append((f"split_ql{QL}_pipe{pipe}_bn{bn}", AD.attn_fp4_split_kernel,
                         (KVH, G, QL, D, PS, bn, 32, 128, 256, *DSTR, *SSTR, pipe)))

rows = []
for name, jf, args in variants:
    if only and name not in only:
        continue
    rec = {"variant": name}
    try:
        tir = jf.get_tir(*args)
        with target:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                artifact = tilelang.lower(tir, target=target)
        cu = out / f'{name}.cu'
        cu.write_text(artifact.kernel_source)
        r = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                            '-I' + TILELANG_TEMPLATE_PATH, '-I' + CUTLASS_INCLUDE_DIR,
                            '-Xptxas', '-v', '-o', str(cu.with_suffix('.cubin')), str(cu)],
                           capture_output=True, text=True)
        log = r.stdout + r.stderr
        (out / f'{name}.compiler.log').write_text(log)
        rec["nvcc_rc"] = r.returncode
        for line in log.splitlines():
            if "registers" in line:
                rec["ptxas_usage"] = line.strip()
            if "spill" in line:
                rec["ptxas_spill"] = line.strip()
            if "error" in line.lower():
                rec.setdefault("errors", []).append(line.strip()[:300])
    except Exception as ex:  # lowering failure is a result, keep going
        rec["lower_error"] = f"{type(ex).__name__}: {str(ex)[-1500:]}"
    rows.append(rec)
    print(json.dumps(rec), flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=1))
