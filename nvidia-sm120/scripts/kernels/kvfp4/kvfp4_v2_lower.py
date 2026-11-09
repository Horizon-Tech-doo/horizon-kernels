# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Offline SM120 compile of the v2 (register-unpack) FP4 decode kernels WITHOUT a CUDA context.

usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<tl-pkgs> python kvfp4_v2_lower.py PKG_PARENT OUT_DIR [variant,...]
  PKG_PARENT/hzkv = copies of horizon_ops/{__init__,attn_decode,kv_write_fp4,attn_decode_fp4,attn_decode_fp4_v2,
  kv_fp4_ref}.py. Writes .cu, .cubin, ptxas log, SASS (cuobjdump -sass) and an instruction histogram of the main
  loop per variant, plus summary.json. Compile success says nothing about GPU correctness or speed.
"""
import collections
import json
import os
from pathlib import Path
import re
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
from hzkv import attn_decode_fp4_v2 as V2  # noqa: E402

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
PS, D = 64, 256


def strides(KVH):
    pb = 2 * KVH * PS * (D // 2 + D // 16)
    return (pb, D // 2, PS * D // 2), (pb, D // 16, PS * D // 16)


variants = []
for KVH, G, tag in ((4, 6, "dense"), (2, 8, "moe")):
    ds, ss = strides(KVH)
    for QL in (1, 4):
        for NST in (4, 3):
            ncw = V2.default_ncw(G, QL)
            variants.append((f"v2_{tag}_ql{QL}_ncw{ncw}_nst{NST}", (KVH, G, QL, PS, *ds, *ss, ncw, NST, 64),
                             V2.smem_bytes(KVH, G, QL, ncw, NST)))

nvdis = Path(get_nvcc_compiler()).with_name("nvdisasm")


def loop_stats(sass):
    """instruction histogram of the kernel and of its largest backward-branch loop (the 16-token block loop)"""
    lines = sass.splitlines()
    seq, labels = [], {}
    for ln in lines:
        ml = re.match(r'\s*(\.L_x_\d+):', ln)
        if ml:
            labels[ml.group(1)] = len(seq)
            continue
        mi = re.match(r'\s*/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]*)?\s*([^;]*);', ln)
        if mi:
            seq.append((mi.group(2), mi.group(4)))
    best = (0, 0)
    for i, (op, rest) in enumerate(seq):
        if op == "BRA":
            mt = re.search(r'(\.L_x_\d+)', rest)
            if mt and mt.group(1) in labels and labels[mt.group(1)] <= i:
                j = labels[mt.group(1)]
                if i - j > best[1] - best[0]:
                    best = (j, i)
    body = seq[best[0]:best[1] + 1]
    return (len(seq), dict(collections.Counter(o for o, _ in seq).most_common(16)), len(body),
            dict(collections.Counter(o for o, _ in body).most_common(16)))
rows = []
for name, args, smem in variants:
    if only and name not in only:
        continue
    rec = {"variant": name, "smem_bytes": smem}
    try:
        jf = V2.attn_fp4_v2_kernel
        tir = jf.get_tir(*args)
        with target:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                artifact = tilelang.lower(tir, target=target)
        cu = out / f'{name}.cu'
        cu.write_text(artifact.kernel_source)
        cub = cu.with_suffix('.cubin')
        r = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                            '-I' + TILELANG_TEMPLATE_PATH, '-I' + CUTLASS_INCLUDE_DIR,
                            '-Xptxas', '-v', '-o', str(cub), str(cu)], capture_output=True, text=True)
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
        if r.returncode == 0 and nvdis.exists():
            sass = subprocess.run([str(nvdis), '-c', str(cub)], capture_output=True, text=True).stdout
            (out / f'{name}.sass').write_text(sass)
            tot, hist, nloop, lhist = loop_stats(sass)
            rec.update(sass_total=tot, sass_top=hist, loop_instructions=nloop, loop_top=lhist)
    except Exception as ex:  # lowering failure is a result
        rec["lower_error"] = f"{type(ex).__name__}: {str(ex)[-2000:]}"
    rows.append(rec)
    print(json.dumps(rec), flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=1))
