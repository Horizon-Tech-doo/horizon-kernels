# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Offline SM120 compile of the G3v kernels (vLLM-layout G3 step + expand) WITHOUT a CUDA context (30-09-2026).

usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=<tl-pkgs> python gdnstate_lower.py PKG_PARENT OUT_DIR [variant,...]
  PKG_PARENT contains a package `hzgs/` = copies of horizon_ops/{__init__,gdn_decode,gdn_state}.py from the fork
  branch horizon/gdn-state. Lowers each variant with get_tir + tilelang.lower (no module load, no execution), then
  nvcc --cubin -Xptxas -v for registers / spills / shared memory. Compile success says nothing about GPU correctness
  or speed. Production geometry: HV 48, H 16, strides qkv 16384, a/b 96, z 16384, state slot 819200 elements.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK'] = '1'
import torch  # noqa: E402
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang  # noqa: E402
from tvm.target import Target  # noqa: E402
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR  # noqa: E402
from tilelang.contrib.nvcc import get_nvcc_compiler  # noqa: E402

sys.path.insert(0, sys.argv[1])
from hzgs import gdn_g3v as G3  # noqa: E402

out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
HV, H = 48, 16
QS, AS, BS, ZS, SS = 16384, 96, 96, 16384, 819200
DT, NW = "bfloat16", "bfloat16"

RF = 30720                      # conv region floats per slot: 10240 channels x 6 taps bf16 = 122,880 B
variants = []
for vr in (32, 64, 128):
    variants.append((f"g3v_step_vr{vr}", G3.g3v_step_kernel, (HV, H, vr, QS, AS, BS, ZS, SS, 4, RF, DT, NW, False)))
variants.append(("g3v_step_vr32_sigmoid", G3.g3v_step_kernel, (HV, H, 32, QS, AS, BS, ZS, SS, 4, RF, DT, NW, True)))
variants.append(("g3v_expand_persistent", G3.g3v_expand_kernel, (HV, H, SS, 4, RF)))
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
        try:  # dynamic shared memory bytes: max(buffer byte offset + buffer bytes) of the shared.dyn arena
            import re
            txt = str(artifact.device_mod)
            offs = {m[0]: int(m[2]) for m in re.findall(
                r'(\w+): T\.handle\("(\w+)", "shared\.dyn"\) = T\.handle_add_byte_offset\(buf_dyn_shmem\.data, (\d+)\)',
                txt)}
            size = {"float32": 4, "int32": 4, "bfloat16": 2, "float16": 2, "int16": 2}
            end = 0
            for m in re.findall(r'\w+ = T\.decl_buffer\(\((\d+),\),(?: "(\w+)",)? data=(\w+), scope="shared\.dyn"\)',
                                txt):
                n, dt, data = int(m[0]), (m[1] or "float32"), m[2]
                if data in offs:
                    end = max(end, offs[data] + n * size.get(dt, 4))
            rec["dyn_smem_bytes"] = end
        except Exception as ex:  # noqa: BLE001
            rec["smem_probe_error"] = str(ex)[:200]
        fm = ['--use_fast_math'] if (jf.pass_configs or {}).get(tilelang.PassConfigKey.TL_ENABLE_FAST_MATH) else []
        rec["fast_math"] = bool(fm)
        r = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w', *fm,
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
        rec["lower_error"] = f"{type(ex).__name__}: {str(ex)[-2500:]}"
    rows.append(rec)
    print(json.dumps(rec), flush=True)
(out / "summary.json").write_text(json.dumps(rows, indent=1))
