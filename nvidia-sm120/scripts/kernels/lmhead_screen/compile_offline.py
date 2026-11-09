# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline SM120 compile of the lm_head screening kernels (tilelang.contrib.sm120_fp8.lmhead_screen): no GPU, no CUDA
# context. Lowers each kernel with the factory's get_tir() for sm_120a, compiles the CUDA source with the production
# nvcc flags (codegen_dump.py / compare_codegen.py recipe) and reports registers, stack and spills from ptxas.
#   PYTHONPATH=<candidate TileLang prefix> python compile_offline.py OUT_DIR [--module PATH_TO_lmhead_screen.py]
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

import tilelang  # noqa: E402
from tvm.target import Target  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
if "--module" in sys.argv:
    spec = importlib.util.spec_from_file_location("hz_lmscreen", sys.argv[sys.argv.index("--module") + 1])
    ls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ls)
else:
    from tilelang.contrib.sm120_fp8 import lmhead_screen as ls  # noqa: E402

TARGET = Target({"kind": "cuda", "arch": "sm_120a"})
CU = "$HK_VENV/lib/python3.12/site-packages/nvidia/cu13"
PREFIX = os.path.dirname(os.path.dirname(tilelang.__file__))
INC = ["-I" + PREFIX + "/tilelang/3rdparty/../src", "-I" + PREFIX + "/tilelang/3rdparty/cutlass/include"]
V, K = 248320, 5120
SPECS = [
    ("tile_lower_max", ls.tile_lower_max_kernel, (V, 128)),
    ("tile_lower_max_t16", ls.tile_lower_max_kernel, (V, 16, 64)),
    ("compact_k4096", ls.compact_kernel, (V, 4096)),
    ("compact_k16384", ls.compact_kernel, (V, 16384)),
    ("recheck_k4096", ls.recheck_kernel, (V, K, 4096)),
    ("recheck_k16384", ls.recheck_kernel, (V, K, 16384)),
    ("full_rows", ls.full_rows_kernel, (V, K)),
]
RE_REG = re.compile(r"Used (\d+) registers")
RE_SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
res = []
for name, jf, args in SPECS:
    rec = {"kernel": name, "args": list(args)}
    try:
        tir = jf.get_tir(*args)
        with TARGET:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                art = tilelang.lower(tir, target=TARGET)
        src = art.kernel_source
        sha = hashlib.sha256(src.encode()).hexdigest()[:16]
        path = os.path.join(OUT, f"{name}_{sha}.cu")
        open(path, "w").write(src)
        cub = path[:-3] + ".cubin"
        cmd = [CU + "/bin/nvcc", "-ccbin=g++", "--cubin", "-O3", "-lineinfo", "-arch=sm_120a", "-std=c++20"] + INC + \
            ["--ptxas-options=--verbose", "-w", "-o", cub, path]
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout + p.stderr
        regs = [int(v) for v in RE_REG.findall(out)]
        sp = [tuple(map(int, t)) for t in RE_SPILL.findall(out)]
        rec.update(sha=sha, nvcc_rc=p.returncode, regs=max(regs) if regs else None,
                   stack=max((s[0] for s in sp), default=None), spill_st=max((s[1] for s in sp), default=None),
                   spill_ld=max((s[2] for s in sp), default=None), src=path)
        if p.returncode:
            rec["log"] = out[-2500:]
    except Exception as e:  # noqa: BLE001
        import traceback
        rec.update(error=f"{type(e).__name__}: {e}"[:3000], tb=traceback.format_exc()[-4000:])
    res.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("tb", "log")}), flush=True)
    if "tb" in rec:
        print(rec["tb"], flush=True)
    if "log" in rec:
        print(rec["log"], flush=True)
json.dump(res, open(os.path.join(OUT, "compile.json"), "w"), indent=1)
print(json.dumps({"LMSCREEN_COMPILE_DONE": OUT, "ok": sum(1 for r in res if r.get("nvcc_rc") == 0),
                  "total": len(res)}), flush=True)
