# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Offline codegen dump of every production TileLang kernel (no GPU, no CUDA context).

Run once per TileLang build (PYTHONPATH = the build prefix first, then the production plugin dir):
    python codegen_dump.py OUT_DIR [--plugin DIR] [--quick] [--device-compile]

For each kernel spec (family, factory, args) it builds the PrimFunc with the factory's get_tir(), lowers it for
SM120 (tilelang.lower, device source only) and writes OUT_DIR/src/<sha>.cu plus OUT_DIR/index.jsonl
({key, family, sha256 of the CUDA source, error}). compare_codegen.py diffs two dumps and compiles the differing (or
all unique) sources with the production nvcc flags for register / spill counts.

Kernel list = what production (<env-file>, 30-09-2026) compiles:
  regfrag / regfrag_xl  W4A16 NVFP4 decode, M 1..1024, 5 main shapes + the NVFP4 MTP layer (incl. fc 5120x10240)
  rsN3                  native-order FP4 prefill (prefill_rs.nvfp4_gemm_rs_kernel, conv 4, NATIVE_OPTS), passes 2 (+1)
  row_scale             horizon_ops.row_scale (row_scale_kernel + quant_rs_nopad_kernel), HORIZON_PREFILL_RS=1
  k5                    skinny BF16 GEMM (in_proj_ba 96x5120, every config for M <= 48) + add/RMSNorm (H 5120)
  k1a                   fused FP8 draft head (49152 x 5120, act fp8) from the plugin's draft_head.py
The dispatch below mirrors decode_regfrag_xl.linear / _run / _rf_pass and decode_regfrag.linear exactly.
"""
import hashlib
import json
import os
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] = "1"
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0

OUT = sys.argv[1]
QUICK = "--quick" in sys.argv
# --device-compile: also run nvcc through tilelang's own callback (tilelang/cuda/backend.py), which stores the cubin in
# TILELANG_CACHE_DIR/<ver>/cuda-binaries keyed by (source, flags): pre-warms a kernel cache without a GPU.
DEVCOMP = "--device-compile" in sys.argv
PLUGIN = sys.argv[sys.argv.index("--plugin") + 1] if "--plugin" in sys.argv else "$HK_DATA/horizon-pkgs-v030-29-09-2026"
sys.path.insert(0, PLUGIN)

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tvm.target import Target  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rf  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs  # noqa: E402
from tilelang.contrib.horizon_ops import row_scale as rsc  # noqa: E402
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as an  # noqa: E402
from horizon_kernels import draft_head as dh  # noqa: E402

os.makedirs(os.path.join(OUT, "src"), exist_ok=True)
TARGET = Target({"kind": "cuda", "arch": "sm_120a"})
SPECS = {}          # key -> (family, jitfunc, args, kwargs)


def add(family, jf, *args, **kwargs):
    name = getattr(jf, "__name__", None) or getattr(getattr(jf, "func", None), "__name__", None) or repr(jf)
    key = f"{family}:{name}:" + ",".join(map(str, args)) + \
        ("|" + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items())) if kwargs else "")
    SPECS.setdefault(key, (family, jf, args, kwargs))


# --- regfrag / regfrag_xl (mirrors decode_regfrag_xl.linear) --------------------------------------------------------
MAIN = [(16384, 5120), (14336, 5120), (5120, 6144), (34816, 5120), (5120, 17408)]
MTP = [(5120, 10240)]                                 # mtp.fc; the MTP layer's other four shapes equal MAIN's


def xl_run(N, K, cfg):
    warps_m, WM, warps_n, nb, block_K, split_k, num_stages = cfg[:7]
    min_blocks = cfg[7] if len(cfg) > 7 else 1
    late_issue = cfg[8] if len(cfg) > 8 else xl._LATE_ISSUE_DEFAULT
    warps_k = cfg[9] if len(cfg) > 9 else 1
    if split_k < 0 or split_k >= 1000:
        add("regfrag_xl_sk", xl.nvfp4_regfrag_xl_sk_kernel, N, K, warps_m, WM, warps_n, nb, num_stages, min_blocks,
            block_K, late_issue, warps_k)
    else:
        add("regfrag_xl", xl.nvfp4_regfrag_xl_kernel, N, K, warps_m, WM, warps_n, nb, block_K, split_k, num_stages,
            min_blocks)


def rf_pass(N, K, M):
    cfg = rf.config(N, K, M)
    if xl._rf_padded(M):
        xl_run(N, K, xl._rf_twin(cfg))
    else:
        add("regfrag", rf.nvfp4_regfrag_kernel, N, K, *cfg)


def xl_linear(N, K, M):
    cfg = xl.config(N, K, M)
    if cfg[0] == "rf" and M <= rf.SUPPORTS_M:
        rcfg = tuple(cfg[1:]) if len(cfg) > 1 else rf.config(N, K, M)
        if not xl._rf_padded(M):
            add("regfrag", rf.nvfp4_regfrag_kernel, N, K, *rcfg)
            return
        cfg = xl._rf_twin(rcfg)
    if cfg[0] == "rf":
        for r0 in range(0, M, rf.SUPPORTS_M):
            rf_pass(N, K, min(rf.SUPPORTS_M, M - r0))
        return
    xl_run(N, K, cfg)


MS = range(1, xl.SUPPORTS_M + 1) if not QUICK else (1, 4, 16, 17, 32, 33, 64, 65, 96, 128, 256, 512, 1000, 1024)
for (N, K) in MAIN + MTP:
    for M in MS:
        xl_linear(N, K, M)

# --- rsN3 native prefill (prefill_rs.linear_native -> get_kernel(n, k, passes, None, conv=4, **NATIVE_OPTS)) --------
for (N, K) in MAIN:
    for p in (2, 1):
        add("rsN3" if p == 2 else "rsN3_p1_unused", prs.nvfp4_gemm_rs_kernel, N, K, p, T.bfloat16, 4, **prs.NATIVE_OPTS[p])

# --- row_scale (HORIZON_PREFILL_RS=1: prefill_rs.quantize_rs -> horizon_ops.row_scale.quantize_rs) -------------------
for K in sorted({K for _, K in MAIN}):
    add("row_scale", rsc.row_scale_kernel, K)
    add("row_scale_quant", rsc.quant_rs_nopad_kernel, K, 2)

# --- K5 -------------------------------------------------------------------------------------------------------------
for cfg in sk.all_configs(96, 5120):
    add("k5_skinny", sk.skinny_bf16_kernel, 96, 5120, *cfg)
for has_res in (True, False):
    add("k5_norm", an.add_rmsnorm_kernel, 5120, 1e-6, has_res, an._threads(5120))

# --- K1a fused FP8 draft head ------------------------------------------------------------------------------------------
add("k1a", dh.draft_head_fp8, 49152, 5120)

print(json.dumps({"tilelang": tilelang.__file__, "version": getattr(tilelang, "__version__", "?"), "specs": len(SPECS)}),
      flush=True)
t0 = time.time()
idx = open(os.path.join(OUT, "index.jsonl"), "w")
ok = err = 0
for key, (family, jf, args, kwargs) in SPECS.items():
    rec = {"key": key, "family": family}
    try:
        tir = jf.get_tir(*args, **kwargs)
        with TARGET:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                art = tilelang.lower(tir, target=TARGET, enable_device_compile=DEVCOMP)
        src = art.kernel_source
        sha = hashlib.sha256(src.encode()).hexdigest()
        path = os.path.join(OUT, "src", sha + ".cu")
        if not os.path.exists(path):
            open(path, "w").write(src)
        rec.update(sha=sha, pass_configs={str(k): str(v) for k, v in (jf.pass_configs or {}).items()})
        ok += 1
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {e}"[:2000], tb=traceback.format_exc()[-3000:])
        err += 1
    idx.write(json.dumps(rec) + "\n")
    idx.flush()
print(json.dumps({"CODEGEN_DUMP_DONE": OUT, "ok": ok, "errors": err, "secs": round(time.time() - t0)}), flush=True)
