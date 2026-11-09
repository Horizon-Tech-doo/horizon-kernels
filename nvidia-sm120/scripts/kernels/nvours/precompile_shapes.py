# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Fill TileLang's disk kernel cache for new NVFP4 layer shapes WITHOUT touching the GPU (no CUDA context: safe while
# another job owns the GPU). 27-09-2026, for nvidia/Qwen3.8-27B-NVFP4's lm_head (248320 x 5120).
#
#   . $HK_WORK/tl/tl_env.sh; export PYTHONPATH=$HK_DATA/tl-pkgs-natpf-07ee3f21 TILELANG_CACHE_DIR=...
#   TVM_FFI_DISABLE_TORCH_C_DLPACK=1 python precompile_shapes.py 248320x5120[,NxK...] [MMAX=1024]
#
# Drives the REAL dispatchers (decode_regfrag_xl.linear for every M = 1..MMAX, prefill_rs native fp4x2 / fp4x1) on
# meta tensors with the kernel constructors wrapped: each distinct kernel is compiled exactly as the plugin would
# compile it at load, and never launched.
import faulthandler
import os
import sys
import time
import types

faulthandler.dump_traceback_later(5400, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang  # noqa: E402,F401
from tvm.target import Target  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rf  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import prefill_rs as rs  # noqa: E402

shapes = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1].split(",")]
MMAX = int(sys.argv[2]) if len(sys.argv) > 2 else xl.SUPPORTS_M
built = []


class _NoLaunch:
    def __call__(self, *a, **k):
        return None


def _wrap(mod, name):
    orig = getattr(mod, name)

    def f(*a, **k):
        t = time.time()
        orig(*a, **k)
        built.append((name, a, k))
        print(f"compiled {name}{a}{k or ''} in {time.time() - t:.0f} s", flush=True)
        return _NoLaunch()

    setattr(mod, name, f)


for m, n in ((rf, "nvfp4_regfrag_kernel"), (xl, "nvfp4_regfrag_xl_kernel"), (xl, "nvfp4_regfrag_xl_sk_kernel"),
             (rs, "nvfp4_gemm_rs_kernel")):
    _wrap(m, n)

t0 = time.time()
err = 0
meta = torch.device("meta")
with Target({"kind": "cuda", "arch": "sm_120a"}):
    for N, K in shapes:
        prep = types.SimpleNamespace(N=N, K=K, Wg=torch.empty(1, device=meta), Sg=torch.empty(1, device=meta),
                                     alpha=torch.empty(N, device=meta), counters=torch.empty(N // 32, device=meta))
        for M in range(1, MMAX + 1):
            try:
                xl.linear(torch.empty(M, K, device=meta, dtype=torch.bfloat16), prep)
            except Exception as ex:  # noqa: BLE001
                err += 1
                print("ERR regfrag", N, K, M, type(ex).__name__, str(ex)[:300], flush=True)
        for passes in (2, 1):
            try:
                rs.get_kernel(N, K, passes, None, 4, **rs.NATIVE_OPTS[passes])
            except Exception as ex:  # noqa: BLE001
                err += 1
                print("ERR prefill_rs", N, K, passes, type(ex).__name__, str(ex)[:300], flush=True)
print(f"PRECOMPILE_SHAPES shapes={shapes} kernels={len(built)} errors={err} secs={time.time() - t0:.0f}", flush=True)
sys.exit(1 if err else 0)
