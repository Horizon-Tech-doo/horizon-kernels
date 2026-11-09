# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Fill TileLang's disk kernel cache (TILELANG_CACHE_DIR) for the MoE kernels WITHOUT a CUDA context (27-09-2026):
# horizon_moe.route (route_shared, route_align at block sizes 8 and 16) and every grouped_w4a16 config the step-4 race
# sweeps, so the GPU window is not spent compiling. Pattern of $HK_WORK/nvours/precompile_shapes.py.
#   TVM_FFI_DISABLE_TORCH_C_DLPACK=1 PYTHONPATH=$HK_WORK/moe-k/tl-pkgs TILELANG_CACHE_DIR=$HK_WORK/moe-k/tlcache \
#     flock $HK_WORK/.compile.lock python precompile_moe.py
import faulthandler, itertools, os, time
faulthandler.dump_traceback_later(7200, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch  # noqa: E402
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang  # noqa: E402,F401
from tvm.target import Target  # noqa: E402
from tilelang.contrib.horizon_moe import grouped_w4a16 as gw, route  # noqa: E402

jobs = [(route.route_shared_kernel, (256, 264, 8, True)), (route.route_align_kernel, (256, 264, 8, True, 8)),
        (route.route_align_kernel, (256, 264, 8, True, 16))]
for nw, sk, st in itertools.product((4, 8), (1, 2, 4, 8), (3, 4)):
    jobs.append((gw.moe_grouped_kernel, ("up", 257, 1024, 2048, 9, nw, 64, sk, st)))
for sk, st in itertools.product((1, 2, 4), (3, 4)):
    jobs.append((gw.moe_grouped_kernel, ("down", 257, 2048, 512, 9, 4, 64, sk, st)))
t0 = time.time(); bad = 0
with Target({"kind": "cuda", "arch": "sm_120a"}):
    for fn, args in jobs:
        t = time.time()
        try:
            fn(*args)
            print(f"compiled {args} in {time.time() - t:.0f} s", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAILED {args}: {str(e)[:300]}", flush=True)
print(f"PRECOMPILE_DONE {len(jobs) - bad}/{len(jobs)} in {time.time() - t0:.0f} s", flush=True)
