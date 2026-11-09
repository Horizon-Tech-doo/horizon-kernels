# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch; torch.cuda.is_available = lambda: False
import tilelang
from tvm.target import Target
from tilelang.contrib.sm120_nvfp4 import prefill_rs as r
with Target({"kind": "cuda", "arch": "sm_120a"}):
    for K in (5120, 6144, 17408):
        k = r.row_scale_kernel(K, 2600.0, 128)
        src = k.get_kernel_source()
        print("OK", K, len(src), "__fdiv_rn" in src)
