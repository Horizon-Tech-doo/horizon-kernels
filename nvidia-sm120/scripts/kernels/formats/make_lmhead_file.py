# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S3 (formats track): write a quantized copy of a BF16 lm_head for HORIZON_LMHEAD_QUANT / HORIZON_LMHEAD_FILE
# (horizon_kernels/lmhead.py). The checkpoint is only read.
#   make_lmhead_file.py fp8 OUT.safetensors [SRC.safetensors]      FP8 e4m3 per row (amax/448), weight_scale (V,) fp32
#   make_lmhead_file.py nvidia OUT.safetensors                      NVIDIA's NVFP4 lm_head tensors (ModelOpt naming)
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OURS = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
NV = "$HK_DATA/models/nvidia-Qwen3.8-27B-NVFP4/model-00003-of-00003.safetensors"
mode, out = sys.argv[1], sys.argv[2]
if mode == "fp8":
    with safe_open(sys.argv[3] if len(sys.argv) > 3 else OURS, "pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")
    q = torch.empty(W.shape, dtype=torch.float8_e4m3fn)
    s = torch.empty(W.shape[0], dtype=torch.float32)
    for r0 in range(0, W.shape[0], 8192):
        w = W[r0:r0 + 8192].float()
        sc = w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
        q[r0:r0 + 8192] = (w / sc).to(torch.float8_e4m3fn)
        s[r0:r0 + 8192] = sc.reshape(-1)
    err = ((q.float() * s[:, None] - W.float()).norm() / W.float().norm()).item()
    save_file({"weight": q, "weight_scale": s}, out, metadata={"format": "horizon_lmhead_fp8_row",
                                                              "source": "Qwen3.8-27B-NVFP4 lm_head.weight (BF16)",
                                                              "rel_err": f"{err:.6f}"})
    print("fp8 per-row lm_head", tuple(q.shape), "rel_err", round(err, 6), "->", out)
elif mode == "nvidia":
    with safe_open(NV, "pt", device="cpu") as f:
        t = {"weight": f.get_tensor("lm_head.weight"), "weight_scale": f.get_tensor("lm_head.weight_scale"),
             "weight_scale_2": f.get_tensor("lm_head.weight_scale_2").float().reshape(1)}
    save_file(t, out, metadata={"format": "horizon_lmhead_nvfp4_modelopt", "source": "nvidia/Qwen3.8-27B-NVFP4 lm_head"})
    print("nvidia nvfp4 lm_head", {k: tuple(v.shape) for k, v in t.items()}, "->", out)
else:
    raise SystemExit("mode: fp8 | nvidia")
