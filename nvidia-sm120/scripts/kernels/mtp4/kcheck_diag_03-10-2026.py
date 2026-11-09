#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""kcheck_diag_03-10-2026.py (Horizon Tech, 03-10-2026): measure-only diagnostic of mtp_train_v4's kernel check, which
failed at 00:29 UTC (emulation vs prefill_rs.linear_native 0.0038 relative L2, W4A16 0.0106; the check wants >= 5x).
Splits the gap into its parts on the same two layers, same seed/rows as kernel_check:
  act_exact   fp4x2 activations (fp32) x fp32 dequantised weights in fp64  vs  the kernel with fp32 output
  trainer     the trainer's path: fp32 activations x BF16 weights (two bf16 GEMMs), fp32 out  vs  kernel fp32 out
  trainer_bf  the trainer's path rounded to bf16 (what kernel_check compares)  vs  kernel bf16 out (the check)
  kbf_vs_k32  the kernel's own bf16 output rounding
  w4a16       bf16 x bf16 weights  vs  kernel fp32 out
Needs the GPU with vLLM stopped (run under gpu_lock by kcheck_diag_03-10-2026.sh). Prints one JSON line."""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("$HK_WORK/mtp4"))
TL = os.environ["TL_PREFIX"]
sys.path.insert(0, TL)
import mtp_train_v4 as M  # noqa: E402
from safetensors import safe_open  # noqa: E402
from tilelang.contrib.sm120_nvfp4 import prefill_rs  # noqa: E402


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    pre = "model.language_model.layers."
    out = {}
    with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", framework="pt", device="cpu") as f:
        for name in ("3.mlp.down_proj", "2.linear_attn.out_proj"):
            packed = f.get_tensor(f"{pre}{name}.weight_packed").to(dev)
            scale = f.get_tensor(f"{pre}{name}.weight_scale").to(dev)
            gsc = f.get_tensor(f"{pre}{name}.weight_global_scale").to(dev)
            K = packed.shape[1] * 2
            x = (torch.randn(192, K, device=dev, generator=g) * 0.5).to(torch.bfloat16)
            x[:, :8] *= 20.0
            h = prefill_rs.prepare_weight_native(packed, scale, gsc)
            yk32 = prefill_rs.linear_native(x, h, passes=2, out_dtype=torch.float32).float()
            yk16 = prefill_rs.linear_native(x, h, passes=2).float()
            a = M.fp4x2_act_qdq(x)
            w32 = M.dequant_nvfp4(packed, scale, gsc, torch.float32)
            w16 = M.dequant_nvfp4(packed, scale, gsc, torch.bfloat16)
            y_exact = a.double() @ w32.double().t()
            y_tr = M.linear_fp32_via_bf16(a, w16)
            y_trbf = M.fp4x2_linear(x, w16).float()
            y_w4 = F.linear(x, w16).float()
            # repeat determinism of the kernel itself
            yk32b = prefill_rs.linear_native(x, h, passes=2, out_dtype=torch.float32).float()
            out[name] = {"act_exact_vs_k32": rel(y_exact, yk32), "trainer_fp32_vs_k32": rel(y_tr, yk32),
                         "trainer_bf16_vs_k16 (the check)": rel(y_trbf, yk16), "w4a16_vs_k16 (the check)": rel(y_w4, yk16),
                         "w4a16_vs_k32": rel(y_w4, yk32), "k16_vs_k32": rel(yk16, yk32), "k32_repeat": rel(yk32b, yk32),
                         "w16_vs_w32": rel(w16.float(), w32)}
            del packed, scale, h
    print("KCHECK_DIAG " + json.dumps({k: {kk: round(vv, 7) for kk, vv in v.items()} for k, v in out.items()}))


if __name__ == "__main__":
    main()
