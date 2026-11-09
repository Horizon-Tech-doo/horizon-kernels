# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Review #14: with per-row scaling, fp4x2 accuracy should not depend on the activation magnitude.
# Compare the fork's quantizer with row scale (current) vs forced rs=1 (old behaviour) across magnitudes.
import json, torch
from tilelang.contrib.sm120_nvfp4 import kernels as K, prepare_weight
from safetensors import safe_open
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    pre = "model.language_model.layers.3.mlp.down_proj."
    wp = f.get_tensor(pre + "weight_packed"); ws = f.get_tensor(pre + "weight_scale"); gs = f.get_tensor(pre + "weight_global_scale").float()
N, K_ = wp.shape[0], wp.shape[1] * 2
u = wp.view(torch.uint8); W = torch.empty(N, K_, device="cuda"); W[:, 0::2] = LUT[(u & 0xF).long()]; W[:, 1::2] = LUT[(u >> 4).long()]
W = W * ws.float().repeat_interleave(16, 1) / gs
w = prepare_weight(wp, ws, gs)
torch.manual_seed(0)
base = torch.randn(512, K_, device="cuda"); base[:, torch.randint(0, K_, (8,))] *= 20
for mag in (1.0, 1e-2, 1e-3, 1e-4, 30.0):
    x = (base * mag).to(torch.bfloat16)
    ref = x.float() @ W.T
    r = {"magnitude": mag}
    for name, rowscale in (("row_scale", True), ("no_row_scale", False)):
        amax = x.abs().amax(1).float()
        rs = 2600.0 / amax if rowscale else torch.ones_like(amax)
        A, SFA = K.nvfp4_quant_kernel(x.contiguous(), rs.contiguous(), passes=2)
        C = K.nvfp4_gemm_kernel(A, w.packed, SFA.reshape(2, 512 * (K_ // 256), 4), w.sfb, w.alpha, (1.0 / rs).contiguous(), passes=2)
        r[name] = round(((C.float() - ref).norm() / ref.norm()).item(), 5)
    print(json.dumps(r), flush=True)
