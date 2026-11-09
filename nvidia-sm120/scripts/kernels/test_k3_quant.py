# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K3 fused two-pass quantizer: bit-exactness vs upstream single-pass quantizer, end-to-end error on a real layer,
# and speed vs two separate upstream quantizer calls.
import sys, json, torch
sys.path.insert(0, "$HK_WORK/tl/k3/upstream"); sys.path.insert(0, "$HK_WORK/tl/k3")
import tilelang.language as T
from safetensors import safe_open
from horizon_kernels.fp4_quant import fp4_residual_quant
from horizon_kernels.fp4_residual_gemm import fp4_residual_gemm
from maint.gemm.gemm_sm120.tilelang_nvfp4_quantizer import tilelang_quantize_bf16_to_nvfp4_blockscaled as up_quant
from examples.dequantize_gemm.quantize.nvfp4 import unswizzle_blockscaled_chunk_kmajor_scale_words, decode_ue4m3_scale_bytes, decode_packed_fp4_e2m1
from sm120_nvfp4_blockscaled_gemm import swizzle_blockscaled_chunk_kmajor_scale_words, _pack_scale_words, _decode_rowmajor_fp4


def dequant_act(A, SFw, K):
    w = unswizzle_blockscaled_chunk_kmajor_scale_words(SFw).to(torch.int64)
    b = torch.stack([(w >> s) & 0xFF for s in (0, 8, 16, 24)], -1).reshape(w.shape[0], -1).to(torch.uint8)
    return decode_packed_fp4_e2m1(A, K).float() * decode_ue4m3_scale_bytes(b).float().repeat_interleave(16, 1)


def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


torch.manual_seed(0)
M, K = 512, 5120
x = torch.randn(M, K, device="cuda"); x[:, torch.randint(0, K, (8,))] *= 20; x = x.to(torch.bfloat16)
A_cat, SF1, SF2 = fp4_residual_quant(x)
# pass 1 must equal upstream's quantizer
U1, US1 = up_quant(x)
q1_bytes = torch.equal(A_cat[:, :K // 2], U1); q1_scales = torch.equal(SF1, US1)
# pass 2 must equal upstream's quantizer applied to the exact residual (bf16 residual may round; report agreement)
r = (x.float() - dequant_act(U1, US1, K))
U2, US2 = up_quant(r.to(torch.bfloat16))
q2_byte_agree = (A_cat[:, K // 2:] == U2).float().mean().item(); q2_scale_agree = (SF2 == US2).float().mean().item()
# our pass 2 is computed from the fp32 residual, so it can be slightly better than upstream on a bf16 residual
x_rec = dequant_act(A_cat[:, :K // 2].contiguous(), SF1, K) + dequant_act(A_cat[:, K // 2:].contiguous(), SF2, K)
print(json.dumps({"q1_bytes_equal": q1_bytes, "q1_scales_equal": q1_scales, "q2_byte_agree": round(q2_byte_agree, 5),
                  "q2_scale_agree": round(q2_scale_agree, 5),
                  "activation_rel_err_1pass": round(((dequant_act(U1, US1, K) - x.float()).norm() / x.float().norm()).item(), 5),
                  "activation_rel_err_2pass": round(((x_rec - x.float()).norm() / x.float().norm()).item(), 5)}), flush=True)

# end to end on a real layer: fused quantizer + fused GEMM vs exact dequantised-weight reference
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    pre = "model.language_model.layers.3.mlp.gate_proj."
    wp = f.get_tensor(pre + "weight_packed"); ws = f.get_tensor(pre + "weight_scale"); wgs = f.get_tensor(pre + "weight_global_scale").float()
N = wp.shape[0]
Wdq = _decode_rowmajor_fp4(wp.view(torch.int8), N, K) * ws.float().repeat_interleave(16, 1) / wgs
ref = x.float() @ Wdq.T
SFB = swizzle_blockscaled_chunk_kmajor_scale_words(_pack_scale_words(ws.view(torch.uint8))).reshape(-1, 4)
g = fp4_residual_gemm(M, N, K, out_dtype=T.float32)
C = torch.empty(M, N, device="cuda")
alpha = (1.0 / wgs).reshape(1)
g(A_cat.view(torch.int8), wp.view(torch.int8), SF1.reshape(-1, 4), SF2.reshape(-1, 4), SFB, alpha, C)
print(json.dumps({"real_layer_end_to_end_rel_err": round(((C - ref).norm() / ref.norm()).item(), 5)}), flush=True)

for (m, k) in [(512, 5120), (2048, 5120), (8192, 5120), (8192, 17408)]:
    xx = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    t_fused = bench(lambda: fp4_residual_quant(xx))
    t_two = bench(lambda: (up_quant(xx), up_quant(xx)))
    print(json.dumps({"M": m, "K": k, "fused_2pass_quant_us": round(t_fused, 1), "upstream_1pass_x2_us": round(t_two, 1),
                      "fused_GBps_read": round(m * k * 2 / t_fused / 1e3)}), flush=True)
