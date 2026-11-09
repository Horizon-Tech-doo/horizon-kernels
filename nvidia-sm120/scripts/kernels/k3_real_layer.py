# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K3 numerics on REAL checkpoint weights: native W4A4 (TileLang quantizer + FP4 MMA) vs Marlin W4A16,
# both against the exact dequantised-weight reference y = x @ (fp4 * scale_fp8 / weight_global_scale)^T.
import sys, json, torch
sys.path.insert(0, "$HK_WORK/tl/k3/upstream"); sys.path.insert(0, "$HK_WORK/tl/k3")
import types
import tilelang.language as T
from safetensors import safe_open
from sm120_nvfp4_blockscaled_gemm import sm120_nvfp4_blockscaled_gemm, swizzle_blockscaled_chunk_kmajor_scale_words, _pack_scale_words, _decode_rowmajor_fp4
from maint.gemm.gemm_sm120.tilelang_nvfp4_quantizer import tilelang_quantize_bf16_to_nvfp4_blockscaled
from examples.dequantize_gemm.quantize.nvfp4 import unswizzle_blockscaled_chunk_kmajor_scale_words, decode_ue4m3_scale_bytes, decode_packed_fp4_e2m1


def dequant_act(A, SFA_words, K):
    # inverse of the TileLang quantizer: fp4 codes x per-16 ue4m3 scales -> float
    w = unswizzle_blockscaled_chunk_kmajor_scale_words(SFA_words).to(torch.int64)
    b = torch.stack([(w >> s) & 0xFF for s in (0, 8, 16, 24)], -1).reshape(w.shape[0], -1).to(torch.uint8)
    return decode_packed_fp4_e2m1(A, K).float() * decode_ue4m3_scale_bytes(b).float().repeat_interleave(16, 1)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear

CK = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
P = "model.language_model.layers.%d.%s."
LAYERS = [(3, "mlp.gate_proj"), (3, "mlp.down_proj"), (0, "linear_attn.in_proj_qkv"), (3, "self_attn.o_proj")]
M = int(sys.argv[1]) if len(sys.argv) > 1 else 512
torch.manual_seed(0)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


with safe_open(CK, "pt", device="cuda") as f:
    for li, name in LAYERS:
        pre = P % (li, name)
        wp = f.get_tensor(pre + "weight_packed"); ws = f.get_tensor(pre + "weight_scale"); wgs = f.get_tensor(pre + "weight_global_scale").float()
        N, K = wp.shape[0], wp.shape[1] * 2
        # exact reference weight
        Wdq = _decode_rowmajor_fp4(wp.view(torch.int8), N, K) * ws.float().repeat_interleave(16, 1) / wgs
        # activations: normal with a few outlier channels (LLM-like)
        x = torch.randn(M, K, device="cuda")
        x[:, torch.randint(0, K, (8,))] *= 20
        x = x.to(torch.bfloat16)
        ref = x.float() @ Wdq.T
        # Marlin W4A16 with the real checkpoint tensors
        l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None,
                                  weight=wp.clone(), weight_scale=ws.clone(), weight_global_scale=(1.0 / wgs).clone())  # vLLM inverts it
        prepare_fp4_layer_for_marlin(l)
        y_marlin = apply_fp4_marlin_linear(x, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K)
        # Native W4A4: quantise activations, repack weight scales into the SM120 chunk layout
        A, SFA_w = tilelang_quantize_bf16_to_nvfp4_blockscaled(x)
        SFA = SFA_w.reshape(-1, 256 // 64)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(_pack_scale_words(ws.view(torch.uint8))).reshape(-1, 256 // 64)
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)
        k = sm120_nvfp4_blockscaled_gemm(M, N, K, out_dtype=T.float32)
        k(A, wp.view(torch.int8), SFA, SFB, C)
        y_fp4 = C / wgs
        # residual two-pass FP4: quantise what the first pass missed, run it through the same FP4 weights
        r = (x.float() - dequant_act(A, SFA_w, K)).to(torch.bfloat16)
        A2, SFA2_w = tilelang_quantize_bf16_to_nvfp4_blockscaled(r)
        C2 = torch.empty_like(C)
        k(A2, wp.view(torch.int8), SFA2_w.reshape(-1, 256 // 64), SFB, C2)
        y_2pass = (C + C2) / wgs
        # simulated W4A8: activations per-row FP8 e4m3 (what a kind::f8f6f4 kernel would compute)
        sx = x.float().abs().amax(-1, keepdim=True) / 448.0
        y_w4a8 = ((x.float() / sx).to(torch.float8_e4m3fn).float() * sx) @ Wdq.T
        print(json.dumps({"layer": f"{li}.{name}", "N": N, "K": K, "M": M,
                          "marlin_w4a16_rel_err": round(rel(y_marlin, ref), 5),
                          "native_w4a4_rel_err": round(rel(y_fp4, ref), 5),
                          "sim_w4a8_rel_err": round(rel(y_w4a8, ref), 5),
                          "residual_2pass_fp4_rel_err": round(rel(y_2pass, ref), 5),
                          "w4a4_vs_marlin_rel": round(rel(y_fp4, y_marlin), 5)}), flush=True)
