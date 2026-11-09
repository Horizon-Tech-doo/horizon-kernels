# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K3 residual two-pass FP4 GEMM: correctness vs separate passes on a real layer, then speed vs Marlin per shape.
import sys, json, types, torch
sys.path.insert(0, "$HK_WORK/tl/k3/upstream"); sys.path.insert(0, "$HK_WORK/tl/k3")
import tilelang.language as T
from safetensors import safe_open
from horizon_kernels.fp4_residual_gemm import fp4_residual_gemm
from sm120_nvfp4_blockscaled_gemm import sm120_nvfp4_blockscaled_gemm, swizzle_blockscaled_chunk_kmajor_scale_words, _pack_scale_words, _make_packed_fp4, _make_binary_scale_words
from maint.gemm.gemm_sm120.tilelang_nvfp4_quantizer import tilelang_quantize_bf16_to_nvfp4_blockscaled
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_fp4_layer_for_marlin, apply_fp4_marlin_linear


def bench(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    best = 1e9
    for _ in range(5):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best * 1e3


# --- correctness on a real layer
with safe_open("$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors", "pt", device="cuda") as f:
    pre = "model.language_model.layers.3.mlp.gate_proj."
    wp = f.get_tensor(pre + "weight_packed").view(torch.int8); ws = f.get_tensor(pre + "weight_scale")
    wgs = f.get_tensor(pre + "weight_global_scale").float()
N, K = wp.shape[0], wp.shape[1] * 2; M = 512
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
A1, S1 = tilelang_quantize_bf16_to_nvfp4_blockscaled(x)
A2, S2 = tilelang_quantize_bf16_to_nvfp4_blockscaled((x * 0.1).contiguous())  # any second operand works for equivalence
SFB = swizzle_blockscaled_chunk_kmajor_scale_words(_pack_scale_words(ws.view(torch.uint8))).reshape(-1, 4)
single = sm120_nvfp4_blockscaled_gemm(M, N, K, out_dtype=T.float32)
C1 = torch.empty(M, N, device="cuda"); C2 = torch.empty(M, N, device="cuda")
single(A1, wp, S1.reshape(-1, 4), SFB, C1); single(A2, wp, S2.reshape(-1, 4), SFB, C2)
fused = fp4_residual_gemm(M, N, K, out_dtype=T.float32)
Cf = torch.empty(M, N, device="cuda")
alpha = (1.0 / wgs).reshape(1).float()
A_cat = torch.cat([A1, A2], dim=1).contiguous()
fused(A_cat, wp, S1.reshape(-1, 4), S2.reshape(-1, 4), SFB, alpha, Cf)
ref = (C1 + C2) / wgs
print(json.dumps({"fused_vs_separate_rel": ((Cf - ref).norm() / ref.norm()).item()}), flush=True)

# --- speed per shape
SHAPES = {"gdn.in_proj_qkvz": (16384, 5120), "attn.qkv_proj": (14336, 5120), "o/out_proj": (5120, 6144),
          "mlp.gate_up": (34816, 5120), "mlp.down": (5120, 17408)}
for name, (N, K) in SHAPES.items():
    l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None,
                              weight=torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda"),
                              weight_scale=(torch.rand(N, K // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn),
                              weight_global_scale=torch.tensor(1.0 / 448, device="cuda"))
    prepare_fp4_layer_for_marlin(l)
    B = _make_packed_fp4(N, K, seed=1)
    SB = swizzle_blockscaled_chunk_kmajor_scale_words(_make_binary_scale_words(N, K, seed=2)).reshape(-1, 4)
    for M in (128, 512, 2048, 8192):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        A = _make_packed_fp4(M, K, seed=3)
        Acat = torch.cat([A, A], dim=1).contiguous()
        SA = swizzle_blockscaled_chunk_kmajor_scale_words(_make_binary_scale_words(M, K, seed=4)).reshape(-1, 4)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        k1 = sm120_nvfp4_blockscaled_gemm(M, N, K); k2 = fp4_residual_gemm(M, N, K)
        t1 = bench(lambda: k1(A, B, SA, SB, C))
        t2 = bench(lambda: k2(Acat, B, SA, SA, SB, alpha, C))
        tq = bench(lambda: tilelang_quantize_bf16_to_nvfp4_blockscaled(x))
        tm = bench(lambda: apply_fp4_marlin_linear(x, l.weight, l.weight_scale, l.weight_global_scale, l.workspace, N, K))
        print(json.dumps({"shape": name, "M": M, "marlin_us": round(tm, 1), "w4a4_1pass_us": round(t1, 1),
                          "residual_2pass_us": round(t2, 1), "quantizer_us_x1": round(tq, 1),
                          "2pass_total_est_us": round(t2 + 2 * tq, 1), "speedup_vs_marlin": round(tm / (t2 + 2 * tq), 2)}), flush=True)
    del l; torch.cuda.empty_cache()
