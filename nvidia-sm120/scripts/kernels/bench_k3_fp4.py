# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K3 baseline: upstream TileLang native NVFP4 block-scaled GEMM (W4A4, mma kind::mxf4nvf4) vs vLLM Marlin (W4A16)
# vs cuBLAS BF16, on Qwen3.8-27B's fused layer shapes, from decode batches (M=32) to long prefills (M=8192).
# GEMM time only: W4A4 additionally needs an activation->FP4 quantisation step, measured separately later.
import json, sys, types, torch
sys.path.insert(0, "$HK_WORK/tl/k3")
from sm120_nvfp4_blockscaled_gemm import sm120_nvfp4_blockscaled_gemm, swizzle_blockscaled_chunk_kmajor_scale_words, _make_packed_fp4, _make_binary_scale_words
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
    return best * 1e3  # us


def marlin_layer(N, K):
    l = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K, params_dtype=torch.bfloat16, bias=None)
    l.weight = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    l.weight_scale = (torch.rand(N, K // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
    l.weight_global_scale = torch.tensor(1.0 / 448, device="cuda")
    prepare_fp4_layer_for_marlin(l)
    return l


SHAPES = {"gdn.in_proj_qkvz": (16384, 5120), "attn.qkv_proj": (14336, 5120), "o/out_proj": (5120, 6144),
          "mlp.gate_up": (34816, 5120), "mlp.down": (5120, 17408)}
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32,64,128,512,2048,8192").split(",")]
for name, (N, K) in SHAPES.items():
    ml = marlin_layer(N, K)
    Wb = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    B = _make_packed_fp4(N, K, seed=1)
    SFB = swizzle_blockscaled_chunk_kmajor_scale_words(_make_binary_scale_words(N, K, seed=2)).reshape(-1, 256 // 64)
    for M in MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        A = _make_packed_fp4(M, K, seed=3)
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(_make_binary_scale_words(M, K, seed=4)).reshape(-1, 256 // 64)
        C = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        k = sm120_nvfp4_blockscaled_gemm(M, N, K)
        t_fp4 = bench(lambda: k(A, B, SFA, SFB, C))
        t_marlin = bench(lambda: apply_fp4_marlin_linear(x, ml.weight, ml.weight_scale, ml.weight_global_scale, ml.workspace, N, K))
        t_bf16 = bench(lambda: x @ Wb.T)
        fl = 2 * M * N * K
        print(json.dumps({"shape": name, "M": M, "fp4_tl_us": round(t_fp4, 1), "marlin_us": round(t_marlin, 1), "bf16_us": round(t_bf16, 1),
                          "fp4_TFLOPS": round(fl / t_fp4 / 1e6), "marlin_TFLOPS": round(fl / t_marlin / 1e6),
                          "fp4_vs_marlin": round(t_marlin / t_fp4, 2)}), flush=True)
    del ml, Wb; torch.cuda.empty_cache()
