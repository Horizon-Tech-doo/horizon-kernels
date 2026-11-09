# K3: fused two-pass NVFP4 activation quantizer for SM120 (feeds fp4_residual_gemm).
#   q1 = nvfp4(x);  r = x - dq(q1);  q2 = nvfp4(r)        (dq = exactly what the tensor core multiplies)
# One read of x; both passes computed in registers. Output layout matches fp4_residual_gemm:
#   A_cat (M, K) int8 = packed FP4 bytes, [:, :K/2] = q1, [:, K/2:] = q2  (i.e. (M, 2K) FP4 values)
#   SF1, SF2 (M, K/64) uint32 = four ue4m3 scale bytes per word, SM120 "blockscaled_chunk_kmajor" order.
# Derived from upstream TileLang's maint/gemm/gemm_sm120/tilelang_nvfp4_quantizer.py (MIT, Tile-AI).
import torch
import tilelang
import tilelang.language as T
from horizon_kernels._nvfp4_tl import (encode_ue4m3_ceil, decode_ue4m3, ue4m3_inverse, scale_source_coords,
                                       FP4_E2M1_MAX, CHUNK_ROWS, CHUNK_WORDS)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def _fp4_residual_quant_kernel(x, rows_per_cta: int = 32, num_threads: int = 512):
    M = T.dynamic("M")
    K = T.const("K")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    f32 = T.float32
    x: T.Tensor[(M, K), T.bfloat16]
    A_cat = T.empty((M, K), T.int8)
    SF1 = T.empty((M, K // 64), T.uint32)
    SF2 = T.empty((M, K // 64), T.uint32)

    with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        x_s = T.alloc_shared((rows_per_cta, 256), T.bfloat16)
        codes1 = T.alloc_shared((rows_per_cta, 16), T.int32)
        codes2 = T.alloc_shared((rows_per_cta, 16), T.int32)
        T.copy(x[pid_m * rows_per_cta, pid_k256 * 256], x_s, prefer_instruction="tma")
        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            grow = pid_m * rows_per_cta + row
            for k64 in T.serial(CHUNK_WORDS):
                elem = k64 * 64 + lane * 2
                byte = pid_k256 * 128 + k64 * 32 + lane
                group = k64 * 4 + (lane // 8)
                glane = lane % 8
                v0 = T.cast(x_s[row, elem], f32)
                v1 = T.cast(x_s[row, elem + 1], f32)
                # ---- pass 1: q1 = nvfp4(x) over this 16-value group (8 lanes x 2 values)
                am1 = T.alloc_var(f32, init=T.max(T.abs(v0), T.abs(v1)))
                am1 = T.max(am1, T.shfl_xor(am1, 4, width=8))
                am1 = T.max(am1, T.shfl_xor(am1, 2, width=8))
                am1 = T.max(am1, T.shfl_xor(am1, 1, width=8))
                code1 = T.alloc_var(T.int32, init=0)
                sc1 = T.alloc_var(f32, init=0.0)
                inv1 = T.alloc_var(f32, init=0.0)
                if glane == 0:
                    code1 = encode_ue4m3_ceil(am1 / FP4_E2M1_MAX)
                    sc1 = decode_ue4m3(code1)
                    inv1 = ue4m3_inverse(code1)
                code1 = T.shfl_sync(code1, 0, width=8)
                sc1 = T.shfl_sync(sc1, 0, width=8)
                inv1 = T.shfl_sync(inv1, 0, width=8)
                fa0 = T.cast(v0 * inv1, T.float4_e2m1fn)
                fa1 = T.cast(v1 * inv1, T.float4_e2m1fn)
                packed1 = T.reinterpret(fa0, T.uint8) | (T.reinterpret(fa1, T.uint8) << 4)
                # ---- residual, exactly what the tensor core will have missed
                r0 = v0 - T.cast(fa0, f32) * sc1
                r1 = v1 - T.cast(fa1, f32) * sc1
                # ---- pass 2: q2 = nvfp4(r)
                am2 = T.alloc_var(f32, init=T.max(T.abs(r0), T.abs(r1)))
                am2 = T.max(am2, T.shfl_xor(am2, 4, width=8))
                am2 = T.max(am2, T.shfl_xor(am2, 2, width=8))
                am2 = T.max(am2, T.shfl_xor(am2, 1, width=8))
                code2 = T.alloc_var(T.int32, init=0)
                inv2 = T.alloc_var(f32, init=0.0)
                if glane == 0:
                    code2 = encode_ue4m3_ceil(am2 / FP4_E2M1_MAX)
                    inv2 = ue4m3_inverse(code2)
                code2 = T.shfl_sync(code2, 0, width=8)
                inv2 = T.shfl_sync(inv2, 0, width=8)
                fb0 = T.cast(r0 * inv2, T.float4_e2m1fn)
                fb1 = T.cast(r1 * inv2, T.float4_e2m1fn)
                packed2 = T.reinterpret(fb0, T.uint8) | (T.reinterpret(fb1, T.uint8) << 4)
                if glane == 0:
                    codes1[row, group] = code1
                    codes2[row, group] = code2
                A_cat[grow, byte] = T.cast(packed1, T.int8)
                A_cat[grow, K // 2 + byte] = T.cast(packed2, T.int8)
        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            if lane < CHUNK_WORDS:
                w1 = (T.cast(codes1[row, lane * 4], T.uint32) | (T.cast(codes1[row, lane * 4 + 1], T.uint32) << 8)
                      | (T.cast(codes1[row, lane * 4 + 2], T.uint32) << 16) | (T.cast(codes1[row, lane * 4 + 3], T.uint32) << 24))
                w2 = (T.cast(codes2[row, lane * 4], T.uint32) | (T.cast(codes2[row, lane * 4 + 1], T.uint32) << 8)
                      | (T.cast(codes2[row, lane * 4 + 2], T.uint32) << 16) | (T.cast(codes2[row, lane * 4 + 3], T.uint32) << 24))
                prow, pword = scale_source_coords(pid_m * rows_per_cta + row, pid_k256 * CHUNK_WORDS + lane, K // 64)
                SF1[prow, pword] = w1
                SF2[prow, pword] = w2

    return A_cat, SF1, SF2


def fp4_residual_quant(x: torch.Tensor):
    """x: (M, K) bf16, M % 128 == 0, K % 256 == 0 -> (A_cat int8 (M, K), SF1, SF2 uint32 (M, K/64))."""
    assert x.dtype == torch.bfloat16 and x.ndim == 2
    assert x.shape[0] % CHUNK_ROWS == 0 and x.shape[1] % 256 == 0, x.shape
    return _fp4_residual_quant_kernel(x.contiguous())
