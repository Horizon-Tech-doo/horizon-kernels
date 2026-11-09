# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 candidate "fp4x3": decode GEMM (1 <= M <= 64) on the SM120 native block-scaled FP4 tensor cores.

y = x @ W^T with W in NVFP4 (checkpoint layout) and x (bf16) re-expressed as P = 3 or 4 residual NVFP4 passes, so the
MMA (``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X``, SASS OMMA.SF.16864) applies both operands' UE4M3
block scales itself: the weights go as packed 4-bit data from DRAM (TMA) straight into the tensor cores, with ZERO
dequantisation instructions.  Weight bytes read are identical to Marlin's (packed codes + ue4m3 scales, once).

Accuracy (rel. Frobenius error vs the exact fp32 reference, randn activations): 3 passes 0.00182, 4 passes 0.00166
(bf16 output rounding alone is 0.00165; the quantisation-only error is 0.00076 for 3 passes, 0.00005 for 4).

Design (see decode_contract.py for the interface):

* Passes folded into M.  The P residual passes of token m become P rows of the MMA A operand (row = P*m + p), so one
  row tile carries all passes of several tokens: M<=8 -> 32 rows (P=4), M<=10 -> 32 (P=3), M<=16 -> 64 (P=4), M<=21 ->
  64 (P=3), M<=32 -> 128 (P=4), M<=42 -> 128 (P=3), M<=48 -> 192 (P=4), M<=64 -> 192 (P=3).  Each weight tile is read
  from DRAM once per K tile and multiplied once against all (token, pass) rows; the passes are summed in the fp32
  epilogue (per-row 1/rs, per-column alpha).
* Scaling.  Each (token, 256-wide K chunk c) gets a power-of-two scale rs = 2**e_c putting its max in (1300, 2600]
  (block scales then use the full ue4m3 range); pass p is scaled by a further 4**p so the residuals never underflow
  (pass-p block scales stay <= 448).  The accumulator is kept "in units of the current chunk": entering chunk c every
  row is multiplied by RR = 2**(e_c - e_{c-1}) (exact), and the epilogue multiplies by RI = 2**-(e_last + 2p).  So the
  quantiser needs no row-wide reduction and runs as one small, fully parallel kernel (one warp per token x chunk).
* GEMM kernel: grid (N/64, S), one CTA per SM (73-91 KB smem, as many stages as fit).  Warp-specialised: a TMA-only
  producer (one elected thread, 5 async copies per 256-K stage: weights 64x128 B and activations Rx128 B as 2D TMA,
  weight scales 1 KB, activation scales R*16 B and RR R*4 B as 1D bulk copies -- no per-thread cp.async, hence no
  proxy fence / MEMBAR per stage) and 2-4 consumer warps running the SM120 full-tile block-scaled MMA micro-kernel.
  B (N, K/2) stays in the checkpoint layout; the weight scales are re-laid-out once in prepare() into per-(64-column
  tile, K tile) 1 KB blocks in the tile-local K-major order the MMA lowering reads.
* Kernel chaining (PDL).  The quantiser triggers its dependents at start; the GEMM launches early, warms L2 with its
  first ``num_stages`` weight tiles (cp.async.bulk.prefetch.L2, weights only), then griddepcontrol.wait()s for the
  quantiser before touching activations.  The split-K reduce is also a PDL dependent.
* Split-K only where the N tiles alone do not cover the SMs (the N=5120 layers: o_proj S=3, down S=4); fp32 partials
  (S, M, N) are summed by a small kernel in fixed order s = 0..S-1 (deterministic, no atomics).  S=1 layers write
  bf16 directly from the GEMM epilogue.

Kernels per call: quantiser + GEMM (+ reduce when S > 1).  All buffers are torch.empty (CUDA-graph capturable).
"""

from __future__ import annotations

import torch

import tilelang
import tilelang.language as T


SUPPORTS_M = 64

CHUNK = 256            # K per pipeline tile == K per activation row-scale chunk
WORDS = CHUNK // 64    # uint32 scale words per row per K tile
BLOCK_N = 64           # output columns per CTA (weight scale layout in prepare() is tied to it)
_BUCKETS = (32, 64, 128, 192)
_MAX_PASSES = 4
_ROW_TARGET = 2600.0   # chunk max * rs in (1300, 2600]  ->  pass-1 block scale <= ceil_ue4m3(2600/6) = 448
_E_CLAMP = 30          # |log2 rs| <= 30 keeps the running-scale accumulator far from fp32 overflow
_MIN_K_TILES_PER_SPLIT = 3

# per bucket: consumer threads (+ one 128-thread TMA producer warpgroup), pipeline stages, warp policy.
# smem/stage = R*(128 + 16 + 4) + 64*(128 + 16) bytes; one CTA per SM, as many stages as fit in 99 KB.
# (2 CTAs/SM for R=32 -- 3 stages each, 192 threads at <= 168 registers -- hung on multi-wave grids with this
#  TMA-only warp-specialised producer on the 570 driver + CUDA 13 forward-compat stack; 1 CTA/SM with the same
#  per-SM bytes in flight is used instead.)
_GEMM_CFG = {
    32: dict(threads=64, num_stages=6, policy=T.GemmWarpPolicy.FullCol),    # 13.6 KB/stage -> 81.8 KB
    64: dict(threads=128, num_stages=5, policy=T.GemmWarpPolicy.Square),    # 18.3 KB/stage -> 91.3 KB
    128: dict(threads=128, num_stages=3, policy=T.GemmWarpPolicy.Square),   # 27.5 KB/stage -> 82.5 KB
    192: dict(threads=128, num_stages=2, policy=T.GemmWarpPolicy.Square),   # 36.8 KB/stage -> 73.5 KB
}

_PC_SIMT = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}
_PC_GEMM = {tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True}

# Quantiser helpers.  fp4x3_pass8: one residual NVFP4 pass over a lane's 8 values (half a 16-group) given the group max
# gm (already reduced over the lane pair): ue4m3 scale = ceil(gm / 6) on the ue4m3 grid (hardware RN convert, then one
# step up if it landed below), 8 E2M1 codes via packed cvt (x -> low nibble, i.e. even k in the low nibble), and the
# residual 4 * (u - q * scale) for the next pass with a single rounding (q * scale is exact).  Returns the scale code.
_QUANT_SOURCE = r"""
#include <cuda_fp8.h>
#include <cuda_fp4.h>
__device__ __forceinline__ unsigned fp4x3_pass8(float *u, float gm, unsigned *packed) {
  const float v = fminf(gm * (1.0f / 6.0f), 448.0f);
  unsigned c = (unsigned)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
  if (__half2float(*reinterpret_cast<__half *>(&hr)) < v) c += 1u;
  hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)c, __NV_E4M3);
  const float sc = __half2float(*reinterpret_cast<__half *>(&hr));
  const float inv = sc > 0.0f ? __frcp_rn(sc) : 0.0f;
  const float s4 = -4.0f * sc;
  unsigned w = 0u;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_fp4x2_storage_t q =
        __nv_cvt_float2_to_fp4x2(make_float2(u[2 * i] * inv, u[2 * i + 1] * inv), __NV_E2M1, cudaRoundNearest);
    w |= (unsigned)q << (8 * i);
    const __half2_raw qh = __nv_cvt_fp4x2_to_halfraw2(q, __NV_E2M1);
    const float2 qf = __half22float2(*reinterpret_cast<const __half2 *>(&qh));
    u[2 * i] = fmaf(qf.x, s4, 4.0f * u[2 * i]);
    u[2 * i + 1] = fmaf(qf.y, s4, 4.0f * u[2 * i + 1]);
  }
  *packed = w;
  return c;
}
// 8 consecutive bf16 (16-byte aligned) -> fp32
__device__ __forceinline__ void fp4x3_load8(const void *p, float *u) {
  const uint4 v = *reinterpret_cast<const uint4 *>(p);
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    u[2 * i] = __uint_as_float(w[i] << 16);
    u[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}
"""

# L2 warm-up of weight tiles (sm_90+ bulk prefetch; size multiple of 16 B, address 16-B aligned)
_PREFETCH_SOURCE = r"""
__device__ __forceinline__ void fp4x3_prefetch_l2(const void *ptr, unsigned int bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(ptr), "r"(bytes) : "memory");
}
"""


def _chunk_exp(am):
    """floor(log2(2600 / am)) clamped to +-30 (0 for an all-zero chunk): exponent of the chunk's power-of-two scale."""
    tq = T.float32(_ROW_TARGET) / T.max(am, T.float32(1e-30))
    e = T.cast((T.reinterpret(tq, T.uint32) >> 23) & T.uint32(0xFF), T.int32) - 127
    return T.if_then_else(am > T.float32(0.0), T.min(T.max(e, -_E_CLAMP), _E_CLAMP), 0)


def _pow2(e):
    """2**e as fp32 for an int32 exponent expression in the normal range."""
    return T.reinterpret(T.cast((e + 127) << 23, T.uint32), T.float32)


@tilelang.jit(pass_configs=_PC_SIMT)
def fp4x3_quant_kernel(x, A32, SF8, RR, RI, R: int = 32, passes: int = 4, warps: int = 4):
    """x (M, K) bf16 -> folded residual NVFP4 activations for one GEMM row tile of R rows (row = passes*m + p).

    A32 (R, K/8) int32   packed E2M1 codes (low nibble = even k)
    SF8 (K/16 * R,) u8   ue4m3 block scales in the tile-local K-major order of an R-row MMA tile:
                         byte 4 * (word * R + (r % 32) * (R / 32) + r // 32) + (k16 % 4), word = k // 64
    RR (K/256, R)        accumulator re-scale entering chunk c: 2**(e_c - e_{c-1})  (the pass offset cancels; for
                         c = 0 the value only ever multiplies a zero accumulator)
    RI (K/256, R)        2**-(e_c + 2p): turns an accumulator held in chunk-c units back into true units
    Rows of token slots m >= M are written as zeros (so the GEMM never reads uninitialised memory).
    One warp per token slot and 256-wide chunk; each lane owns 8 consecutive k (half a 16-group).  The warp also
    re-derives the previous chunk's exponent (max is exact and order-free, so it matches that chunk's own warp)."""
    M = T.dynamic("M")
    K = T.dynamic("K")
    f32 = T.float32
    x: T.Tensor[(M, K), T.bfloat16]
    A32: T.Tensor[(R, K // 8), T.int32]
    SF8: T.Tensor[(K // 16 * R,), T.uint8]
    RR: T.Tensor[(K // 256, R), f32]
    RI: T.Tensor[(K // 256, R), f32]
    slots = (R + passes - 1) // passes
    rg = R // 32

    with T.Kernel(T.ceildiv(slots, warps), K // 256, threads=32 * warps) as (pid_t, pid_c):
        T.import_source(_QUANT_SOURCE)
        T.pdl_trigger()        # let the GEMM launch now: it warms L2 with weights, then waits for this grid
        T.pdl_sync()           # x may come from the previous kernel
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        m = pid_t * warps + warp
        u = T.alloc_local((8,), f32)
        up = T.alloc_local((8,), f32)
        pk = T.alloc_local((1,), T.uint32)
        am = T.alloc_var(f32, init=0.0)
        ap = T.alloc_var(f32, init=0.0)
        ex = T.alloc_var(T.int32, init=0)
        ep = T.alloc_var(T.int32, init=0)
        gm = T.alloc_var(f32, init=0.0)
        code = T.alloc_var(T.uint32, init=0)
        if m < slots:
            # token slots m >= M are padding: all-zero rows (warp-uniform branch; x is never read past row M-1)
            for e in T.unroll(8):
                u[e] = T.float32(0.0)
                up[e] = T.float32(0.0)
            if m < M:
                T.call_extern("handle", "fp4x3_load8", T.address_of(x[m, pid_c * 256 + lane * 8]),
                              T.access_ptr(u, "w"))
                if pid_c > 0:
                    T.call_extern("handle", "fp4x3_load8", T.address_of(x[m, (pid_c - 1) * 256 + lane * 8]),
                                  T.access_ptr(up, "w"))
            am = T.float32(0.0)
            ap = T.float32(0.0)
            for e in T.unroll(8):
                am = T.max(am, T.abs(u[e]))
                ap = T.max(ap, T.abs(up[e]))
            for s in T.unroll(5):
                am = T.max(am, T.shfl_xor(am, 16 >> s))
                ap = T.max(ap, T.shfl_xor(ap, 16 >> s))
            # rs = 2**e, e = floor(log2(2600 / chunk_max)) clamped; exact power of two
            ex = _chunk_exp(am)
            ep = _chunk_exp(ap)
            for e in T.unroll(8):
                u[e] = u[e] * _pow2(ex)
            for p in T.unroll(passes):
                r = m * passes + p
                # block scale of this pass over the 16-group held by lanes (2g, 2g+1)
                gm = T.float32(0.0)
                for e in T.unroll(8):
                    gm = T.max(gm, T.abs(u[e]))
                gm = T.max(gm, T.shfl_xor(gm, 1))
                # codes, scale and residual (in the next pass's units, x4) for this lane's 8 values
                code = T.call_extern("uint32", "fp4x3_pass8", T.access_ptr(u, "rw"), gm, T.access_ptr(pk, "w"))
                if r < R:
                    A32[r, pid_c * 32 + lane] = T.reinterpret(pk[0], T.int32)
                    if lane % 2 == 0:
                        flat = (pid_c * 4 + lane // 8) * R + (r % 32) * rg + r // 32
                        SF8[flat * 4 + (lane // 2) % 4] = T.cast(code, T.uint8)
                    if lane == 0:
                        RR[pid_c, r] = _pow2(ex - ep)
                        RI[pid_c, r] = _pow2(-ex - 2 * p)


def _pass_sum(C_s, RI, kl, t, j, passes):
    """sum_p C_s[t*passes + p, j] * RI[kl, t*passes + p] as one expression (fixed order p = 0..passes-1)."""
    v = C_s[t * passes, j] * RI[kl, t * passes]
    for p in range(1, passes):
        v = v + C_s[t * passes + p, j] * RI[kl, t * passes + p]
    return v


@tilelang.jit(pass_configs=_PC_GEMM)
def fp4x3_gemm_kernel(A, B, Bb, SFA, SFB, RR, RI, alpha, Y, P, R: int = 32, passes: int = 4, threads: int = 64,
                      num_stages: int = 3, policy=T.GemmWarpPolicy.FullCol, pdl: int = 1, l2pf: int = 1):
    """Y (M, N) = alpha[n] * sum_p RI * (A_p @ B^T) on the block-scaled FP4 MMA; S = P.shape[0] K splits.

    Grid (N / 64, S).  Each CTA streams its K range in 256-wide tiles with a TMA-only producer (5 boxes per stage):
    packed weights (64 x 128 B), weight scales (64 x 16 B), folded activation rows (R x 128 B), their scales (R x 16 B)
    and the per-row chunk re-scale RR (R x 4 B).  No per-thread cp.async, so no proxy fence / MEMBAR per stage.
    S == 1: bf16 result straight into Y.  S > 1: fp32 partial into P[s] (reduced by fp4x3_splitk_reduce_kernel).
    Launched as a PDL dependent of the quantiser: before the grid-dependency wait it only touches the (immutable)
    weights, warming L2 with its first ``num_stages`` weight tiles (Bb is an int8 view of B for the addresses)."""
    M = T.dynamic("M")
    N = T.dynamic("N")
    S = T.dynamic("S")
    PM = T.dynamic("PM")
    PN = T.dynamic("PN")
    K = T.const("K")
    fp4 = T.float4_e2m1fn
    f32 = T.float32
    KT = K // CHUNK
    A: T.Tensor[(R, K), fp4]
    B: T.Tensor[(N, K), fp4]
    Bb: T.Tensor[(N, K // 2), T.int8]
    SFA: T.Tensor[(KT * R, WORDS), T.uint32]
    SFB: T.Tensor[(N // BLOCK_N * KT * BLOCK_N, WORDS), T.uint32]
    RR: T.Tensor[(KT, R), f32]
    RI: T.Tensor[(KT, R), f32]
    alpha: T.Tensor[(N,), f32]
    Y: T.Tensor[(M, N), T.bfloat16]
    P: T.Tensor[(S, PM, PN), f32]
    tok = R // passes

    with T.Kernel(N // BLOCK_N, S, threads=threads) as (bx, bk):
        T.import_source(_PREFETCH_SOURCE)
        A_s = T.alloc_shared((R, CHUNK), fp4)
        B_s = T.alloc_shared((BLOCK_N, CHUNK), fp4)
        SFA_s = T.alloc_shared((R, WORDS), T.uint32)
        SFB_s = T.alloc_shared((BLOCK_N, WORDS), T.uint32)
        RR_s = T.alloc_shared((1, R), f32)
        C_l = T.alloc_fragment((R, BLOCK_N), f32)
        C_s = T.alloc_shared((R, BLOCK_N), f32)
        kt_per = KT // S
        k0 = bk * kt_per
        # (warp/lane ids, not the thread binding: the simplifier assumes the pre-warp-specialisation thread extent
        #  there, while this prologue runs on producer + consumer threads)
        wid = T.get_warp_idx_sync()
        lid = T.get_lane_idx()
        if pdl:
            T.pdl_trigger()
        pf = T.min(kt_per, num_stages)
        if l2pf and wid < BLOCK_N // 32:
            T.call_extern("handle", "fp4x3_prefetch_l2",
                          T.address_of(Bb[bx * BLOCK_N + wid * 32 + lid, k0 * (CHUNK // 2)]),
                          T.cast(pf * (CHUNK // 2), T.uint32))
        if l2pf and wid == BLOCK_N // 32 and lid == 0:
            T.call_extern("handle", "fp4x3_prefetch_l2", T.address_of(SFB[(bx * KT + k0) * BLOCK_N, 0]),
                          T.cast(pf * (BLOCK_N * WORDS * 4), T.uint32))
        if pdl:
            T.pdl_sync()
        T.clear(C_l)
        for ko in T.Pipelined(kt_per, num_stages=num_stages):
            kt = k0 + ko
            T.copy(A[0, kt * CHUNK], A_s)
            T.copy(B[bx * BLOCK_N, kt * CHUNK], B_s)
            T.copy(SFA[kt * R, 0], SFA_s)
            T.copy(SFB[(bx * KT + kt) * BLOCK_N, 0], SFB_s)
            T.copy(RR[kt, 0], RR_s)
            # move the accumulator into this chunk's units: * rs[c] / rs[c-1] (exact powers of two)
            for i, j in T.Parallel(R, BLOCK_N):
                C_l[i, j] = C_l[i, j] * RR_s[0, i]
            T.mma_gemm_blockscaled(A_s, B_s, C_l, SFA_s, SFB_s, transpose_B=True, policy=policy, clear_accum=False,
                                   k_start=0, sf_a_granularity_k=16, sf_b_granularity_k=16,
                                   sf_layout="blockscaled_chunk_kmajor")
        T.copy(C_l, C_s)
        kl = k0 + kt_per - 1
        for t, j in T.Parallel(tok, BLOCK_N):
            if t < M:
                v = _pass_sum(C_s, RI, kl, t, j, passes) * alpha[bx * BLOCK_N + j]
                if S == 1:
                    Y[t, bx * BLOCK_N + j] = T.cast(v, T.bfloat16)
                else:
                    P[bk, t, bx * BLOCK_N + j] = v


@tilelang.jit(pass_configs=_PC_SIMT)
def fp4x3_splitk_reduce_kernel(P, Y):
    """Y (M, N) bf16 = sum_{s = 0..S-1} P[s] in fixed order (deterministic)."""
    S = T.dynamic("S")
    M = T.dynamic("M")
    N = T.dynamic("N")
    P: T.Tensor[(S, M, N), T.float32]
    Y: T.Tensor[(M, N), T.bfloat16]
    with T.Kernel(T.ceildiv(N, 256), M, threads=256) as (bx, by):
        T.pdl_sync()
        tx = T.get_thread_binding()
        n = bx * 256 + tx
        acc = T.alloc_var(T.float32, init=0.0)
        if n < N:
            for s in T.serial(S):
                acc = acc + P[s, by, n]
            Y[by, n] = T.cast(acc, T.bfloat16)


# ---------------------------------------------------------------------------------------------------------------
# host side


def _pack_scale_words(scale_fp8):
    b = scale_fp8.view(torch.uint8).to(torch.int64).reshape(scale_fp8.shape[0], -1, 4)
    return (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)).to(torch.uint32).contiguous()


def _choose_split(n_tiles, k_tiles, n_sms):
    """K splits S (S | k_tiles): none once the N tiles alone cover every SM (one deep-pipelined CTA per SM streams
    at full rate, so split-K would only add the partial round trip and a reduce kernel); otherwise the smallest S
    with n_tiles * S >= n_sms, keeping >= 3 K tiles per split."""
    best = 1
    for d in range(1, k_tiles + 1):
        if k_tiles % d:
            continue
        if k_tiles // d < _MIN_K_TILES_PER_SPLIT:
            break
        best = d
        if n_tiles * d >= n_sms:
            break
    return best


def prepare(packed_int8, scale_fp8, alpha_fp32):
    """Checkpoint NVFP4 tensors -> kernel operands. Weights untouched; scales re-laid-out per (64-col tile, K tile)."""
    N, K = packed_int8.shape[0], packed_int8.shape[1] * 2
    assert N % BLOCK_N == 0 and K % CHUNK == 0, (N, K)
    words = _pack_scale_words(scale_fp8)                                   # (N, K/64) semantic words
    # tile-local K-major order of a 64-row MMA tile: [n_tile][k64 word][row % 32][row // 32]
    sfb = words.reshape(N // BLOCK_N, 2, 32, K // 64).permute(0, 3, 2, 1).contiguous().reshape(-1, WORDS)
    k_tiles = K // CHUNK
    split = _choose_split(N // BLOCK_N, k_tiles, torch.cuda.get_device_properties(packed_int8.device).multi_processor_count)
    return {
        "B": packed_int8.view(torch.int8).contiguous(),
        "SFB": sfb,
        "alpha": alpha_fp32.float().contiguous(),
        "N": N, "K": K, "split": split,
        "dummy": torch.zeros(1, 1, 1, device=packed_int8.device, dtype=torch.float32),
    }


def _bucket(m):
    rows = 3 * m
    for R in _BUCKETS:
        if rows <= R:
            return R, min(_MAX_PASSES, R // m)
    raise ValueError(f"fp4x3 decode GEMM supports M <= {SUPPORTS_M}, got {m}")


def linear(x, prep):
    """y (M, N) bf16 = x (M, K) bf16 @ W^T, M <= 64."""
    M, K = x.shape
    N = prep["N"]
    assert K == prep["K"], (K, prep["K"])
    R, passes = _bucket(M)
    dev = x.device
    x = x.contiguous()
    if x.data_ptr() % 16:        # the quantiser reads x with 16-byte vector loads (host-side check, capture-safe)
        x = x.clone()
    A = torch.empty(R, K // 2, dtype=torch.int8, device=dev)
    SFA = torch.empty(K // 64 * R, dtype=torch.uint32, device=dev)
    RR = torch.empty(K // CHUNK, R, dtype=torch.float32, device=dev)
    RI = torch.empty(K // CHUNK, R, dtype=torch.float32, device=dev)
    fp4x3_quant_kernel(x, A.view(torch.int32), SFA.view(torch.uint8), RR, RI, R=R, passes=passes)
    S = prep["split"]
    Y = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
    P = torch.empty(S, M, N, dtype=torch.float32, device=dev) if S > 1 else prep["dummy"]
    cfg = _GEMM_CFG[R]
    fp4x3_gemm_kernel(A, prep["B"], prep["B"], SFA.view(K // CHUNK * R, WORDS), prep["SFB"], RR, RI, prep["alpha"], Y, P,
                      R=R, passes=passes, **cfg)
    if S > 1:
        fp4x3_splitk_reduce_kernel(P, Y)
    return Y
