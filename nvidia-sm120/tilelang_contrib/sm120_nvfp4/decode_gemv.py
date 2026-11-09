# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K2 candidate "gemv": CUDA-core NVFP4 W4A16 decode GEMV for tiny M (1..8 tokens), no tensor cores.

Contract: see decode_contract.py (prepare / linear / SUPPORTS_M).

Design
------
* One lane owns one weight row (n) and streams it along K in 16-byte chunks (32 FP4 codes) straight from global
  memory into registers. prepare() re-lays the checkpoint bytes out as [row group of 32][k chunk][lane][16 B], so one
  warp-wide LDG.128 reads 512 contiguous bytes; the UE4M3 scales go to [row group][128-k unit][lane][8 B] (one LDG.64
  per 4 chunks). Nibble order inside a word is the checkpoint's own (low nibble = even k), nothing is re-encoded.
* Register double buffer: iteration i+1 (4 x LDG.128 + 1 x LDG.64 per lane) is issued before iteration i is consumed,
  and the very first iteration is issued before the x prologue, so DRAM is busy from the first cycle.
* Dequant uses the SM120 hardware converters on whole words: cvt.rn.f16x2.e2m1x2 is F2FP.F16.E2M1.UNPACK_B with a
  byte selector (4 per 32-bit word, no extraction ops), cvt.rn.f16x2.e4m3x2 turns 2 block scales into f16x2.
* The FMA is the SM120 mixed-precision FHFMA (fma.rn.f32.f16: f16 x f16 -> f32 accumulate, half selectors on both
  operands), so the activations must be f16: each CTA converts its K slice of x (bf16) ONCE into shared memory as
  x' = x * 2^-e_m (e_m from the slice's per-row absmax: exact power-of-two prescale, so bf16 -> f16 is exact and
  nothing overflows), and undoes 2^e_m in fp32 at the end. All lanes of a warp read the same x' address
  (smem broadcast, 1 wavefront per LDS.128).
* MODE 0 "exact" (default for M <= 4): w * s formed in f16 by one HMUL2 per 2 weights (fp4 x ue4m3 has <= 6
  significant bits: exact), then one FHFMA per (weight, token): exact products, fp32 accumulation, i.e. the
  contract's "exact class" (rel. error = the final bf16 rounding).
  MODE 1 "hfma2" (default for M >= 5): per 16-k block and token, 8 HFMA2 build an f16x2 partial of raw fp4 * x'
  (products exact; x' prescaled to max |x'| < 2^10 so |partial| <= 8*6*1024 < 65504), then 2 FHFMA fold it into fp32
  with the block scale. Halves the per-token instruction count; adds ~5e-4 relative error (f16 rounding of 8-term
  partial sums), total stays ~0.0017 vs the 0.00249 limit.
* Split of work: CTA = 8 warps = WR row-warps x WK k-warps, tile = 32*WR rows x K/S. k-warps are reduced through
  shared memory in fixed order; S > 1 CTAs of one row tile write fp32 partials and the last-arriving CTA (int32 ticket,
  self-resetting) sums them in split order 0..S-1: deterministic, no fp32 atomics, CUDA-graph safe.
  config() picks (WR, S) per (N, K, M) for a single wave of equal CTAs (see _config).

Main-loop SASS per weight (thread instructions; counted on sm_120a, 2 iterations = 256 weights per lane):
  exact : 0.53 F2FP + 0.5 HMUL2 + 0.04 LDG + ~0.02 loop + M * (1 FHFMA + 0.125 LDS.128)
          = 2.24 (M=1), 3.35 (M=2), 4.46 (M=3), 5.6 (M=4), 10.1 (M=8)
  hfma2 : 0.53 F2FP + 0.04 LDG + ~0.03 + M * (0.5 HFMA2/HMUL2 + 0.125 FHFMA + 0.125 LDS.128)
          = 3.6 (M=4), 4.35 (M=5), 6.6 (M=8)
No spills (47..128 registers at 2 CTAs/SM, up to 241 at 1 CTA/SM).
"""

from dataclasses import dataclass

import functools
import os

import torch
import tilelang
import tilelang.language as T

SUPPORTS_M = 8

NUM_SMS = 188
THREADS = 256
WARPS = THREADS // 32
UNIT_K = 128                 # k per loop iteration per warp: 4 x 32-k chunks (4 LDG.128 weights + 1 LDG.64 scales)
EXACT, HFMA2 = 0, 1
# accumulation mode per token count; DECODE_GEMV_MODE=exact|hfma2 forces one mode for every M (for racing)
_FORCED_MODE = {"exact": EXACT, "hfma2": HFMA2}.get(os.environ.get("DECODE_GEMV_MODE", "").strip().lower())
MODE_BY_M = {m: (_FORCED_MODE if _FORCED_MODE is not None else (EXACT if m <= 4 else HFMA2))
             for m in range(1, SUPPORTS_M + 1)}
PRESCALE_EXP = {EXACT: 14, HFMA2: 9}      # per row slice, max |x'| lands in [2^T, 2^(T+1))
SMEM_2CTA = 48 * 1024                     # at most this much shared memory per CTA -> 2 CTAs per SM
SMEM_MAX = 99 * 1024

# (N, K, M) -> (WR, S) overrides; anything missing is chosen by _config()
CONFIG_OVERRIDE: dict[tuple[int, int, int], tuple[int, int]] = {}

# ---------------------------------------------------------------------------------------------------------------
# Device helpers (inlined into the TileLang kernel through T.Kernel(prelude=...)).
GV_SRC = r"""
// 8 FP4 (e2m1) codes of one 32-bit word -> 4 x f16x2 (byte b -> h_b, low nibble -> low half). F2FP with byte selects.
__device__ __forceinline__ void gv_fp4x8(unsigned w, unsigned &h0, unsigned &h1, unsigned &h2, unsigned &h3) {
  asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
      "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
      : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3) : "r"(w));
}
// two UE4M3 scale bytes (16-bit half `hi` of s) -> f16x2 (first byte -> low half)
__device__ __forceinline__ unsigned gv_e4m3x2(unsigned s, int hi) {
  unsigned r;
  if (hi) asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(r) : "r"(s));
  else    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(r) : "r"(s));
  return r;
}
// f16x2 whose both halves are half `hi` of v
__device__ __forceinline__ unsigned gv_bcast(unsigned v, int hi) {
  unsigned r;
  if (hi) asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tmov.b32 %0, {h, h};\n\t}" : "=r"(r) : "r"(v));
  else    asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tmov.b32 %0, {l, l};\n\t}" : "=r"(r) : "r"(v));
  return r;
}
__device__ __forceinline__ unsigned gv_hmul2(unsigned a, unsigned b) {
  unsigned r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
__device__ __forceinline__ unsigned gv_hfma2(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
// c0 += a.lo * b.lo ; c1 += a.hi * b.hi  -- exact f16 x f16 products, fp32 accumulation (one FHFMA each)
__device__ __forceinline__ void gv_fma2(float &c0, float &c1, unsigned a, unsigned b) {
  asm("{\n\t.reg .b16 al, ah, bl, bh;\n\tmov.b32 {al, ah}, %2;\n\tmov.b32 {bl, bh}, %3;\n\t"
      "fma.rn.f32.f16 %0, al, bl, %0;\n\tfma.rn.f32.f16 %1, ah, bh, %1;\n\t}"
      : "+f"(c0), "+f"(c1) : "r"(a), "r"(b));
}
// c0 += p.lo * s ; c1 += p.hi * s   with s = half `hi` of sv
__device__ __forceinline__ void gv_fma2s(float &c0, float &c1, unsigned p, unsigned sv, int hi) {
  if (hi)
    asm("{\n\t.reg .b16 pl, ph, sl, sh;\n\tmov.b32 {pl, ph}, %2;\n\tmov.b32 {sl, sh}, %3;\n\t"
        "fma.rn.f32.f16 %0, pl, sh, %0;\n\tfma.rn.f32.f16 %1, ph, sh, %1;\n\t}"
        : "+f"(c0), "+f"(c1) : "r"(p), "r"(sv));
  else
    asm("{\n\t.reg .b16 pl, ph, sl, sh;\n\tmov.b32 {pl, ph}, %2;\n\tmov.b32 {sl, sh}, %3;\n\t"
        "fma.rn.f32.f16 %0, pl, sl, %0;\n\tfma.rn.f32.f16 %1, ph, sl, %1;\n\t}"
        : "+f"(c0), "+f"(c1) : "r"(p), "r"(sv));
}
// One 32-k chunk of one weight row against M token rows.
//   wq: 4 words (8 FP4 codes each, low nibble = even k); scale bytes of the chunk's two 16-k blocks = 16-bit half
//   `shalf` of sword (low byte = k 0..15); xp = &x'[0][k0] (f16, row stride ldx elements, 16-byte aligned).
//   acc[2m] / acc[2m+1] = even-k / odd-k fp32 accumulators of token row m.
template <int M, int MODE>
__device__ __forceinline__ void gv_chunk(float *acc, const int *wq, int sword, int shalf, const void *xp, int ldx) {
  const unsigned short *x16 = reinterpret_cast<const unsigned short *>(xp);
  const unsigned s2 = gv_e4m3x2((unsigned)sword, shalf);
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    unsigned h[8];
    gv_fp4x8((unsigned)wq[2 * b], h[0], h[1], h[2], h[3]);
    gv_fp4x8((unsigned)wq[2 * b + 1], h[4], h[5], h[6], h[7]);
    if (MODE == 0) {
      const unsigned sb = gv_bcast(s2, b);
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = gv_hmul2(h[i], sb);      // w * s, exact in f16
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 xa = *reinterpret_cast<const uint4 *>(x16 + m * ldx + 16 * b);
        const uint4 xb = *reinterpret_cast<const uint4 *>(x16 + m * ldx + 16 * b + 8);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[0], xa.x);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[1], xa.y);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[2], xa.z);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[3], xa.w);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[4], xb.x);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[5], xb.y);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[6], xb.z);
        gv_fma2(acc[2 * m], acc[2 * m + 1], h[7], xb.w);
      }
    } else {
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const uint4 xa = *reinterpret_cast<const uint4 *>(x16 + m * ldx + 16 * b);
        const uint4 xb = *reinterpret_cast<const uint4 *>(x16 + m * ldx + 16 * b + 8);
        unsigned p = gv_hmul2(h[0], xa.x);
        p = gv_hfma2(h[1], xa.y, p);
        p = gv_hfma2(h[2], xa.z, p);
        p = gv_hfma2(h[3], xa.w, p);
        p = gv_hfma2(h[4], xb.x, p);
        p = gv_hfma2(h[5], xb.y, p);
        p = gv_hfma2(h[6], xb.z, p);
        p = gv_hfma2(h[7], xb.w, p);
        gv_fma2s(acc[2 * m], acc[2 * m + 1], p, s2, b);
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------------------------------------------
# Kernel

def _smem_bytes(M: int, KCTA: int, WR: int) -> int:
    WK = WARPS // WR
    red = (WK - 1) * M * 32 * WR * 4 if WK > 1 else 4
    return M * KCTA * 2 + WARPS * M * 4 + red + 64


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
                            tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def gemv_kernel(N: int, K: int, M: int, WR: int, S: int, mode: int, cps: int):
    WK = WARPS // WR
    BN = 32 * WR
    KCTA = K // S                       # k per CTA
    KW = KCTA // WK                     # k per warp
    NIT = KW // UNIT_K                  # loop iterations per warp
    assert N % BN == 0 and K % (S * WK * UNIT_K) == 0 and NIT >= 1, (N, K, WR, S)
    assert 1 <= M <= SUPPORTS_M
    NVEC = KCTA // 8                    # 16-byte x vectors per token row of the CTA slice
    NVI = (NVEC + THREADS - 1) // THREADS
    TGT = PRESCALE_EXP[mode]
    fname = f"gv_chunk<{M}, {mode}>"
    bf16, f16, f32, i32, u32 = T.bfloat16, T.float16, T.float32, T.int32, T.uint32
    P_shape = (S, M, N) if S > 1 else (1, 1, 1)
    red_shape = (WK - 1, M, BN) if WK > 1 else (1, 1, 1)

    @T.prim_func
    def main(
        x: T.Tensor((M, K), bf16),
        Wq: T.Tensor((N // 32, K // 32, 32, 4), i32),
        Sq: T.Tensor((N // 32, K // 128, 32, 2), i32),
        alpha: T.Tensor((N,), f32),
        y: T.Tensor((M, N), bf16),
        P: T.Tensor(P_shape, f32),
        counter: T.Tensor((N // 32,), i32),
    ):
        with T.Kernel(N // BN, S, threads=THREADS, prelude=GV_SRC) as (bx, bs):
            x_s = T.alloc_shared((M, KCTA), f16)
            wmax_s = T.alloc_shared((WARPS, M), f32)
            red_s = T.alloc_shared(red_shape, f32)
            flag_s = T.alloc_shared((1,), i32)
            wA = T.alloc_local((16,), i32)
            wB = T.alloc_local((16,), i32)
            sA = T.alloc_local((2,), i32)
            sB = T.alloc_local((2,), i32)
            acc = T.alloc_local((2 * M,), f32)
            am = T.alloc_local((M,), f32)
            sc = T.alloc_local((M,), f32)
            isc = T.alloc_local((M,), f32)
            res = T.alloc_local((M,), f32)
            xv = T.alloc_local((8,), bf16)
            xh = T.alloc_local((8,), f16)
            T.annotate_min_blocks_per_sm(cps)

            tid = T.get_thread_binding()
            lane = tid % 32
            warp = tid // 32
            wr = warp % WR
            wk = warp // WR
            rg = bx * WR + wr                               # global 32-row group of this warp
            kc0 = bs * (KCTA // 32) + wk * (KW // 32)       # first global 32-k chunk of this warp
            ku0 = bs * (KCTA // 128) + wk * (KW // 128)     # first global 128-k scale unit of this warp
            kl0 = wk * KW                                   # first k of this warp inside the CTA slice
            n = bx * BN + wr * 32 + lane                    # output column of this lane

            # (1) iteration 0 of the weight stream goes out before the x prologue
            for c in T.unroll(4):
                for j in T.vectorized(4):
                    wA[c * 4 + j] = Wq[rg, kc0 + c, lane, j]
            for j in T.vectorized(2):
                sA[j] = Sq[rg, ku0, lane, j]

            # (2) x prologue: per token row absmax over the CTA's K slice -> exact power-of-two prescale -> f16 x'
            for m in T.unroll(M):
                am[m] = T.float32(0)
                for i in T.serial(NVI):
                    v = i * THREADS + tid
                    if v < NVEC:
                        for j in T.vectorized(8):
                            xv[j] = x[m, bs * KCTA + v * 8 + j]
                        for j in T.unroll(8):
                            am[m] = T.max(am[m], T.abs(T.cast(xv[j], f32)))
                am[m] = T.max(am[m], T.shfl_xor(am[m], 16))
                am[m] = T.max(am[m], T.shfl_xor(am[m], 8))
                am[m] = T.max(am[m], T.shfl_xor(am[m], 4))
                am[m] = T.max(am[m], T.shfl_xor(am[m], 2))
                am[m] = T.max(am[m], T.shfl_xor(am[m], 1))
                if lane == 0:
                    wmax_s[warp, m] = am[m]
            T.sync_threads()
            for m in T.unroll(M):
                for w in T.unroll(WARPS):
                    am[m] = T.max(am[m], wmax_s[w, m])
                # biased exponent E of the absmax; scale = 2^(TGT + 127 - E) as an fp32 bit pattern, kept normal
                ebits = T.cast((T.reinterpret(am[m], u32) >> T.uint32(23)) & T.uint32(0xFF), i32)
                sfield = T.max(T.min(TGT + 254 - ebits, 253), 1)
                sc[m] = T.reinterpret(T.cast(sfield, u32) << T.uint32(23), f32)
                isc[m] = T.reinterpret(T.cast(254 - sfield, u32) << T.uint32(23), f32)
            T.sync_threads()        # wmax_s may share storage with x_s: all reads done before x_s is written
            for m in T.unroll(M):
                for i in T.serial(NVI):
                    v = i * THREADS + tid
                    if v < NVEC:
                        for j in T.vectorized(8):
                            xv[j] = x[m, bs * KCTA + v * 8 + j]
                        for j in T.unroll(8):
                            xh[j] = T.cast(T.cast(xv[j], f32) * sc[m], f16)
                        for j in T.vectorized(8):
                            x_s[m, v * 8 + j] = xh[j]
            T.sync_threads()

            # (3) main loop: register double buffer A/B, 4 chunks (128 k) per iteration and lane
            for q in T.unroll(2 * M):
                acc[q] = T.float32(0)
            for it2 in T.serial(NIT // 2):
                it = it2 * 2
                for c in T.unroll(4):
                    for j in T.vectorized(4):
                        wB[c * 4 + j] = Wq[rg, kc0 + (it + 1) * 4 + c, lane, j]
                for j in T.vectorized(2):
                    sB[j] = Sq[rg, ku0 + it + 1, lane, j]
                for c in T.unroll(4):
                    T.call_extern("handle", fname, T.access_ptr(acc, "rw"), T.access_ptr(wA[c * 4], "r"),
                                  sA[c // 2], c % 2, T.access_ptr(x_s[0, kl0 + it * UNIT_K + c * 32], "r"), KCTA)
                if it + 2 < NIT:
                    for c in T.unroll(4):
                        for j in T.vectorized(4):
                            wA[c * 4 + j] = Wq[rg, kc0 + (it + 2) * 4 + c, lane, j]
                    for j in T.vectorized(2):
                        sA[j] = Sq[rg, ku0 + it + 2, lane, j]
                for c in T.unroll(4):
                    T.call_extern("handle", fname, T.access_ptr(acc, "rw"), T.access_ptr(wB[c * 4], "r"),
                                  sB[c // 2], c % 2,
                                  T.access_ptr(x_s[0, kl0 + (it + 1) * UNIT_K + c * 32], "r"), KCTA)
            if NIT % 2 == 1:
                for c in T.unroll(4):
                    T.call_extern("handle", fname, T.access_ptr(acc, "rw"), T.access_ptr(wA[c * 4], "r"),
                                  sA[c // 2], c % 2,
                                  T.access_ptr(x_s[0, kl0 + (NIT - 1) * UNIT_K + c * 32], "r"), KCTA)

            # (4) epilogue: undo the prescale, fixed-order k-warp reduction, then split-K (if any), alpha, bf16
            for m in T.unroll(M):
                res[m] = (acc[2 * m] + acc[2 * m + 1]) * isc[m]
            if WK > 1:
                if wk > 0:
                    for m in T.unroll(M):
                        red_s[wk - 1, m, wr * 32 + lane] = res[m]
                T.sync_threads()
                if wk == 0:
                    for w in T.serial(WK - 1):
                        for m in T.unroll(M):
                            res[m] = res[m] + red_s[w, m, wr * 32 + lane]
            if S == 1:
                if wk == 0:
                    for m in T.unroll(M):
                        y[m, n] = T.cast(res[m] * alpha[n], bf16)
            else:
                if wk == 0:
                    for m in T.unroll(M):
                        P[bs, m, n] = res[m]
                T.call_extern("handle", "__threadfence")
                T.sync_threads()
                if tid == 0:
                    flag_s[0] = T.atomic_add(counter[bx], 1, return_prev=True)
                T.sync_threads()
                if flag_s[0] == S - 1:
                    # last CTA of this row tile: sum the partials in split order 0..S-1 (deterministic)
                    T.call_extern("handle", "__threadfence")
                    if wk == 0:
                        for m in T.unroll(M):
                            res[m] = T.call_extern(f32, "__ldcg", T.access_ptr(P[0, m, n], "r"))
                            for s in T.serial(S - 1):
                                res[m] = res[m] + T.call_extern(f32, "__ldcg", T.access_ptr(P[s + 1, m, n], "r"))
                            y[m, n] = T.cast(res[m] * alpha[n], bf16)
                    if tid == 0:
                        counter[bx] = 0

    return main


# ---------------------------------------------------------------------------------------------------------------
# Host side

@dataclass
class GemvWeight:
    Wq: torch.Tensor        # int32 (N/32, K/32, 32, 4): [row group][32-k chunk][lane = row][16 B of FP4 codes]
    Sq: torch.Tensor        # int32 (N/32, K/128, 32, 2): [row group][128-k unit][lane][8 UE4M3 bytes]
    alpha: torch.Tensor     # fp32 (N,)
    counter: torch.Tensor   # int32 (N/32,) split-K tickets (always back to 0 after a call)
    dummy: torch.Tensor     # fp32 (1, 1, 1) placeholder workspace for S == 1
    N: int
    K: int


def prepare(packed_int8: torch.Tensor, scale_fp8: torch.Tensor, alpha_fp32: torch.Tensor) -> GemvWeight:
    """Re-layout only (untimed): bytes and scale bytes are moved, never re-encoded."""
    N, K = packed_int8.shape[0], packed_int8.shape[1] * 2
    assert N % 32 == 0 and K % 256 == 0, (N, K)
    assert scale_fp8.shape == (N, K // 16), (tuple(scale_fp8.shape), N, K)
    u = packed_int8.contiguous().view(torch.uint8).reshape(N // 32, 32, K // 32, 16)
    Wq = u.permute(0, 2, 1, 3).contiguous().view(torch.int32)                 # (N/32, K/32, 32, 4)
    s = scale_fp8.contiguous().view(torch.uint8).reshape(N // 32, 32, K // 128, 8)
    Sq = s.permute(0, 2, 1, 3).contiguous().view(torch.int32)                 # (N/32, K/128, 32, 2)
    dev = packed_int8.device
    alpha = alpha_fp32.to(device=dev, dtype=torch.float32).contiguous()
    counter = torch.zeros(N // 32, dtype=torch.int32, device=dev)
    dummy = torch.zeros(1, 1, 1, dtype=torch.float32, device=dev)
    return GemvWeight(Wq, Sq, alpha, counter, dummy, N, K)


def _instr_per_weight(M: int, mode: int) -> float:
    """Main-loop thread instructions per weight (model, see module docstring)."""
    return (1.06 + 1.125 * M) if mode == EXACT else (0.56 + 0.75 * M)


def _config(N: int, K: int, M: int) -> tuple[int, int]:
    """(WR, S) for a single wave of equal CTAs, scored with a bandwidth / issue / overhead model (no measurements)."""
    mode = MODE_BY_M[M]
    bw, clk = 1.63e12, 2.2e9
    best, best_key = None, None
    for WR in (1, 2, 4, 8):
        WK, BN = WARPS // WR, 32 * WR
        if N % BN:
            continue
        for S in range(1, K // (WK * UNIT_K) + 1):
            if K % (S * WK * UNIT_K):
                continue
            KCTA = K // S
            NIT = KCTA // WK // UNIT_K
            smem = _smem_bytes(M, KCTA, WR)
            if smem > SMEM_MAX:
                continue
            cps = 2 if smem <= SMEM_2CTA else 1
            ctas = (N // BN) * S
            if ctas > NUM_SMS * cps:                       # single wave only
                continue
            nbytes = N * K * 9 / 16 + (2 * S * M * N * 4 if S > 1 else 0)
            inflight = ctas * THREADS * 144                # ~2 iterations of 72 B per lane
            t_mem = nbytes / bw / min(1.0, inflight / 3e6)
            per_sm = -(-ctas // NUM_SMS)                   # CTAs on the busiest SM
            instr = per_sm * (BN * KCTA * _instr_per_weight(M, mode) + M * KCTA * 4)   # + x prologue
            t_issue = instr / (128 * clk) / 0.7
            t = max(t_mem, t_issue) + (1.5e-6 if S > 1 else 0.0) + (0.5e-6 if NIT < 3 else 0.0)
            key = (round(t * 1e8), -ctas, S)
            if best_key is None or key < best_key:
                best, best_key = (WR, S), key
    assert best is not None, (N, K, M)
    return best


_config = functools.lru_cache(maxsize=None)(_config)


def config(N: int, K: int, M: int) -> tuple[int, int, int, int]:
    """(WR, S, mode, CTAs per SM) for one layer shape and token count."""
    WR, S = CONFIG_OVERRIDE.get((N, K, M)) or _config(N, K, M)
    cps = 2 if _smem_bytes(M, K // S, WR) <= SMEM_2CTA else 1
    return WR, S, MODE_BY_M[M], cps


def linear(x: torch.Tensor, prep: GemvWeight) -> torch.Tensor:
    """y = x @ W^T (bf16) for M <= SUPPORTS_M tokens (larger M is processed in groups of SUPPORTS_M rows)."""
    M, K = x.shape
    assert K == prep.K and M >= 1, (tuple(x.shape), prep.K)
    assert x.dtype == torch.bfloat16, x.dtype
    x = x.contiguous()
    if x.data_ptr() % 16:                  # the kernel reads x with 16-byte vector loads
        x = x.clone()
    if M > SUPPORTS_M:
        return torch.cat([linear(x[i:i + SUPPORTS_M], prep) for i in range(0, M, SUPPORTS_M)])
    WR, S, mode, cps = config(prep.N, K, M)
    kernel = gemv_kernel(prep.N, K, M, WR, S, mode, cps)
    y = torch.empty(M, prep.N, dtype=torch.bfloat16, device=x.device)
    P = torch.empty(S, M, prep.N, dtype=torch.float32, device=x.device) if S > 1 else prep.dummy
    kernel(x, prep.Wq, prep.Sq, prep.alpha, y, P, prep.counter)
    return y
