# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Split-KV decode attention over the Horizon NVFP4 paged KV cache, v2: register-level FP4 unpack (SM120).

Same contract and output semantics as attn_decode_fp4.decode (opart / lse per split, attn_decode's fixed-order
combine): out = softmax(q . k * bmm1_scale * gk[h]) @ v * gv[h] * bmm2_scale over the same dequantised cache
(e2m1(code) * e4m3(sf)), fp32 scores / online softmax / accumulators, P rounded to bf16 for P @ V, causal among the
QL (1..4) query tokens of a request. Deterministic (fixed partition and summation order, no atomics).

Why v2: v1 expanded every FP4 element to bf16 in shared memory with ~15 scalar instructions and then re-read the
tile with ldmatrix (5-10x slower than XQA on the GPU, 29-09-2026). v2 never materialises bf16 tiles:
  * raw K / V bytes + scales are staged per warp with a NST-deep cp.async ring (no CTA barrier for staging);
  * each lane unpacks its bytes straight into mma.m16n8k16 fragments: 8 nibbles per 32-bit word with the Marlin
    bit trick y -> (y & 0x80008000) | ((y & 0x70007000) >> 6) (one bf16x2 = two e2m1 values * 2^-126), then ONE
    bf16x2 multiply by the block scale pre-multiplied by 2^117 (e4m3 * 2^117 fits bf16): values come out as
    code * sf * 2^-9 EXACTLY (<= 5 significant bits, no rounding); the 2^9 is folded into the fp32 score scale and
    the output scale (exact powers of two);
  * S^T = K Q^T (tokens as the M dimension, 8 query rows per n-tile: QL 1 uses 6 of 8 rows, QL 4 uses 24 of 24);
    channels are assigned to MMA k-slots in the order the lane's bytes arrive, and Q's fragments use the same
    permutation (a dot product does not care about channel order);
  * P^T is moved from the S accumulator layout to the B-fragment layout with movmatrix.trans (no shared memory);
  * O^T = V^T P^T: V's token pairs are gathered with one byte_perm per byte;
  * the head dim is split over NCW warps (channel slices of 256 / NCW) so the fp32 O accumulator fits registers;
    the partial scores of the slices are summed through shared memory in fixed warp order (one __syncthreads per
    16-token block).
Split / chunk rule: chunk = max(min_chunk, ceil16(ceil(L / S))), split s covers [s * chunk, (s + 1) * chunk).
Requires D = 256, PS % 16 == 0, G * QL <= 32, bf16 q, the kv_fp4_ref page geometry (any strides with 16-byte aligned
rows and 4/8-byte aligned scale slices).
"""

import tilelang
import tilelang.language as T

from .attn_decode import LOG2E, get_combine, _workspace

_KERNELS = {}
PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
D = 256
NUM_SMS = 188

V2_SRC = r"""
#ifndef HZ4_V2_SRC_DEFINED
#define HZ4_V2_SRC_DEFINED
#include <cuda_bf16.h>
#include <stdint.h>
namespace hz4 {
// two e2m1 nibbles at bits 12..15 / 28..31 -> bf16x2 of (value * 2^-126), sign kept (Marlin's FP4 -> bf16 placement)
__device__ __forceinline__ uint32_t mar(uint32_t y) { return (y & 0x80008000u) | ((y & 0x70007000u) >> 6); }
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
// bf16 bits of e4m3(byte & 0x7f) * 2^117, exact; garbage bytes stay finite (0x7f -> 1.875 * 2^125)
__device__ __forceinline__ uint32_t sf117(uint32_t byte) {
  uint32_t e = (byte >> 3) & 15u, m = byte & 7u;
  float f = e ? __uint_as_float(((e + 237u) << 23) | (m << 20)) : __fmul_rn((float)m, __uint_as_float(235u << 23));
  return __float_as_uint(f) >> 16;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t movt(uint32_t x) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t sa(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa(s)), "l"(g) : "memory");
}
template <int N> __device__ __forceinline__ void cpn(void* s, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(sa(s)), "l"(g), "n"(N) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

template <int KVH, int G, int QL, int NCW, int NST, int MINC, int PS,
          long long DSP, long long DST, long long DSH, long long SSP, long long SST, long long SSH>
struct Cfg {
  static constexpr int D = 256, HQ = KVH * G, R = G * QL, NQT = (R + 7) / 8, BH = 32;
  static constexpr int DW = D / NCW;            // channel slice per warp
  static constexpr int RB = DW / 2;             // K / V data bytes per token in the slice
  static constexpr int VRS = RB + 16;           // V staging row stride (bank-conflict-free V reads)
  static constexpr int SB = DW / 16;            // scale bytes per token in the slice
  static constexpr int CPL = DW / 4;            // K channels per lane (lane c of a token row)
  static constexpr int KW = CPL / 8;            // K words per token per lane
  static constexpr int KS = DW / 16;            // MMA k-steps per slice (= 2 * KW)
  static constexpr int CPG = DW / 8;            // V channels per lane group g
  static constexpr int MT = DW / 16;            // V m-tiles (16 channels) per slice = V bytes per token per lane
  static constexpr int VW = (MT + 3) / 4;       // V words per token per lane
  static constexpr int STAGE = 16 * RB + 16 * VRS + 32 * SB;
  static constexpr int WARP_SMEM = NST * STAGE;
  static constexpr int XBUF = 2 * NCW * NQT * 4 * 32 * 4;
  static constexpr int SMEM = NCW * WARP_SMEM + XBUF;
  static_assert(NCW == 2 || NCW == 4, "channel slices of 128 or 64");
  static_assert(PS % 16 == 0 && R <= BH && NST >= 2, "config");
};

template <int KVH, int G, int QL, int NCW, int NST, int MINC, int PS,
          long long DSP, long long DST, long long DSH, long long SSP, long long SST, long long SSH>
__device__ __forceinline__ void attn_task(const void* q_, const void* kd_, const void* ksf_, const void* vd_,
                                          const void* vsf_, const void* bt_, const void* sl_, const void* gk_,
                                          const void* gv_, void* opart_, void* lse_, float scale_log2, int MP,
                                          void* smem_) {
  using C = Cfg<KVH, G, QL, NCW, NST, MINC, PS, DSP, DST, DSH, SSP, SST, SSH>;
  constexpr int D = C::D, HQ = C::HQ, R = C::R, NQT = C::NQT, BH = C::BH;
  constexpr int RB = C::RB, VRS = C::VRS, SB = C::SB, CPL = C::CPL, KW = C::KW, KS = C::KS, CPG = C::CPG,
                MT = C::MT, VW = C::VW, STAGE = C::STAGE;
  const uint16_t* q = static_cast<const uint16_t*>(q_);
  const uint8_t* kd = static_cast<const uint8_t*>(kd_);
  const uint8_t* ksf = static_cast<const uint8_t*>(ksf_);
  const uint8_t* vd = static_cast<const uint8_t*>(vd_);
  const uint8_t* vsf = static_cast<const uint8_t*>(vsf_);
  const int* bt = static_cast<const int*>(bt_);
  const int* sl = static_cast<const int*>(sl_);
  float* opart = static_cast<float*>(opart_);
  float* lse = static_cast<float*>(lse_);
  uint8_t* smem = static_cast<uint8_t*>(smem_);

  const int lane = threadIdx.x & 31, cs = threadIdx.x >> 5, g = lane >> 2, c = lane & 3;
  const int b = blockIdx.x, h = blockIdx.y, s = blockIdx.z, S = gridDim.z;
  const int cb = cs * C::DW;
  const int L = sl[b];
  int chunk = (L + S - 1) / S;
  chunk = ((chunk + 15) / 16) * 16;
  chunk = chunk > MINC ? chunk : MINC;
  const int st = s * chunk;
  const int en = min(L, st + chunk);
  const int nblk = en > st ? (en - st + 15) / 16 : 0;
  const float sc = scale_log2 * static_cast<const float*>(gk_)[h] * 512.0f;
  const float NEG = -__int_as_float(0x7f800000);

  // Q^T B-fragments of this lane: k-step t = 2w + u uses channels cb + CPL*c + 8w + 2u + {0,4} (b0), + {1,5} (b1)
  uint32_t qf[NQT][KS][2];
#pragma unroll
  for (int nt = 0; nt < NQT; ++nt) {
    const int row = nt * 8 + g;
    const bool ok = row < R;
    const uint16_t* qr = q + ((size_t)(b * QL + (ok ? row / G : 0)) * HQ + h * G + (ok ? row % G : 0)) * D;
#pragma unroll
    for (int w = 0; w < KW; ++w)
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int ch = cb + CPL * c + 8 * w + 2 * u;
        qf[nt][2 * w + u][0] = ok ? ((uint32_t)qr[ch] | ((uint32_t)qr[ch + 4] << 16)) : 0u;
        qf[nt][2 * w + u][1] = ok ? ((uint32_t)qr[ch + 1] | ((uint32_t)qr[ch + 5] << 16)) : 0u;
      }
  }
  float m[NQT][2], l[NQT][2], acc[MT][NQT][4];
#pragma unroll
  for (int nt = 0; nt < NQT; ++nt) {
    m[nt][0] = m[nt][1] = NEG;
    l[nt][0] = l[nt][1] = 0.0f;
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j][nt][0] = acc[j][nt][1] = acc[j][nt][2] = acc[j][nt][3] = 0.0f;
  }

  uint8_t* wsm = smem + cs * C::WARP_SMEM;
  float* xbuf = reinterpret_cast<float*>(smem + NCW * C::WARP_SMEM);
  // stage layout: K data [16][RB] | V data [16][VRS] | K scales [16][SB] | V scales [16][SB]
  auto issue = [&](int k, int stg) {
    const int p0 = st + 16 * k;
    const long long page = bt[(size_t)b * MP + p0 / PS];
    const int pos0 = p0 % PS;
    uint8_t* sK = wsm + stg * STAGE;
    uint8_t* sV = sK + 16 * RB;
    uint8_t* sKS = sV + 16 * VRS;
    uint8_t* sVS = sKS + 16 * SB;
    const uint8_t* gK = kd + page * DSP + pos0 * DST + h * DSH + cb / 2;
    const uint8_t* gV = vd + page * DSP + pos0 * DST + h * DSH + cb / 2;
    constexpr int CH = RB / 16;                 // 16-byte chunks per row
#pragma unroll
    for (int i = lane; i < 16 * CH; i += 32) {
      const int r = i / CH, k16 = i % CH;
      cp16(sK + r * RB + k16 * 16, gK + r * DST + k16 * 16);
      cp16(sV + r * VRS + k16 * 16, gV + r * DST + k16 * 16);
    }
    const int r = lane & 15;
    if (lane < 16)
      cpn<SB>(sKS + r * SB, ksf + page * SSP + (pos0 + r) * SST + h * SSH + cs * SB);
    else
      cpn<SB>(sVS + r * SB, vsf + page * SSP + (pos0 + r) * SST + h * SSH + cs * SB);
  };
#pragma unroll
  for (int i = 0; i < NST - 1; ++i) {
    if (i < nblk) issue(i, i);
    commit();
  }

  for (int k = 0; k < nblk; ++k) {
    wait<NST - 2>();
    __syncwarp();
    if (k + NST - 1 < nblk) issue(k + NST - 1, (k + NST - 1) % NST);
    commit();
    const uint8_t* sK = wsm + (k % NST) * STAGE;
    const uint8_t* sV = sK + 16 * RB;
    const uint8_t* sKS = sV + 16 * VRS;
    const uint8_t* sVS = sKS + 16 * SB;
    const int p0 = st + 16 * k;

    // ---- S^T = K Q^T over this warp's channel slice
    float sacc[NQT][4];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.0f;
    uint32_t kw0[KW], kw1[KW];
    if constexpr (KW == 4) {
      const uint4 x0 = *reinterpret_cast<const uint4*>(sK + g * RB + 16 * c);
      const uint4 x1 = *reinterpret_cast<const uint4*>(sK + (g + 8) * RB + 16 * c);
      kw0[0] = x0.x; kw0[1] = x0.y; kw0[2] = x0.z; kw0[3] = x0.w;
      kw1[0] = x1.x; kw1[1] = x1.y; kw1[2] = x1.z; kw1[3] = x1.w;
    } else {
      const uint2 x0 = *reinterpret_cast<const uint2*>(sK + g * RB + 8 * c);
      const uint2 x1 = *reinterpret_cast<const uint2*>(sK + (g + 8) * RB + 8 * c);
      kw0[0] = x0.x; kw0[1] = x0.y;
      kw1[0] = x1.x; kw1[1] = x1.y;
    }
#pragma unroll
    for (int w = 0; w < KW; ++w) {
      const int blk = (CPL * c + 8 * w) / 16;     // local scale block of word w
      const uint32_t s0 = sf117(sKS[g * SB + blk]), s1 = sf117(sKS[(g + 8) * SB + blk]);
      const uint32_t s0p = s0 | (s0 << 16), s1p = s1 | (s1 << 16);
      const uint32_t y0 = kw0[w], y1 = kw1[w];
      const uint32_t P3a = hmul2(mar(y0), s0p), P2a = hmul2(mar(y0 << 4), s0p);
      const uint32_t P1a = hmul2(mar(y0 << 8), s0p), P0a = hmul2(mar(y0 << 12), s0p);
      const uint32_t P3b = hmul2(mar(y1), s1p), P2b = hmul2(mar(y1 << 4), s1p);
      const uint32_t P1b = hmul2(mar(y1 << 8), s1p), P0b = hmul2(mar(y1 << 12), s1p);
#pragma unroll
      for (int nt = 0; nt < NQT; ++nt) {
        mma(sacc[nt], P0a, P0b, P1a, P1b, qf[nt][2 * w][0], qf[nt][2 * w][1]);
        mma(sacc[nt], P2a, P2b, P3a, P3b, qf[nt][2 * w + 1][0], qf[nt][2 * w + 1][1]);
      }
    }
    // ---- sum the NCW channel slices in fixed order
    float* xb = xbuf + (k & 1) * (NCW * NQT * 4 * 32);
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) xb[((cs * NQT + nt) * 4 + e) * 32 + lane] = sacc[nt][e];
    __syncthreads();
    float x[NQT][4];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = xb[(nt * 4 + e) * 32 + lane];
#pragma unroll
        for (int w2 = 1; w2 < NCW; ++w2) v += xb[((w2 * NQT + nt) * 4 + e) * 32 + lane];
        const int tok = (e < 2) ? g : g + 8;
        const int row = nt * 8 + 2 * c + (e & 1);
        const int pos = p0 + tok;
        const bool ok = row < R && pos < en && pos < L - QL + row / G + 1;
        x[nt][e] = ok ? v * sc : NEG;
      }
    // ---- online softmax (per query row = S^T column), per-lane partial row sums
    bool resc = false;
    float alpha[NQT][2];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
      for (int par = 0; par < 2; ++par) {
        float bm = fmaxf(x[nt][par], x[nt][par + 2]);
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 4));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 8));
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 16));
        const float mn = fmaxf(m[nt][par], bm);
        const float a = (mn == NEG) ? 1.0f : exp2f(m[nt][par] - mn);
        const float pa = (mn == NEG) ? 0.0f : exp2f(x[nt][par] - mn);
        const float pb = (mn == NEG) ? 0.0f : exp2f(x[nt][par + 2] - mn);
        l[nt][par] = l[nt][par] * a + (pa + pb);
        m[nt][par] = mn;
        alpha[nt][par] = a;
        resc = resc || (a != 1.0f);
        x[nt][par] = pa;
        x[nt][par + 2] = pb;
      }
    if (__any_sync(0xffffffffu, resc)) {
#pragma unroll
      for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc[j][nt][e] *= alpha[nt][e & 1];
    }
    // ---- P^T B-fragments (movmatrix: S^T accumulator layout -> B layout)
    uint32_t pb[NQT][2];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) {
      pb[nt][0] = movt(pack_bf16(x[nt][0], x[nt][1]));
      pb[nt][1] = movt(pack_bf16(x[nt][2], x[nt][3]));
    }
    // ---- O^T += V^T P^T: lane (g, c) owns channels cb + CPG*g + 2j (+1) of tokens 2c, 2c+1, 2c+8, 2c+9
    uint32_t va[VW], vb[VW], vc[VW], vdw[VW];
    if constexpr (VW == 2) {
      const uint2 A = *reinterpret_cast<const uint2*>(sV + (2 * c) * VRS + 8 * g);
      const uint2 B = *reinterpret_cast<const uint2*>(sV + (2 * c + 1) * VRS + 8 * g);
      const uint2 Cc = *reinterpret_cast<const uint2*>(sV + (2 * c + 8) * VRS + 8 * g);
      const uint2 Dd = *reinterpret_cast<const uint2*>(sV + (2 * c + 9) * VRS + 8 * g);
      va[0] = A.x; va[1] = A.y; vb[0] = B.x; vb[1] = B.y; vc[0] = Cc.x; vc[1] = Cc.y; vdw[0] = Dd.x; vdw[1] = Dd.y;
    } else {
      va[0] = *reinterpret_cast<const uint32_t*>(sV + (2 * c) * VRS + 4 * g);
      vb[0] = *reinterpret_cast<const uint32_t*>(sV + (2 * c + 1) * VRS + 4 * g);
      vc[0] = *reinterpret_cast<const uint32_t*>(sV + (2 * c + 8) * VRS + 4 * g);
      vdw[0] = *reinterpret_cast<const uint32_t*>(sV + (2 * c + 9) * VRS + 4 * g);
    }
    const int vblk = (CPG * g) / 16;
    const uint32_t sA = sf117(sVS[(2 * c) * SB + vblk]), sB = sf117(sVS[(2 * c + 1) * SB + vblk]);
    const uint32_t sC = sf117(sVS[(2 * c + 8) * SB + vblk]), sD = sf117(sVS[(2 * c + 9) * SB + vblk]);
    const uint32_t sAB = sA | (sB << 16), sCD = sC | (sD << 16);
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const uint32_t sel = (j & 3) | ((j & 3) << 4) | ((4 + (j & 3)) << 8) | ((4 + (j & 3)) << 12);
      const uint32_t yab = __byte_perm(va[j >> 2], vb[j >> 2], sel);
      const uint32_t ycd = __byte_perm(vc[j >> 2], vdw[j >> 2], sel);
      const uint32_t a0 = hmul2(mar(yab << 4), sAB), a1 = hmul2(mar(yab), sAB);
      const uint32_t a2 = hmul2(mar(ycd << 4), sCD), a3 = hmul2(mar(ycd), sCD);
#pragma unroll
      for (int nt = 0; nt < NQT; ++nt) mma(acc[j][nt], a0, a1, a2, a3, pb[nt][0], pb[nt][1]);
    }
  }
  wait<0>();

  // ---- epilogue: full row sums, normalised partial output (times gv * 2^9), log2-sum-exp
  const float gvs = static_cast<const float*>(gv_)[h] * 512.0f;
  const size_t base = ((size_t)(b * KVH + h) * S + s) * BH;
#pragma unroll
  for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
    for (int par = 0; par < 2; ++par) {
      float lt = l[nt][par];
      lt += __shfl_xor_sync(0xffffffffu, lt, 4);
      lt += __shfl_xor_sync(0xffffffffu, lt, 8);
      lt += __shfl_xor_sync(0xffffffffu, lt, 16);
      l[nt][par] = lt;
      const int row = nt * 8 + 2 * c + par;
      if (cs == 0 && g == 0 && row < R) lse[base + row] = lt > 0.0f ? log2f(lt) + m[nt][par] : NEG;
    }
#pragma unroll
  for (int j = 0; j < MT; ++j)
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = nt * 8 + 2 * c + (e & 1);
        const int ch = cb + CPG * g + 2 * j + (e >> 1);
        const float lt = l[nt][e & 1];
        if (row < R) opart[(base + row) * D + ch] = lt > 0.0f ? acc[j][nt][e] / lt * gvs : 0.0f;
      }
}
}  // namespace hz4
#endif
"""


def default_ncw(G: int, QL: int) -> int:
    return 2 if (G * QL + 7) // 8 <= 3 else 4


def smem_bytes(KVH, G, QL, NCW, NST):
    R = G * QL
    NQT = (R + 7) // 8
    DW = D // NCW
    RB, SB = DW // 2, DW // 16
    stage = 16 * RB + 16 * (RB + 16) + 32 * SB
    return NCW * NST * stage + 2 * NCW * NQT * 4 * 32 * 4


@tilelang.jit(pass_configs=PASS_CONFIGS)
def attn_fp4_v2_kernel(KVH: int, G: int, QL: int, PS: int, DSP: int, DST: int, DSH: int, SSP: int, SST: int,
                       SSH: int, NCW: int = 2, NST: int = 4, MINC: int = 64):
    """grid (B, KVH, S), NCW warps per CTA. kd / vd uint8 [page, position, kv head, byte] (NP, PS, KVH, 128)
    strides (DSP, DST, DSH, 1); ksf / vsf uint8 (NP, PS, KVH, 16) strides (SSP, SST, SSH, 1)."""
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    NP = T.dynamic("NP")
    MP = T.dynamic("MP")
    S = T.dynamic("S")
    HQ = KVH * G
    f32, i32 = T.float32, T.int32
    SM = smem_bytes(KVH, G, QL, NCW, NST)
    fn = (f"hz4::attn_task<{KVH}, {G}, {QL}, {NCW}, {NST}, {MINC}, {PS}, {DSP}LL, {DST}LL, {DSH}LL, {SSP}LL, "
          f"{SST}LL, {SSH}LL>")

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), T.bfloat16),
             kd: T.StridedTensor((NP, PS, KVH, D // 2), (DSP, DST, DSH, 1), T.uint8),
             ksf: T.StridedTensor((NP, PS, KVH, D // 16), (SSP, SST, SSH, 1), T.uint8),
             vd: T.StridedTensor((NP, PS, KVH, D // 2), (DSP, DST, DSH, 1), T.uint8),
             vsf: T.StridedTensor((NP, PS, KVH, D // 16), (SSP, SST, SSH, 1), T.uint8),
             bt: T.Tensor((B, MP), i32),
             sl: T.Tensor((B,), i32),
             gk: T.Tensor((KVH,), f32),
             gv: T.Tensor((KVH,), f32),
             opart: T.Tensor((B, KVH, S, 32, D), f32),
             lse: T.Tensor((B, KVH, S, 32), f32),
             scale_log2: f32):
        with T.Kernel(B, KVH, S, threads=32 * NCW, prelude=V2_SRC) as (b, h, s):
            smem = T.alloc_shared((SM,), T.uint8)
            T.call_extern("handle", fn, T.address_of(q[0, 0, 0]), T.address_of(kd[0, 0, 0, 0]),
                          T.address_of(ksf[0, 0, 0, 0]), T.address_of(vd[0, 0, 0, 0]), T.address_of(vsf[0, 0, 0, 0]),
                          T.address_of(bt[0, 0]), T.address_of(sl[0]), T.address_of(gk[0]), T.address_of(gv[0]),
                          T.address_of(opart[0, 0, 0, 0, 0]), T.address_of(lse[0, 0, 0, 0]), scale_log2, MP,
                          T.address_of(smem[0]))

    return main


def get_v2(KVH, G, QL, PS, dstr, sstr, NCW=None, NST=4, MINC=64):
    NCW = NCW or default_ncw(G, QL)
    key = ("v2", KVH, G, QL, PS, tuple(dstr), tuple(sstr), NCW, NST, MINC)
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = attn_fp4_v2_kernel(KVH, G, QL, PS, *dstr, *sstr, NCW, NST, MINC)
    return k


def plan_splits_v2(B, KVH, max_seq_len=0, ctas_per_sm=4, min_chunk=64, max_splits=64):
    """one wave of ~ctas_per_sm CTAs per SM; capped by the longest request when max_seq_len (host int) is given, so
    short contexts do not pay for empty splits in the combine"""
    S = max(1, min(max_splits, NUM_SMS * ctas_per_sm // max(1, B * KVH)))
    if max_seq_len and max_seq_len > 0:
        S = max(1, min(S, -(-int(max_seq_len) // min_chunk)))
    return S


def decode(query, k_data, k_sf, v_data, v_sf, block_tables, seq_lens, max_seq_len, bmm1_scale, bmm2_scale,
           q_len_per_req, gk, gv, out=None, splits=None, ncw=None, nst=4, min_chunk=64, ctas_per_sm=4):
    """Drop-in for attn_decode_fp4.decode (same arguments; query rotated with rotate_q when keys were written
    rotated). splits: fixed split count (CUDA-graph use: keep it constant); default plans from B and max_seq_len."""
    import torch
    NT, HQ, Dq = query.shape
    _, PS, KVH, DH = k_data.shape
    assert Dq == D and DH * 2 == D and PS % 16 == 0
    assert k_data.stride() == v_data.stride() and k_sf.stride() == v_sf.stride()
    QL = int(q_len_per_req or 1)
    B = NT // QL
    G = HQ // KVH
    assert G * QL <= 32 and query.dtype == torch.bfloat16 and query.is_contiguous()
    if out is None:
        out = torch.empty_like(query)
    S = splits or plan_splits_v2(B, KVH, max_seq_len, ctas_per_sm, min_chunk)
    op, ls = _workspace(query.device, B, KVH, S, 32, D)
    get_v2(KVH, G, QL, PS, k_data.stride()[:3], k_sf.stride()[:3], ncw, nst, min_chunk)(
        query, k_data, k_sf.view(torch.uint8), v_data, v_sf.view(torch.uint8), block_tables, seq_lens,
        gk.float().contiguous(), gv.float().contiguous(), op, ls, float(bmm1_scale) * LOG2E)
    get_combine(KVH, G, QL, D)(op, ls, out, float(bmm2_scale))
    return out
