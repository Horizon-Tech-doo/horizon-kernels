# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Serving kernels for the Horizon NVFP4 KV cache (hz-kvfp4 v1) inside vLLM: what the attention backend calls.

1. decode_ragged: attn_decode_fp4_v2's register-unpack decode, generalised for vLLM decode batches
   * per-request query length from query_start_loc on the device (0 .. QLMAX tokens, e.g. MTP verify 4 / 2, draft
     steps 1, CUDA-graph padding requests with 0 tokens), queries addressed at query_start_loc[b] + t; requests with
     more than QLMAX tokens (prefill chunks) are skipped, so one launch covers any batch order and the caller runs
     only real prefill rows through the prefill path;
     causal among a request's own query tokens: token t of a request with q tokens and seq_len L sees positions
     < L - q + t + 1;
   * cache strides and page size are runtime arguments (one compiled kernel serves any page / block stride vLLM picks);
   * the same arithmetic as v2 (exact code * sf * 2^-9 bf16 fragments, fp32 online softmax, bf16 P, fixed-order
     slice sum), so v2's GPU gate carries over for uniform batches; the ragged row mapping is new (CPU-emulated).
   * combine_ragged: attn_decode's fixed-order split merge, writing only the rows a request really has.
2. dequant_tokens: positions [start, start + n) of one request's paged cache -> contiguous bf16 K and V
   (n, KVH, 256), EXACT (code * sf fits bf16), for the prefill path (FlashInfer FA2 over dequantised chunks).
   The global scales are not applied here (the caller folds them into sm_scale / the output, like the decode).
"""

import tilelang
import tilelang.language as T

from .attn_decode_fp4_v2 import V2_SRC, NUM_SMS
from .kv_write_fp4 import FP4_SRC
from .attn_decode import LOG2E

_KERNELS = {}
PASS_CONFIGS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
D = 256

SERVE_SRC = V2_SRC + FP4_SRC + r"""
#ifndef HZ4_SERVE_SRC_DEFINED
#define HZ4_SERVE_SRC_DEFINED
#include <cuda_fp16.h>
namespace hz4 {
struct Strides { long long dsp, dst, dsh, ssp, sst, ssh; };
// exact scalar decodes (normal fp32 only: safe under flush-to-zero)
__device__ __forceinline__ float e2m1v(uint32_t c) {        // 4-bit code, bit 3 = sign
  const uint32_t e = (c >> 1) & 3u, l = c & 1u;
  const float v = e ? __uint_as_float(((e + 126u) << 23) | (l << 22)) : (l ? 0.5f : 0.0f);
  return (c & 8u) ? -v : v;
}
__device__ __forceinline__ float e4m3v(uint32_t byte) {     // e4m3(byte & 0x7f), 0x7f read as 480 (finite)
  const uint32_t e = (byte >> 3) & 15u, m = byte & 7u;
  return e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
}

// TW token groups: the CTA runs TW independent groups of NCW warps; group tg takes the 16-token blocks tg, tg + TW,
// ... of the split, with its own pipeline stages, softmax state and named barrier (id 1 + tg), and the groups are
// merged in fixed order (0, 1, .., TW-1) at the end. TW = 1 is the original kernel. More warps per CTA without more
// split partials: for short contexts, where a CTA's serial 16-token loop (not bandwidth) sets the time.
// two dequantised values -> one 32-bit word of bf16 (F16 = 0) or fp16 (F16 = 1). Both are exact: code * sf needs at
// most 5 significant bits, and |code * sf| lies in [2^-10, 2688] (inside fp16's normal range, and bf16's)
template <int F16>
__device__ __forceinline__ uint32_t pack_out2(float a, float b) {
  if constexpr (F16) { const __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<const uint32_t*>(&h); }
  else { const __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *reinterpret_cast<const uint32_t*>(&h); }
}

// in-register unnormalised FWHT of one 256-row by one warp: lane l holds x[8l .. 8l+7]. Butterfly order of
// kv_fp4_ref.fwht (stage h = 1, 2, .., 128; (x[i], x[i+h]) <- (x[i] + x[i+h], x[i] - x[i+h]) for (i & h) == 0): h = 1, 2, 4
// inside the lane, h = 8m (m = 1 .. 16) with the partner lane l ^ m (the lane with bit m clear holds x[i]). IEEE adds /
// subtracts without contraction on the same operands: bit-identical to the reference.
__device__ __forceinline__ void fwht256_warp(float (&x)[8], int lane) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1)
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if ((j & h) == 0) {
        const float a = x[j], b = x[j + h];
        x[j] = hz_fadd(a, b);
        x[j + h] = hz_fsub(a, b);
      }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool up = (lane & m) != 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float o = __shfl_xor_sync(0xffffffffu, x[j], m);
      x[j] = up ? hz_fsub(o, x[j]) : hz_fadd(x[j], o);
    }
  }
}

// HAD = 1: the keys in the cache are Hadamard-rotated (kv_fp4_ref.rotate); the query rows are rotated here, in shared
// memory, while they are loaded: q' = bf16(FWHT(q * signs) * D^-0.5) exactly as kv_fp4_ref.rotate_q / the separate
// rotate_q kernel, so the output is bit-identical to rotate_q + the HAD = 0 kernel, without that extra launch.
// two e4m3 bytes (byte 1 and byte 3 of t) -> one bf16x2 of their exact values: sign to bit 15, the 7 exponent /
// mantissa bits to bits 10..4 (a bf16 2^-120 times the e4m3 value, subnormals included), then one exact multiply by
// 2^120 (bf16 0x7B80)
__device__ __forceinline__ uint32_t e4m3x2_bf16(uint32_t t) {
  return hmul2((t & 0x80008000u) | ((t & 0x7f007f00u) >> 4), 0x7B807B80u);
}

// K8 = 1 (hz-kv K8V4): keys are FP8 E4M3 (vLLM's fp8 KV bytes: e4m3(k / k_scale), saturating), one 400-byte slot per
// (token, kv head) = [256 B K | 128 B V E2M1 | 16 B V E4M3 block scales]; kd_ / vd_ / vsf_ point at the three parts
// (same strides), ksf_ is unused. Values are the FP4 format unchanged.
template <int KVH, int G, int QLMAX, int NCW, int NST, int MINC, int TW, int HAD, int K8>
__device__ __forceinline__ void attn_task_rg(const void* q_, const void* kd_, const void* ksf_, const void* vd_,
                                             const void* vsf_, const void* bt_, const void* sl_, const void* qsl_,
                                             const void* gk_, const void* gv_, void* opart_, void* lse_,
                                             float scale_log2, int MP, int PS, long long dsp, long long dst,
                                             long long dsh, long long ssp, long long sst, long long ssh,
                                             void* smem_, void* out_, float v_scale, const void* sg_) {
  const Strides sd{dsp, dst, dsh, ssp, sst, ssh};
  constexpr int D = 256, HQ = KVH * G, R = G * QLMAX, NQT = (R + 7) / 8, BH = 32;
  constexpr int DW = D / NCW, RB = DW / 2, VRS = RB + 16, SB = DW / 16, CPL = DW / 4, KW = CPL / 8, KS = DW / 16,
                CPG = DW / 8, MT = DW / 16, VW = (MT + 3) / 4;
  constexpr int RBK = K8 ? DW : RB;                          // K bytes per token row of the slice
  constexpr int STAGE = 16 * RBK + 16 * VRS + (K8 ? 16 : 32) * SB, WARP_SMEM = NST * STAGE;
  static_assert(NCW == 2 || NCW == 4, "channel slices of 128 or 64");
  static_assert(!(K8 && HAD), "the Hadamard rotation is for FP4 keys only");
  static_assert(R <= BH && NST >= 2 && TW >= 1 && TW <= 8, "config");
  constexpr int GT = 32 * NCW, XB = 2 * NCW * NQT * 4 * 32;   // threads per group, xbuf floats per group
  const uint16_t* q = static_cast<const uint16_t*>(q_);
  const uint8_t* kd = static_cast<const uint8_t*>(kd_);
  const uint8_t* ksf = static_cast<const uint8_t*>(ksf_);
  const uint8_t* vd = static_cast<const uint8_t*>(vd_);
  const uint8_t* vsf = static_cast<const uint8_t*>(vsf_);
  const int* bt = static_cast<const int*>(bt_);
  const int* sl = static_cast<const int*>(sl_);
  const int* qsl = static_cast<const int*>(qsl_);
  float* opart = static_cast<float*>(opart_);
  float* lse = static_cast<float*>(lse_);
  uint8_t* smem = static_cast<uint8_t*>(smem_);

  const int lane = threadIdx.x & 31, cs = (threadIdx.x >> 5) % NCW, tg = threadIdx.x / GT, g = lane >> 2, c = lane & 3;
  const int b = blockIdx.x, h = blockIdx.y, s = blockIdx.z, S = gridDim.z;
  const int cb = cs * DW;
  const int q0 = qsl[b];
  int ql = qsl[b + 1] - q0;
  ql = (ql < 0 || ql > QLMAX) ? 0 : ql;          // rows with more tokens are prefills: skipped here
  const int L = ql > 0 ? sl[b] : 0;
  int chunk = (L + S - 1) / S;
  chunk = ((chunk + 15) / 16) * 16;
  chunk = chunk > MINC ? chunk : MINC;
  const int st = s * chunk;
  const int en = min(L, st + chunk);
  const int nblk = en > st ? (en - st + 15) / 16 : 0;
  const int nb = nblk > tg ? (nblk - tg + TW - 1) / TW : 0;   // this group's blocks: tg, tg + TW, ...
  const float sc = scale_log2 * static_cast<const float*>(gk_)[h] * (K8 ? 1.0f : 512.0f);
  const float NEG = -__int_as_float(0x7f800000);

  uint32_t qf[NQT][KS][2];
  if constexpr (HAD) {
    // each warp rotates whole query rows in registers (fwht256_warp; lane = 8 consecutive channels) and stores
    // bf16(x * D^-0.5) row-major in shared memory; the fragments are then read from there exactly like the plain
    // kernel reads global q. The pipeline stages reuse this space afterwards (synchronised below).
    constexpr int NW = NCW * TW;
    uint16_t* QB = reinterpret_cast<uint16_t*>(smem);
    const float* sg = static_cast<const float*>(sg_);
    if (nblk > 0) {                                  // uniform per CTA: empty splits skip the rotation too
      const int wid = threadIdx.x >> 5;
      for (int row = wid; row < R; row += NW) {
        float x[8];
        if (row / G < ql) {
          const uint16_t* qr = q + ((size_t)(q0 + row / G) * HQ + h * G + row % G) * D + 8 * lane;
#pragma unroll
          for (int j = 0; j < 8; ++j) x[j] = hz_fmul(__bfloat162float(__ushort_as_bfloat16(qr[j])), sg[8 * lane + j]);
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) x[j] = 0.0f;
        }
        fwht256_warp(x, lane);
        uint32_t pk[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
          pk[j] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hz_fmul(x[2 * j], 0.0625f))) |
                  ((uint32_t)__bfloat16_as_ushort(__float2bfloat16_rn(hz_fmul(x[2 * j + 1], 0.0625f))) << 16);
        *reinterpret_cast<uint4*>(QB + row * 256 + 8 * lane) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
      }
      __syncthreads();
    }
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) {
      const int row = nt * 8 + g;
      const bool ok = nblk > 0 && row < R && row / G < ql;
      const uint16_t* qr = QB + (ok ? row : 0) * 256;
#pragma unroll
      for (int w = 0; w < KW; ++w)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int ch = cb + CPL * c + 8 * w + 2 * u;
          qf[nt][2 * w + u][0] = ok ? ((uint32_t)qr[ch] | ((uint32_t)qr[ch + 4] << 16)) : 0u;
          qf[nt][2 * w + u][1] = ok ? ((uint32_t)qr[ch + 1] | ((uint32_t)qr[ch + 5] << 16)) : 0u;
        }
    }
    __syncthreads();                                 // before the pipeline overwrites QB
  } else {
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) {
      const int row = nt * 8 + g;
      const bool ok = nblk > 0 && row < R && row / G < ql;     // empty splits skip the Q loads
      const uint16_t* qr = q + ((size_t)(q0 + (ok ? row / G : 0)) * HQ + h * G + (ok ? row % G : 0)) * D;
#pragma unroll
      for (int w = 0; w < KW; ++w)
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int ch = cb + CPL * c + 8 * w + 2 * u;
          qf[nt][2 * w + u][0] = ok ? ((uint32_t)qr[ch] | ((uint32_t)qr[ch + 4] << 16)) : 0u;
          qf[nt][2 * w + u][1] = ok ? ((uint32_t)qr[ch + 1] | ((uint32_t)qr[ch + 5] << 16)) : 0u;
        }
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
  uint8_t* wsm = smem + (tg * NCW + cs) * WARP_SMEM;
  float* xbuf = reinterpret_cast<float*>(smem + TW * NCW * WARP_SMEM) + tg * XB;
  auto issue = [&](int k, int stg) {
    const int p0 = st + 16 * k;
    const long long page = bt[(size_t)b * MP + p0 / PS];
    const long long pos0 = p0 % PS;
    uint8_t* sK = wsm + stg * STAGE;
    uint8_t* sV = sK + 16 * RBK;
    uint8_t* sKS = sV + 16 * VRS;
    uint8_t* sVS = K8 ? sKS : sKS + 16 * SB;
    const uint8_t* gK = kd + page * sd.dsp + pos0 * sd.dst + h * sd.dsh + (K8 ? cb : cb / 2);
    const uint8_t* gV = vd + page * sd.dsp + pos0 * sd.dst + h * sd.dsh + cb / 2;
    constexpr int CH = RB / 16, CHK = RBK / 16;
#pragma unroll
    for (int i = lane; i < 16 * CH; i += 32) {
      const int r = i / CH, k16 = i % CH;
      cp16(sV + r * VRS + k16 * 16, gV + r * sd.dst + k16 * 16);
    }
#pragma unroll
    for (int i = lane; i < 16 * CHK; i += 32) {
      const int r = i / CHK, k16 = i % CHK;
      cp16(sK + r * RBK + k16 * 16, gK + r * sd.dst + k16 * 16);
    }
    const int r = lane & 15;
    if constexpr (K8) {
      if (lane < 16) cpn<SB>(sVS + r * SB, vsf + page * sd.ssp + (pos0 + r) * sd.sst + h * sd.ssh + cs * SB);
    } else {
      if (lane < 16)
        cpn<SB>(sKS + r * SB, ksf + page * sd.ssp + (pos0 + r) * sd.sst + h * sd.ssh + cs * SB);
      else
        cpn<SB>(sVS + r * SB, vsf + page * sd.ssp + (pos0 + r) * sd.sst + h * sd.ssh + cs * SB);
    }
  };
#pragma unroll
  for (int i = 0; i < NST - 1; ++i) {
    if (i < nb) issue(tg + i * TW, i);
    commit();
  }
  for (int k = 0; k < nb; ++k) {
    wait<NST - 2>();
    __syncwarp();
    if (k + NST - 1 < nb) issue(tg + (k + NST - 1) * TW, (k + NST - 1) % NST);
    commit();
    const uint8_t* sK = wsm + (k % NST) * STAGE;
    const uint8_t* sV = sK + 16 * RBK;
    const uint8_t* sKS = sV + 16 * VRS;
    const uint8_t* sVS = K8 ? sKS : sKS + 16 * SB;
    const int p0 = st + 16 * (tg + k * TW);
    float sacc[NQT][4];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) sacc[nt][0] = sacc[nt][1] = sacc[nt][2] = sacc[nt][3] = 0.0f;
    if constexpr (K8) {
      // lane (g, c) holds channels CPL*c .. +CPL-1 of tokens g and g+8 as e4m3 bytes; per 8 channels (w) the pairs
      // (0,4) (1,5) (2,6) (3,7) are the same A-fragment channel order as the FP4 path (and as the q fragments)
      uint32_t k0[2 * KW], k1[2 * KW];
#pragma unroll
      for (int u = 0; u < KW / 2; ++u) {
        const uint4 x0 = *reinterpret_cast<const uint4*>(sK + g * RBK + CPL * c + 16 * u);
        const uint4 x1 = *reinterpret_cast<const uint4*>(sK + (g + 8) * RBK + CPL * c + 16 * u);
        k0[4 * u] = x0.x; k0[4 * u + 1] = x0.y; k0[4 * u + 2] = x0.z; k0[4 * u + 3] = x0.w;
        k1[4 * u] = x1.x; k1[4 * u + 1] = x1.y; k1[4 * u + 2] = x1.z; k1[4 * u + 3] = x1.w;
      }
#pragma unroll
      for (int w = 0; w < KW; ++w) {
        const uint32_t a0 = k0[2 * w], a1 = k0[2 * w + 1], b0 = k1[2 * w], b1 = k1[2 * w + 1];
        const uint32_t P0a = e4m3x2_bf16(__byte_perm(a0, a1, 0x4000)), P0b = e4m3x2_bf16(__byte_perm(b0, b1, 0x4000));
        const uint32_t P1a = e4m3x2_bf16(__byte_perm(a0, a1, 0x5010)), P1b = e4m3x2_bf16(__byte_perm(b0, b1, 0x5010));
        const uint32_t P2a = e4m3x2_bf16(__byte_perm(a0, a1, 0x6020)), P2b = e4m3x2_bf16(__byte_perm(b0, b1, 0x6020));
        const uint32_t P3a = e4m3x2_bf16(__byte_perm(a0, a1, 0x7030)), P3b = e4m3x2_bf16(__byte_perm(b0, b1, 0x7030));
#pragma unroll
        for (int nt = 0; nt < NQT; ++nt) {
          mma(sacc[nt], P0a, P0b, P1a, P1b, qf[nt][2 * w][0], qf[nt][2 * w][1]);
          mma(sacc[nt], P2a, P2b, P3a, P3b, qf[nt][2 * w + 1][0], qf[nt][2 * w + 1][1]);
        }
      }
    } else {
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
      const int blk = (CPL * c + 8 * w) / 16;
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
    }
    float* xb = xbuf + (k & 1) * (NCW * NQT * 4 * 32);
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) xb[((cs * NQT + nt) * 4 + e) * 32 + lane] = sacc[nt][e];
    if constexpr (TW == 1) __syncthreads();
    else asm volatile("bar.sync %0, %1;" :: "r"(tg + 1), "n"(GT) : "memory");
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
        const int tq = row / G;
        const bool ok = row < R && tq < ql && pos < en && pos < L - ql + tq + 1;
        x[nt][e] = ok ? v * sc : NEG;
      }
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
    uint32_t pb[NQT][2];
#pragma unroll
    for (int nt = 0; nt < NQT; ++nt) {
      pb[nt][0] = movt(pack_bf16(x[nt][0], x[nt][1]));
      pb[nt][1] = movt(pack_bf16(x[nt][2], x[nt][3]));
    }
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
    }
  if constexpr (TW > 1) {
    // fixed-order merge of the groups' (m, l, acc) through shared memory (the stage buffers are free now)
    constexpr int NV = 4 * NQT + MT * NQT * 4;
    __syncthreads();
    float* mb = reinterpret_cast<float*>(smem) + (threadIdx.x % GT);
    if (tg > 0) {
      float* o = mb + (size_t)(tg - 1) * NV * GT;
      int k = 0;
#pragma unroll
      for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
        for (int par = 0; par < 2; ++par) {
          o[(k++) * GT] = m[nt][par];
          o[(k++) * GT] = l[nt][par];
        }
#pragma unroll
      for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[(k++) * GT] = acc[j][nt][e];
    }
    __syncthreads();
    if (tg > 0) return;                           // no barrier below this point
#pragma unroll 1
    for (int t2 = 1; t2 < TW; ++t2) {
      const float* o = mb + (size_t)(t2 - 1) * NV * GT;
      float sa[NQT][2], sb[NQT][2];
      int k = 0;
#pragma unroll
      for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
        for (int par = 0; par < 2; ++par) {
          const float m2 = o[(k++) * GT], l2 = o[(k++) * GT];
          const float m1 = m[nt][par], mn = fmaxf(m1, m2);
          const float a = (m1 == NEG) ? 0.0f : exp2f(m1 - mn);
          const float bb = (m2 == NEG) ? 0.0f : exp2f(m2 - mn);
          l[nt][par] = l[nt][par] * a + l2 * bb;
          m[nt][par] = mn;
          sa[nt][par] = a;
          sb[nt][par] = bb;
        }
#pragma unroll
      for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float x2 = o[(k++) * GT];
            acc[j][nt][e] = acc[j][nt][e] * sa[nt][e & 1] + x2 * sb[nt][e & 1];
          }
    }
  }
  if (S == 1) {
    // one split: write the final bf16 output here (the same arithmetic as combine_rg with a single split, whose
    // weight is exactly 1: bf16((acc / l * gv * 512) * v_scale)); the caller launches no combine
    __nv_bfloat16* out = static_cast<__nv_bfloat16*>(out_);
#pragma unroll
    for (int j = 0; j < MT; ++j)
#pragma unroll
      for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int row = nt * 8 + 2 * c + (e & 1);
          const int ch = cb + CPG * g + 2 * j + (e >> 1);
          const float lt = l[nt][e & 1];
          if (row < R && row / G < ql) {
            const float o = (lt > 0.0f ? acc[j][nt][e] / lt * gvs : 0.0f) * v_scale;
            out[((size_t)(q0 + row / G) * HQ + h * G + row % G) * D + ch] = __float2bfloat16_rn(o);
          }
        }
    return;
  }
#pragma unroll
  for (int nt = 0; nt < NQT; ++nt)
#pragma unroll
    for (int par = 0; par < 2; ++par) {
      const float lt = l[nt][par];
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
        // only rows the request really has, only splits that saw tokens (lse = -inf marks the rest; the combine
        // never reads their partials, so nothing else needs to be written)
        if (row < R && row / G < ql && nblk > 0) opart[(base + row) * D + ch] = lt > 0.0f ? acc[j][nt][e] / lt * gvs : 0.0f;
      }
}

// fixed-order merge of the S split partials of one (request, kv head, row); 256 threads = one channel each
// (the same arithmetic as attn_decode.attn_combine_kernel: max, sum of weights, weighted sum, all in split order)
// v2 (29-09-2026): warp 0 reads the S split lse values once (S <= 64), computes the weights with fixed shuffle trees
// and writes the non-empty splits (in split order) to shared memory; every thread then sums only those. Deterministic
// (fixed order), not bit-identical to v1 (tree instead of serial sums of the weights).
template <int KVH, int G, int QLMAX>
__device__ __forceinline__ void combine_rg(const void* opart_, const void* lse_, const void* qsl_, void* out_,
                                           float v_scale, int S, void* smem_) {
  constexpr int D = 256, HQ = KVH * G, BH = 32;
  const float* opart = static_cast<const float*>(opart_);
  const float* lse = static_cast<const float*>(lse_);
  const int* qsl = static_cast<const int*>(qsl_);
  __nv_bfloat16* out = static_cast<__nv_bfloat16*>(out_);
  const int bh = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int b = bh / KVH, h = bh % KVH, t = r / G, gg = r % G;
  const int q0 = qsl[b];
  int ql = qsl[b + 1] - q0;
  ql = (ql < 0 || ql > QLMAX) ? 0 : ql;          // rows with more tokens are prefills: skipped here
  if (t >= ql) return;
  const float NEG = -__int_as_float(0x7f800000);
  const size_t lb = (size_t)(b * KVH + h) * S * BH + r;
  float* wsm = static_cast<float*>(smem_);        // [0, 64): weights, [64, 128): split index (as int), [128]: count
  int* isn = reinterpret_cast<int*>(wsm + 64);
  if (d < 32) {
    const float x0 = d < S ? lse[lb + (size_t)d * BH] : NEG;
    const float x1 = d + 32 < S ? lse[lb + (size_t)(d + 32) * BH] : NEG;
    float mx = fmaxf(x0, x1);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float e0 = x0 == NEG ? 0.0f : exp2f(x0 - mx), e1 = x1 == NEG ? 0.0f : exp2f(x1 - mx);
    float tot = e0 + e1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) tot += __shfl_xor_sync(0xffffffffu, tot, o);
    const unsigned m0 = __ballot_sync(0xffffffffu, x0 != NEG), m1 = __ballot_sync(0xffffffffu, x1 != NEG);
    const unsigned below = (1u << d) - 1u;
    if (x0 != NEG) { const int p = __popc(m0 & below); wsm[p] = e0 / tot; isn[p] = d; }
    if (x1 != NEG) { const int p = __popc(m0) + __popc(m1 & below); wsm[p] = e1 / tot; isn[p] = d + 32; }
    if (d == 0) isn[64] = __popc(m0) + __popc(m1);
  }
  __syncthreads();
  const int n = isn[64];
  float acc = 0.0f;
#pragma unroll 4
  for (int i = 0; i < n; ++i) acc = acc + opart[(lb + (size_t)isn[i] * BH) * D + d] * wsm[i];
  out[((size_t)(q0 + t) * HQ + h * G + gg) * D + d] = __float2bfloat16_rn(acc * v_scale);
}

// exact dequantisation of positions [start, start + n) of one request's paged cache into contiguous bf16 / fp16
// (n, KVH, 256) K and V. grid (ceil(n / 16), KVH, 2 sides), 128 threads: 16 tokens x 128 bytes, 16 bytes / thread.
// 32 e4m3 bytes (two uint4) -> 16 packed pairs of exact values (FP8 keys of the K8V4 layout), no global scale
template <int F16>
__device__ __forceinline__ void dequant_e4m3_32(const uint8_t* src, uint32_t* o) {
  const uint4 a = *reinterpret_cast<const uint4*>(src), b = *reinterpret_cast<const uint4*>(src + 16);
  const uint32_t ws[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
  for (int i = 0; i < 8; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint32_t b0 = (ws[i] >> (16 * j)) & 0xFFu, b1 = (ws[i] >> (16 * j + 8)) & 0xFFu;
      const float v0 = (b0 & 0x80u) ? -e4m3v(b0) : e4m3v(b0), v1 = (b1 & 0x80u) ? -e4m3v(b1) : e4m3v(b1);
      o[2 * i + j] = pack_out2<F16>(v0, v1);
    }
}

template <int KVH, int F16, int K8>
__device__ __forceinline__ void dequant_task(const void* kd_, const void* ksf_, const void* vd_, const void* vsf_,
                                             const void* btrow_, int start, int n, int PS, long long dsp,
                                             long long dst, long long dsh, long long ssp, long long sst,
                                             long long ssh, void* ok_, void* ov_) {
  const int side = blockIdx.z, h = blockIdx.y, tb = blockIdx.x * 16, tid = threadIdx.x;
  const uint8_t* data = static_cast<const uint8_t*>(side ? vd_ : kd_);
  const uint8_t* scl = static_cast<const uint8_t*>(side ? vsf_ : ksf_);
  const int* btrow = static_cast<const int*>(btrow_);
  uint32_t* out = static_cast<uint32_t*>(side ? ov_ : ok_);
  const int tok = tb + tid / 8, part = tid % 8;        // 8 threads per token, 16 bytes (32 values) each
  if (tok >= n) return;
  const long long p = start + tok;
  const long long page = btrow[p / PS], pos = p % PS;
  if (K8 && side == 0) {
    dequant_e4m3_32<F16>(data + page * dsp + pos * dst + h * dsh + part * 32, out + ((size_t)tok * KVH + h) * 128 + part * 16);
    return;
  }
  const uint4 w = *reinterpret_cast<const uint4*>(data + page * dsp + pos * dst + h * dsh + part * 16);
  const float sf0 = e4m3v(scl[page * ssp + pos * sst + h * ssh + 2 * part]);
  const float sf1 = e4m3v(scl[page * ssp + pos * sst + h * ssh + 2 * part + 1]);
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  uint32_t* o = out + ((size_t)tok * KVH + h) * 128 + part * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float sc = i < 2 ? sf0 : sf1;                // bytes 0..7 of the part = block 2*part, 8..15 = next
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (ws[i] >> (8 * j)) & 0xFFu;
      o[4 * i + j] = pack_out2<F16>(e2m1v(byte & 15u) * sc, e2m1v(byte >> 4) * sc);   // exact
    }
  }
}

// KV write, 4 threads per 16-value block (one per scale candidate). grid (tokens), KVH * 128 threads.
// Bit-exact to kv_write_fp4 / kv_fp4_ref.write_reference: the same per-candidate arithmetic and the same candidate
// order with strict < (k = 0 first). Each thread then encodes 4 of the 16 values.
// HAD = 1: keys are rotated first (k' = FWHT(k * signs) * D^-0.5 in fp32, kv_fp4_ref.rotate), in shared memory by the
// whole CTA (one butterfly per thread per stage), then quantised from fp32. Values are never rotated.
// K8 = 1: keys are written as FP8 E4M3 exactly like vLLM's fp8 reshape_and_cache_flash (e4m3(k / k_scale), round to
// nearest even, saturating to +-448; gk_ = k_scale per head), 4 bytes per thread; values as above.
template <int KVH, int SEARCH, int HAD, int K8>
__device__ __forceinline__ void write_task(const void* key_, const void* val_, long long ks0, long long ks1,
                                           long long vs0, long long vs1, void* kd_, void* ksf_, void* vd_, void* vsf_,
                                           const void* slots_, const void* gk_, const void* gv_, int PS, long long dsp,
                                           long long dst, long long dsh, long long ssp, long long sst, long long ssh,
                                           const void* sg_, void* xsm_) {
  const int t = blockIdx.x, tid = threadIdx.x, cand = tid & 3, bg = tid >> 2;   // bg: 0 .. 2*KVH*16-1
  const long long slot = static_cast<const long long*>(slots_)[t];
  if (slot < 0) return;                           // uniform per CTA (one token per CTA)
  float* XS = static_cast<float*>(xsm_);
  if constexpr (HAD) {                            // warp w < KVH rotates key row w in registers, then shares it
    const int wid = tid >> 5, lane = tid & 31;
    if (wid < KVH) {
      const __nv_bfloat16* kp = static_cast<const __nv_bfloat16*>(key_) + t * ks0 + wid * ks1 + 8 * lane;
      const float* sg = static_cast<const float*>(sg_);
      float x[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) x[j] = hz_fmul(__bfloat162float(kp[j]), sg[8 * lane + j]);
      fwht256_warp(x, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j) XS[wid * 256 + 8 * lane + j] = x[j];
    }
    __syncthreads();
  }
  const int side = bg / (KVH * 16), h = (bg % (KVH * 16)) / 16, blk = bg % 16;
  const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(side ? val_ : key_);
  const long long s0 = side ? vs0 : ks0, s1 = side ? vs1 : ks1;
  const float g = static_cast<const float*>(side ? gv_ : gk_)[h];
  float xs[16];
  if (HAD && side == 0) {
    const float* xr = XS + h * 256 + blk * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) xs[i] = hz_fmul(xr[i], 0.0625f);     // * D^-0.5, exact power of two
  } else {
    const __nv_bfloat16* xp = src + t * s0 + h * s1 + blk * 16;
#pragma unroll
    for (int i = 0; i < 16; ++i) xs[i] = __bfloat162float(xp[i]);
  }
  if (K8 && side == 0) {                          // whole warps (the K side is KVH * 64 threads)
    uint32_t w = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float x = cand == 0 ? xs[j] : cand == 1 ? xs[4 + j] : cand == 2 ? xs[8 + j] : xs[12 + j];
      w |= (uint32_t)__nv_cvt_float_to_fp8(__fdiv_rn(x, g), __NV_SATFINITE, __NV_E4M3) << (8 * j);
    }
    const long long page = slot / PS, pos = slot % PS;
    *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(kd_) + page * dsp + pos * dst + h * dsh + blk * 16 + 4 * cand) = w;
    return;
  }
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(xs[i]));
  const int c0 = (int)hz_fp4_c6(amax, g);
  int cb = c0;
  if (SEARCH) {
    const int cc = cand == 0 ? c0 : cand == 1 ? min(c0 + 1, 126) : cand == 2 ? max(c0 - 1, 0) : (int)hz_fp4_c4(amax, g);
    const float sc = hz_fp4_scale(cc, g);
    float err = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) err = hz_fadd(err, hz_fp4_sqerr(xs[i], sc));
    const int lane0 = (threadIdx.x & 31) & ~3;
    float best = __shfl_sync(0xffffffffu, err, lane0);
    cb = __shfl_sync(0xffffffffu, cc, lane0);
#pragma unroll
    for (int k = 1; k < 4; ++k) {
      const float ek = __shfl_sync(0xffffffffu, err, lane0 + k);
      const int ck = __shfl_sync(0xffffffffu, cc, lane0 + k);
      if (ek < best) { best = ek; cb = ck; }
    }
  }
  const float sb = hz_fp4_scale(cb, g);
  const long long page = slot / PS, pos = slot % PS;
  uint8_t* data = static_cast<uint8_t*>(side ? vd_ : kd_) + page * dsp + pos * dst + h * dsh + blk * 8 + 2 * cand;
  float x4[4];                                    // this thread's 4 values, selected in registers (no local array)
#pragma unroll
  for (int j = 0; j < 4; ++j)
    x4[j] = cand == 0 ? xs[j] : cand == 1 ? xs[4 + j] : cand == 2 ? xs[8 + j] : xs[12 + j];
  const uint32_t b0 = hz_fp4_code(x4[0], sb) | (hz_fp4_code(x4[1], sb) << 4);
  const uint32_t b1 = hz_fp4_code(x4[2], sb) | (hz_fp4_code(x4[3], sb) << 4);
  data[0] = (uint8_t)b0;
  data[1] = (uint8_t)b1;
  if (cand == 0) static_cast<uint8_t*>(side ? vsf_ : ksf_)[page * ssp + pos * sst + h * ssh + blk] = (uint8_t)cb;
}

// exact dequantisation of several requests' full contexts into ONE packed bf16 buffer (row r occupies
// [cu[r], cu[r+1]) of the output, positions 0 .. cu[r+1]-cu[r]-1 of request rows[r]). grid (ceil(N / 16), KVH, 2).
template <int KVH, int F16, int K8>
__device__ __forceinline__ void dequant_multi_task(const void* kd_, const void* ksf_, const void* vd_,
                                                   const void* vsf_, const void* bt_, int MP, const void* rows_,
                                                   const void* cu_, int NR, int N, int PS, long long dsp,
                                                   long long dst, long long dsh, long long ssp, long long sst,
                                                   long long ssh, void* ok_, void* ov_) {
  const int side = blockIdx.z, h = blockIdx.y, tid = threadIdx.x;
  const int tok = blockIdx.x * 16 + tid / 8, part = tid % 8;
  if (tok >= N) return;
  const int* cu = static_cast<const int*>(cu_);
  int lo = 0, hi = NR;                            // largest r with cu[r] <= tok
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (cu[mid] <= tok) lo = mid; else hi = mid; }
  const int req = static_cast<const int*>(rows_)[lo];
  const long long p = tok - cu[lo];
  const long long page = static_cast<const int*>(bt_)[(size_t)req * MP + p / PS], pos = p % PS;
  const uint8_t* data = static_cast<const uint8_t*>(side ? vd_ : kd_);
  if (K8 && side == 0) {
    dequant_e4m3_32<F16>(data + page * dsp + pos * dst + h * dsh + part * 32,
                         static_cast<uint32_t*>(ok_) + ((size_t)tok * KVH + h) * 128 + part * 16);
    return;
  }
  const uint8_t* scl = static_cast<const uint8_t*>(side ? vsf_ : ksf_);
  const uint4 w = *reinterpret_cast<const uint4*>(data + page * dsp + pos * dst + h * dsh + part * 16);
  const float sf0 = e4m3v(scl[page * ssp + pos * sst + h * ssh + 2 * part]);
  const float sf1 = e4m3v(scl[page * ssp + pos * sst + h * ssh + 2 * part + 1]);
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  uint32_t* o = static_cast<uint32_t*>(side ? ov_ : ok_) + ((size_t)tok * KVH + h) * 128 + part * 16;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float sc = i < 2 ? sf0 : sf1;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t byte = (ws[i] >> (8 * j)) & 0xFFu;
      o[4 * i + j] = pack_out2<F16>(e2m1v(byte & 15u) * sc, e2m1v(byte >> 4) * sc);
    }
  }
}
}  // namespace hz4
#endif
"""


def default_ncw(G: int, QLMAX: int) -> int:
    return 2 if (G * QLMAX + 7) // 8 <= 3 else 4


def smem_bytes(G, QLMAX, NCW, NST, TW=1, HAD=0, K8=0):
    NQT = (G * QLMAX + 7) // 8
    DW = D // NCW
    RB, SB = DW // 2, DW // 16
    RBK = DW if K8 else RB
    loop = TW * (NCW * NST * (16 * RBK + 16 * (RB + 16) + (16 if K8 else 32) * SB) + 2 * NCW * NQT * 4 * 32 * 4)
    merge = (TW - 1) * (4 * NQT + (DW // 16) * NQT * 4) * 32 * NCW * 4     # group merge reuses the stage buffers
    qrot = G * QLMAX * D * 2 if HAD else 0                                  # fused q rotation (bf16 rows) too
    return max(loop, merge, qrot)


@tilelang.jit(pass_configs=PASS_CONFIGS)
def decode_ragged_kernel(KVH: int, G: int, QLMAX: int, NCW: int = 2, NST: int = 3, MINC: int = 64, TW: int = 1,
                         HAD: int = 0, K8: int = 0):
    """grid (B, KVH, S), TW * NCW warps. Cache parts are flat uint8 buffers addressed with runtime strides.
    HAD = 1: the query is rotated inside the kernel (keys written with the same signs). K8 = 1: FP8 keys (K8V4 slot);
    kd / vd / vsf are the slot's K, V-data and V-scale parts, ksf is ignored (pass vsf)."""
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    MP = T.dynamic("MP")
    S = T.dynamic("S")
    NBK, NBV = T.dynamic("NBK"), T.dynamic("NBV")      # flat buffers start at different offsets: own lengths
    NSK, NSV = T.dynamic("NSK"), T.dynamic("NSV")
    NO = T.dynamic("NO")                                 # the output buffer may be longer than the query (padding)
    HQ = KVH * G
    f32, i32, i64 = T.float32, T.int32, T.int64
    SM = smem_bytes(G, QLMAX, NCW, NST, TW, HAD, K8)
    fn = f"hz4::attn_task_rg<{KVH}, {G}, {QLMAX}, {NCW}, {NST}, {MINC}, {TW}, {HAD}, {K8}>"

    @T.prim_func
    def main(q: T.Tensor((NT, HQ, D), T.bfloat16), kd: T.Tensor((NBK,), T.uint8), ksf: T.Tensor((NSK,), T.uint8),
             vd: T.Tensor((NBV,), T.uint8), vsf: T.Tensor((NSV,), T.uint8), bt: T.Tensor((B, MP), i32),
             sl: T.Tensor((B,), i32), qsl: T.Tensor((B + 1,), i32), gk: T.Tensor((KVH,), f32),
             gv: T.Tensor((KVH,), f32), opart: T.Tensor((B, KVH, S, 32, D), f32), lse: T.Tensor((B, KVH, S, 32), f32),
             out: T.Tensor((NO, HQ, D), T.bfloat16), sg: T.Tensor((D,), f32), scale_log2: f32, v_scale: f32,
             PS: i32, dsp: i64, dst: i64, dsh: i64, ssp: i64, sst: i64, ssh: i64):
        with T.Kernel(B, KVH, S, threads=32 * NCW * TW, prelude=SERVE_SRC) as (b, h, s):
            smem = T.alloc_shared((SM,), T.uint8)
            T.call_extern("handle", fn, T.address_of(q[0, 0, 0]), T.address_of(kd[0]), T.address_of(ksf[0]),
                          T.address_of(vd[0]), T.address_of(vsf[0]), T.address_of(bt[0, 0]), T.address_of(sl[0]),
                          T.address_of(qsl[0]), T.address_of(gk[0]), T.address_of(gv[0]),
                          T.address_of(opart[0, 0, 0, 0, 0]), T.address_of(lse[0, 0, 0, 0]), scale_log2, MP, PS,
                          dsp, dst, dsh, ssp, sst, ssh, T.address_of(smem[0]), T.address_of(out[0, 0, 0]), v_scale,
                          T.address_of(sg[0]))

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def combine_ragged_kernel(KVH: int, G: int, QLMAX: int):
    NT = T.dynamic("NT")
    B = T.dynamic("B")
    S = T.dynamic("S")
    HQ = KVH * G
    R = G * QLMAX
    f32, i32 = T.float32, T.int32
    fn = f"hz4::combine_rg<{KVH}, {G}, {QLMAX}>"

    @T.prim_func
    def main(opart: T.Tensor((B, KVH, S, 32, D), f32), lse: T.Tensor((B, KVH, S, 32), f32),
             qsl: T.Tensor((B + 1,), i32), out: T.Tensor((NT, HQ, D), T.bfloat16), v_scale: f32):
        with T.Kernel(B * KVH, R, threads=D, prelude=SERVE_SRC) as (bh, r):
            wsm = T.alloc_shared((132,), f32)
            T.call_extern("handle", fn, T.address_of(opart[0, 0, 0, 0, 0]), T.address_of(lse[0, 0, 0, 0]),
                          T.address_of(qsl[0]), T.address_of(out[0, 0, 0]), v_scale, S, T.address_of(wsm[0]))

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def dequant_kernel(KVH: int, F16: int = 0, K8: int = 0):
    NBK, NBV = T.dynamic("NBK"), T.dynamic("NBV")
    NSK, NSV = T.dynamic("NSK"), T.dynamic("NSV")
    MP = T.dynamic("MP")
    NO = T.dynamic("NO")
    f32, i32, i64 = T.float32, T.int32, T.int64
    fn = f"hz4::dequant_task<{KVH}, {F16}, {K8}>"
    odt = T.float16 if F16 else T.bfloat16

    @T.prim_func
    def main(kd: T.Tensor((NBK,), T.uint8), ksf: T.Tensor((NSK,), T.uint8), vd: T.Tensor((NBV,), T.uint8),
             vsf: T.Tensor((NSV,), T.uint8), btrow: T.Tensor((MP,), i32), ok: T.Tensor((NO, KVH, D), odt),
             ov: T.Tensor((NO, KVH, D), odt), start: i32, n: i32, PS: i32, dsp: i64, dst: i64, dsh: i64,
             ssp: i64, sst: i64, ssh: i64):
        with T.Kernel(T.ceildiv(n, 16), KVH, 2, threads=128, prelude=SERVE_SRC) as (x, y, z):
            T.call_extern("handle", fn, T.address_of(kd[0]), T.address_of(ksf[0]), T.address_of(vd[0]),
                          T.address_of(vsf[0]), T.address_of(btrow[0]), start, n, PS,
                          dsp, dst, dsh, ssp, sst, ssh, T.address_of(ok[0, 0, 0]), T.address_of(ov[0, 0, 0]))

    return main


def _get(key, make):
    k = _KERNELS.get(key)
    if k is None:
        k = _KERNELS[key] = make()
    return k


def get_decode(KVH, G, QLMAX, NCW=None, NST=3, MINC=64, TW=1, HAD=0, K8=0):
    NCW = NCW or default_ncw(G, QLMAX)
    return _get(("dr", KVH, G, QLMAX, NCW, NST, MINC, TW, HAD, K8),
                lambda: decode_ragged_kernel(KVH, G, QLMAX, NCW, NST, MINC, TW, HAD, K8))


def _ones(device):
    import torch
    k = ("ones", str(device))
    t = _KERNELS.get(k)
    if t is None:
        t = _KERNELS[k] = torch.ones(D, dtype=torch.float32, device=device)
    return t


def get_combine(KVH, G, QLMAX):
    return _get(("cr", KVH, G, QLMAX), lambda: combine_ragged_kernel(KVH, G, QLMAX))


def get_dequant(KVH, f16=0, k8=0):
    f16, k8 = int(bool(f16)), int(bool(k8))
    return _get(("dq", KVH, f16, k8), lambda: dequant_kernel(KVH, f16, k8))


def flat_views(k_data, k_sf, v_data, v_sf):
    """[page, position, head, byte] views -> (flat uint8 base tensors, strides). The flat tensors start at each view's
    first byte and span its whole storage extent, so kernels can address any page with the runtime strides."""
    import torch

    def flat(v):
        u = v.view(torch.uint8)
        # declared length capped below 2^31 (TileLang's int32 shape); kernels address with 64-bit offsets from the
        # base pointer, so pages past the declared length are still reached (no bounds checks on raw pointers)
        n = min(u.untyped_storage().nbytes() - u.storage_offset(), 2 ** 31 - 1)
        return torch.as_strided(u, (n,), (1,), u.storage_offset())
    assert k_data.stride() == v_data.stride() and k_sf.stride() == v_sf.stride()
    st = tuple(int(x) for x in (*k_data.stride()[:3], *k_sf.stride()[:3]))
    return (flat(k_data), flat(k_sf), flat(v_data), flat(v_sf)), st


def plan_splits(B, KVH, ctas_per_sm=4, max_splits=64):
    """depends on the batch size only (CUDA-graph safe: the kernel sizes each request's chunks itself)"""
    return max(1, min(max_splits, NUM_SMS * ctas_per_sm // max(1, B * KVH)))


def decode_ragged(query, flat, strides, PS, block_tables, seq_lens, query_start_loc, bmm1_scale, bmm2_scale, gk, gv,
                  qlmax, out, workspace, splits=None, nst=3, ncw=None, min_chunk=64, tw=1, signs=None, k8=False):
    """query (NT, HQ, 256) bf16 contiguous (rotated when keys are); flat / strides from flat_views; block_tables
    (B, MP) int32; seq_lens (B,) int32; query_start_loc (B + 1,) int32; out (NT, HQ, 256) bf16 (rows of absent query
    tokens are left untouched); workspace: float32 tensor with >= B * KVH * S * 32 * 257 elements."""
    B = block_tables.shape[0]
    KVH = gk.shape[0]
    HQ = query.shape[1]
    G = HQ // KVH
    S = splits or plan_splits(B, KVH)
    assert S <= 64, "combine stages at most 64 splits"
    need = B * KVH * S * 32 * D
    op = workspace[:need].view(B, KVH, S, 32, D)
    ls = workspace[need:need + B * KVH * S * 32].view(B, KVH, S, 32)
    kd, ksf, vd, vsf = flat
    had = signs is not None
    get_decode(KVH, G, qlmax, ncw, nst, min_chunk, tw, int(had), int(bool(k8)))(
        query, kd, ksf, vd, vsf, block_tables, seq_lens, query_start_loc, gk, gv, op, ls, out,
        signs if had else _ones(query.device), float(bmm1_scale) * LOG2E, float(bmm2_scale), int(PS), *strides)
    if S > 1:                                   # one split: the decode kernel wrote the output itself
        get_combine(KVH, G, qlmax)(op, ls, query_start_loc, out, float(bmm2_scale))
    return out


def dequant_tokens(flat, strides, PS, block_table_row, start, n, out_k, out_v, k8=False):
    """positions [start, start + n) of one request -> out_k / out_v[:n] (n, KVH, 256) bf16 or fp16 (the output dtype
    picks the variant), exact (no global scale)"""
    import torch
    kd, ksf, vd, vsf = flat
    get_dequant(out_k.shape[1], out_k.dtype == torch.float16, k8)(kd, ksf, vd, vsf, block_table_row, out_k, out_v, int(start), int(n), int(PS), *strides)
    return out_k[:n], out_v[:n]


@tilelang.jit(pass_configs=PASS_CONFIGS)
def write_kernel(KVH: int, SEARCH: int = 1, HAD: int = 0, K8: int = 0):
    """flat bf16 key / value buffers with runtime element strides (ks0, ks1) / (vs0, vs1); cache parts as flat uint8
    buffers with runtime byte strides. grid (slots), KVH * 128 threads. HAD = 1 rotates the keys (sg = signs)."""
    NK, NV = T.dynamic("NK"), T.dynamic("NV")
    NBK, NBV = T.dynamic("NBK"), T.dynamic("NBV")
    NSK, NSV = T.dynamic("NSK"), T.dynamic("NSV")
    NSL = T.dynamic("NSL")
    f32, i32, i64 = T.float32, T.int32, T.int64
    fn = f"hz4::write_task<{KVH}, {SEARCH}, {HAD}, {K8}>"

    @T.prim_func
    def main(key: T.Tensor((NK,), T.bfloat16), val: T.Tensor((NV,), T.bfloat16), kd: T.Tensor((NBK,), T.uint8),
             ksf: T.Tensor((NSK,), T.uint8), vd: T.Tensor((NBV,), T.uint8), vsf: T.Tensor((NSV,), T.uint8),
             slots: T.Tensor((NSL,), T.int64), gk: T.Tensor((KVH,), f32), gv: T.Tensor((KVH,), f32),
             sg: T.Tensor((D,), f32), ks0: i64, ks1: i64, vs0: i64, vs1: i64, PS: i32, dsp: i64, dst: i64, dsh: i64,
             ssp: i64, sst: i64, ssh: i64):
        with T.Kernel(NSL, threads=KVH * 128, prelude=SERVE_SRC) as t:
            xsm = T.alloc_shared((KVH * D if HAD else 1,), f32)
            T.call_extern("handle", fn, T.address_of(key[0]), T.address_of(val[0]), ks0, ks1, vs0, vs1,
                          T.address_of(kd[0]), T.address_of(ksf[0]), T.address_of(vd[0]), T.address_of(vsf[0]),
                          T.address_of(slots[0]), T.address_of(gk[0]), T.address_of(gv[0]), PS, dsp, dst, dsh, ssp,
                          sst, ssh, T.address_of(sg[0]), T.address_of(xsm[0]))

    return main


@tilelang.jit(pass_configs=PASS_CONFIGS)
def dequant_multi_kernel(KVH: int, F16: int = 0, K8: int = 0):
    NBK, NBV = T.dynamic("NBK"), T.dynamic("NBV")
    NSK, NSV = T.dynamic("NSK"), T.dynamic("NSV")
    B, MP, NO = T.dynamic("B"), T.dynamic("MP"), T.dynamic("NO")
    NR1, NRR = T.dynamic("NR1"), T.dynamic("NRR")
    i32, i64 = T.int32, T.int64
    fn = f"hz4::dequant_multi_task<{KVH}, {F16}, {K8}>"
    odt = T.float16 if F16 else T.bfloat16

    @T.prim_func
    def main(kd: T.Tensor((NBK,), T.uint8), ksf: T.Tensor((NSK,), T.uint8), vd: T.Tensor((NBV,), T.uint8),
             vsf: T.Tensor((NSV,), T.uint8), bt: T.Tensor((B, MP), i32), rows: T.Tensor((NRR,), i32),
             cu: T.Tensor((NR1,), i32), ok: T.Tensor((NO, KVH, D), odt), ov: T.Tensor((NO, KVH, D), odt),
             NR: i32, N: i32, PS: i32, dsp: i64, dst: i64, dsh: i64, ssp: i64, sst: i64, ssh: i64):
        with T.Kernel(T.ceildiv(N, 16), KVH, 2, threads=128, prelude=SERVE_SRC) as (x, y, z):
            T.call_extern("handle", fn, T.address_of(kd[0]), T.address_of(ksf[0]), T.address_of(vd[0]),
                          T.address_of(vsf[0]), T.address_of(bt[0, 0]), MP, T.address_of(rows[0]), T.address_of(cu[0]),
                          NR, N, PS, dsp, dst, dsh, ssp, sst, ssh, T.address_of(ok[0, 0, 0]), T.address_of(ov[0, 0, 0]))

    return main


def get_write(KVH, search=1, had=0, k8=0):
    a = (int(bool(search)), int(bool(had)), int(bool(k8)))
    return _get(("wr", KVH) + a, lambda: write_kernel(KVH, *a))


def get_dequant_multi(KVH, f16=0, k8=0):
    f16, k8 = int(bool(f16)), int(bool(k8))
    return _get(("dqm", KVH, f16, k8), lambda: dequant_multi_kernel(KVH, f16, k8))


def _flat_elems(t):
    """a strided view -> flat 1-D view from its first element (length < 2^31) + its element strides"""
    import torch
    n = min(t.untyped_storage().nbytes() // t.element_size() - t.storage_offset(), 2 ** 31 - 1)
    return torch.as_strided(t, (n,), (1,), t.storage_offset())


def write(key, value, flat, strides, PS, slot_mapping, gk, gv, search=True, signs=None, k8=False):
    """key / value (NT, KVH, 256) bf16 views (any strides, last dim contiguous); slot_mapping (NSL,) int64 (slot < 0
    skipped; tokens beyond NSL untouched); flat / strides from flat_views. Bit-exact to kv_write_fp4.write
    (no Hadamard); signs (D,) fp32 +-1 -> keys rotated (bit-exact to kv_fp4_ref.write_reference with signs)."""
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    KVH = key.shape[1]
    kd, ksf, vd, vsf = flat
    had = signs is not None
    assert not (had and k8)
    get_write(KVH, search, had, k8)(_flat_elems(key), _flat_elems(value), kd, ksf, vd, vsf, slot_mapping, gk, gv,
                                signs if had else _ones(key.device), key.stride(0), key.stride(1), value.stride(0),
                                value.stride(1), int(PS), *strides)


def dequant_multi(flat, strides, PS, block_tables, rows, cu, n_rows, n_tokens, out_k, out_v, k8=False):
    """requests rows[r] (int32, device) full contexts -> out_k / out_v[cu[r]:cu[r+1]] (bf16 or fp16 by the output
    dtype, exact, no global scale); cu (n_rows + 1,) int32 device, cu[n_rows] == n_tokens"""
    import torch
    kd, ksf, vd, vsf = flat
    get_dequant_multi(out_k.shape[1], out_k.dtype == torch.float16, k8)(kd, ksf, vd, vsf, block_tables, rows, cu, out_k, out_v, int(n_rows),
                                      int(n_tokens), int(PS), *strides)
