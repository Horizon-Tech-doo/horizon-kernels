// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
// dequant cost probe (27-09-2026, CPU-only SASS count): regfrag's twiddled chunk vs a native #3257-layout chunk
#include <cuda_bf16.h>

__device__ __forceinline__ unsigned tl_rf_mul_lo(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {l, l};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
__device__ __forceinline__ unsigned tl_rf_mul_hi(unsigned a, unsigned p) {
  unsigned d;
  asm("{\n\t.reg .b16 l, h;\n\t.reg .b32 b;\n\tmov.b32 {l, h}, %2;\n\tmov.b32 b, {h, h};\n\t"
      "mul.rn.bf16x2 %0, %1, b;\n\t}" : "=r"(d) : "r"(a), "r"(p));
  return d;
}
template <typename TQ, typename TS, typename TO>
__device__ __forceinline__ void tl_rf_dq_chunk(TQ *qp, TS *sp, TO *op) {
  const unsigned *q = reinterpret_cast<const unsigned *>(qp);
  const unsigned *s = reinterpret_cast<const unsigned *>(sp);
  unsigned *out = reinterpret_cast<unsigned *>(op);
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const unsigned sw = s[st];
    // bf16x2 {scale(n_a), scale(n_b)} * 2^119 for n16 tile j = 0 (bytes 0, 2) and j = 1 (bytes 1, 3)
    const unsigned p0 = __byte_perm(sw, 0u, 0x4240u) * 16u + 0x70007000u;
    const unsigned p1 = __byte_perm(sw, 0u, 0x4341u) * 16u + 0x70007000u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned x = q[2 * st + j];
      const unsigned p = j ? p1 : p0;
      const unsigned ra = x & 0x81C081C0u;
      const unsigned rb = (x << 3) & 0x81C081C0u;
      const unsigned rc = (x << 6) & 0x81C081C0u;
      const unsigned rd = ((x << 1) & 0x80008000u) | ((x >> 3) & 0x01800180u) | ((x >> 7) & 0x00400040u);
      out[8 * st + 4 * j + 0] = tl_rf_mul_lo(ra, p);
      out[8 * st + 4 * j + 1] = tl_rf_mul_lo(rb, p);
      out[8 * st + 4 * j + 2] = tl_rf_mul_hi(rc, p);
      out[8 * st + 4 * j + 3] = tl_rf_mul_hi(rd, p);
    }
  }
}

__device__ __forceinline__ unsigned nat_conv(unsigned x) {       // nibbles 0 / 4 (standard e2m1) -> bf16x2 fp4*2^-126
  return ((x << 6) & 0x01C001C0u) | ((x << 12) & 0x80008000u);
}
// native layout: the 1 KB (32 rows x 64 k) atom tile in shared memory, [row atom j][FP4 lane][r0 r1 r2 r3]; this lane
// (g, t) needs, for k16 step st of 32-k chunk h (h = 0: r0/r1, 1: r2/r3), the half (t & 1) of FP4 lane 2 st + (t >> 1)
// with a decode-friendly in-group permutation (k0 -> nibble 0, k0+1 -> 4, k0+8 -> 1, k0+9 -> 5 of that byte pair)
__device__ __forceinline__ void nat_dq_chunk(const unsigned *tile, int h, const unsigned *s, unsigned *out, int lane) {
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int st = 0; st < 2; ++st) {
    const unsigned sw = s[st];
    const unsigned p0 = __byte_perm(sw, 0u, 0x4240u) * 16u + 0x70007000u;
    const unsigned p1 = __byte_perm(sw, 0u, 0x4341u) * 16u + 0x70007000u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint2 w = *reinterpret_cast<const uint2 *>(tile + j * 128 + 4 * (4 * g + 2 * st + (t >> 1)) + 2 * h);
      const unsigned sh = 8 * (t & 1);
      const unsigned xa = w.x >> sh, xb = w.y >> sh;
      const unsigned p = j ? p1 : p0;
      out[8 * st + 4 * j + 0] = tl_rf_mul_lo(nat_conv(xa), p);
      out[8 * st + 4 * j + 1] = tl_rf_mul_lo(nat_conv(xa >> 4), p);
      out[8 * st + 4 * j + 2] = tl_rf_mul_hi(nat_conv(xb), p);
      out[8 * st + 4 * j + 3] = tl_rf_mul_hi(nat_conv(xb >> 4), p);
    }
  }
}
// one lane, 8 blocks (32 rows x 32 k each): load + dequant; results summed so nothing is dead code
extern "C" __global__ void cur_kernel(const unsigned *__restrict__ q, const unsigned *__restrict__ s, unsigned *o) {
  __shared__ unsigned wq[8 * 128], ws[8 * 16];
  for (int i = threadIdx.x; i < 8 * 128; i += 32) wq[i] = q[i];
  for (int i = threadIdx.x; i < 8 * 16; i += 32) ws[i] = s[i];
  __syncwarp();
  const int lane = threadIdx.x & 31;
  unsigned acc = 0;
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const uint4 v = *reinterpret_cast<const uint4 *>(wq + b * 128 + 4 * lane);
    const uint2 sv = *reinterpret_cast<const uint2 *>(ws + b * 16 + (lane >> 2) * 2);
    unsigned qq[4] = {v.x, v.y, v.z, v.w}, ss[2] = {sv.x, sv.y}, out[16];
    tl_rf_dq_chunk(qq, ss, out);
#pragma unroll
    for (int r = 0; r < 16; ++r) acc ^= out[r];
  }
  o[threadIdx.x] = acc;
}
extern "C" __global__ void nat_kernel(const unsigned *__restrict__ q, const unsigned *__restrict__ s, unsigned *o) {
  __shared__ unsigned wq[8 * 128], ws[8 * 16];
  for (int i = threadIdx.x; i < 8 * 128; i += 32) wq[i] = q[i];
  for (int i = threadIdx.x; i < 8 * 16; i += 32) ws[i] = s[i];
  __syncwarp();
  const int lane = threadIdx.x & 31;
  unsigned acc = 0;
#pragma unroll
  for (int b = 0; b < 8; ++b) {
    const uint2 sv = *reinterpret_cast<const uint2 *>(ws + b * 16 + (lane >> 2) * 2);
    unsigned ss[2] = {sv.x, sv.y}, out[16];
    nat_dq_chunk(wq + (b >> 1) * 256, b & 1, ss, out, lane);
#pragma unroll
    for (int r = 0; r < 16; ++r) acc ^= out[r];
  }
  o[threadIdx.x] = acc;
}
