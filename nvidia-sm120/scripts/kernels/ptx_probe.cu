// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
// Probe SM120 PTX features on driver 570 + cuda-compat: e2m1x2 cvt, cp.async.bulk (no tensor map), mbarrier,
// and block-scaled FP4 MMA (mma.sync kind::mxf4nvf4). Each test writes a result; host checks it.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;}}while(0)

__global__ void t_cvt(uint32_t* out) {            // e2m1x2 -> f16x2 hardware convert
  uint16_t packed = 0x2A;                           // lo nibble 0xA = -1.0, hi nibble 0x2 = 1.0
  uint32_t r;
  asm volatile("{ .reg .b8 b; mov.b16 {b, _}, %1; cvt.rn.f16x2.e2m1x2 %0, b; }" : "=r"(r) : "h"(packed));
  *out = r;
}

__global__ void t_bulk(const int4* __restrict__ src, int4* dst) {   // cp.async.bulk global->shared with mbarrier
  __shared__ alignas(128) int4 buf[256];
  __shared__ alignas(8) uint64_t bar;
  uint32_t sb = (uint32_t)__cvta_generic_to_shared(&bar), sd = (uint32_t)__cvta_generic_to_shared(buf);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sb));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sb), "r"(4096));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(sd), "l"(src), "r"(4096), "r"(sb) : "memory");
  }
  uint32_t done = 0;
  while (!done) asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0,1,0,p; }" : "=r"(done) : "r"(sb));
  dst[threadIdx.x] = buf[threadIdx.x];
}

// m16n8k64 block-scaled FP4 MMA: A 16x64 e2m1, B 64x8 e2m1, scales ue4m3 per 16 elements (scale_vec::4X).
// All A,B nibbles = 1.0 (0x2), all scales = 1.0 (ue4m3 0x38) -> every D element should be 64.0.
__global__ void t_mma(float* out) {
  uint32_t a[4], b[2]; float d[4] = {0,0,0,0};
  for (int i = 0; i < 4; i++) a[i] = 0x22222222u;
  for (int i = 0; i < 2; i++) b[i] = 0x22222222u;
  uint32_t sa = 0x38383838u, sb = 0x38383838u;
  asm volatile(
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
  for (int i = 0; i < 4; i++) out[threadIdx.x * 4 + i] = d[i];
}

int main() {
  uint32_t* dc; CK(cudaMalloc(&dc, 4)); t_cvt<<<1,1>>>(dc); CK(cudaDeviceSynchronize());
  uint32_t hc; CK(cudaMemcpy(&hc, dc, 4, cudaMemcpyDeviceToHost));
  __half2 h = *reinterpret_cast<__half2*>(&hc);
  printf("cvt e2m1x2: lo=%.2f hi=%.2f  -> %s\n", __low2float(h), __high2float(h),
         (__low2float(h) == -1.f && __high2float(h) == 1.f) ? "OK" : "WRONG");

  int4 *s, *d; CK(cudaMalloc(&s, 4096)); CK(cudaMalloc(&d, 4096));
  int hs[1024], hd[1024]; for (int i = 0; i < 1024; i++) hs[i] = i * 7 + 1;
  CK(cudaMemcpy(s, hs, 4096, cudaMemcpyHostToDevice)); t_bulk<<<1,256>>>(s, d); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd, d, 4096, cudaMemcpyDeviceToHost)); int bad = 0; for (int i = 0; i < 1024; i++) bad += hs[i] != hd[i];
  printf("cp.async.bulk + mbarrier (no tensor map): %s\n", bad ? "WRONG" : "OK");

  float* dm; CK(cudaMalloc(&dm, 32 * 4 * 4)); t_mma<<<1,32>>>(dm); CK(cudaDeviceSynchronize());
  float hm[128]; CK(cudaMemcpy(hm, dm, sizeof hm, cudaMemcpyDeviceToHost)); bad = 0;
  for (int i = 0; i < 128; i++) bad += hm[i] != 64.f;
  printf("mma.sync kind::mxf4nvf4 block_scale m16n8k64: d[0]=%.1f -> %s\n", hm[0], bad ? "WRONG" : "OK");
  return 0;
}
