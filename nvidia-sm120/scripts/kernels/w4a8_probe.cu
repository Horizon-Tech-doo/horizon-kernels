// SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
// SPDX-License-Identifier: Apache-2.0
// Which FP8-activation x FP4-weight MMA forms does ptxas accept for sm_120a?
#include <cstdint>
__global__ void k(float* o, uint32_t sa, uint32_t sb) {
  uint32_t a[4] = {0,0,0,0}, b[2] = {0,0}; float d[4] = {0,0,0,0};
#if VARIANT == 1   // MX: e4m3 x e2m1, ue8m0 per-32 scales (scale_vec::1X)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#elif VARIANT == 2 // NV-style: e4m3 x e2m1 with ue4m3 scales
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue4m3 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{0,0},%11,{0,0};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
#elif VARIANT == 3 // unscaled: e4m3 x e2m1 (kind::f8f6f4, scales applied in software)
  asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#endif
  o[threadIdx.x] = d[0] + d[1] + d[2] + d[3];
}
