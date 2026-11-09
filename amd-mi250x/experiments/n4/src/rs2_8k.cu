#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/scan.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

extern "C" __global__ void __launch_bounds__(512) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_s = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_s = ((void*)((char*)buf_dyn_shmem + 16384));
  float C_l[64];
  half_t a0[8];
  half_t a1[8];
  half_t b0[8];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0.000000e+00f;
    *(float4*)(C_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(uint4*)(a0 + 0) = *(uint4*)(A + (((((int)blockIdx.y) * 2097152) + ((((int)threadIdx.x) >> 2) * 8192)) + ((((int)threadIdx.x) & 3) * 8)));
  *(uint4*)(a1 + 0) = *(uint4*)(A + ((((((int)blockIdx.y) * 2097152) + ((((int)threadIdx.x) >> 2) * 8192)) + ((((int)threadIdx.x) & 3) * 8)) + 1048576));
  *(uint4*)(b0 + 0) = *(uint4*)(B + (((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + ((((int)threadIdx.x) & 3) * 8)));
  for (int k = 0; k < 256; ++k) {
    __syncthreads();
    *(uint4*)(((half_t*)A_s) + ((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(a0 + 0);
    *(uint4*)(((half_t*)A_s) + (((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096)) = *(uint4*)(a1 + 0);
    *(uint4*)(((half_t*)B_s) + ((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(b0 + 0);
    if (k < 255) {
      *(uint4*)(a0 + 0) = *(uint4*)(A + (((((((int)blockIdx.y) * 2097152) + ((((int)threadIdx.x) >> 2) * 8192)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
      *(uint4*)(a1 + 0) = *(uint4*)(A + (((((((int)blockIdx.y) * 2097152) + ((((int)threadIdx.x) >> 2) * 8192)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 1048608));
      *(uint4*)(b0 + 0) = *(uint4*)(B + (((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) >> 2) * 8192)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    {
      half_t A_local[32];
      half_t B_local[32];
      __syncthreads();
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        *(uint4*)(A_local + (i_1 * 8)) = *(uint4*)(((half_t*)A_s) + (((((((((int)threadIdx.x) & 255) >> 6) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)));
      }
      for (int j = 0; j < 4; ++j) {
        *(uint4*)(B_local + (j * 8)) = *(uint4*)(((half_t*)B_s) + ((((((((int)threadIdx.x) >> 8) * 2048) + (j * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)));
      }
      for (int kp = 0; kp < 2; ++kp) {
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            {
      *(((float32x4*)C_l) + ((i_2 * 4) + j_1)) = __builtin_amdgcn_mfma_f32_16x16x16f16(*(((float16x4*)B_local) + ((j_1 * 2) + kp)),
                    *(((float16x4*)A_local) + ((i_2 * 2) + kp)),
                    *(((float32x4*)C_l) + ((i_2 * 4) + j_1)), 0, 0, 0);
    };
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    half_t C_local_cast[4];
    uint2 __1;
    float4 v_ = *(float4*)(C_l + (i_3 * 4));
    *((half_t*)(&(((half2*)(&(__1.x)))->x))) = (half_t)(v_.x);
    *((half_t*)(&(((half2*)(&(__1.x)))->y))) = (half_t)(v_.y);
    *((half_t*)(&(((half2*)(&(__1.y)))->x))) = (half_t)(v_.z);
    *((half_t*)(&(((half2*)(&(__1.y)))->y))) = (half_t)(v_.w);
    *(uint2*)(C_local_cast + 0) = __1;
    *(uint2*)(C + ((((((((((int)blockIdx.y) * 2097152) + (((((int)threadIdx.x) & 255) >> 6) * 524288)) + ((i_3 >> 2) * 131072)) + ((((int)threadIdx.x) & 15) * 8192)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 8) * 64)) + ((i_3 & 3) * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(C_local_cast + 0);
  }
}

