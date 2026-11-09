
__device__ __forceinline__ int horizon_pair_rank(int expert) {
  unsigned peers = __match_any_sync(0xffffffffu, expert);
  unsigned lane = threadIdx.x & 31;
  return __popc(peers & ((1u << lane) - 1u));
}
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/atomic.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(int* __restrict__ expert_ids, const int* __restrict__ ids, int* __restrict__ ntp, int* __restrict__ sorted_ids);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(int* __restrict__ expert_ids, const int* __restrict__ ids, int* __restrict__ ntp, int* __restrict__ sorted_ids) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* hist = ((void*)((char*)buf_dyn_shmem + 0));
  void* counts = ((void*)((char*)buf_dyn_shmem + 256));
  void* offsets = ((void*)((char*)buf_dyn_shmem + 384));
  int ex = 0;
  int total = 0;
  int rank = 0;
  if (((int)threadIdx.x) < 64) {
    ((int*)hist)[((int)threadIdx.x)] = 0;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 33) {
    ex = ids[((int)threadIdx.x)];
    if ((0 <= ex) && (ex < 32)) {
      AtomicAdd((&(((int*)hist)[(((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32) + ((int64_t)ex))])), 1);
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    total = 0;
    for (int c = 0; c < 2; ++c) {
      total = (total + ((int*)hist)[((c * 32) + ((int)threadIdx.x))]);
    }
    ((int*)counts)[((int)threadIdx.x)] = (((total + 31) >> 5) * 32);
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    total = 0;
    for (int e = 0; e < 32; ++e) {
      ((int*)offsets)[e] = total;
      total = (total + ((int*)counts)[e]);
    }
    ((int*)offsets)[32] = total;
    ntp[0] = total;
  }
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 1025) {
      sorted_ids[((i * 256) + ((int)threadIdx.x))] = 33;
    }
  }
  if (((int)threadIdx.x) < 33) {
    expert_ids[((int)threadIdx.x)] = -1;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 32) {
    total = (((int*)counts)[((int)threadIdx.x)] >> 5);
    for (int b = 0; b < total; ++b) {
      if (0 <= ((((int*)offsets)[((int)threadIdx.x)] >> 5) + b)) {
        if (((((int*)offsets)[((int)threadIdx.x)] >> 5) + b) < 33) {
          expert_ids[((((int64_t)((int*)offsets)[((int64_t)((int)threadIdx.x))]) >> (int64_t)5) + ((int64_t)b))] = ((int)threadIdx.x);
        }
      }
    }
  }
  ex = 32;
  if (((int)threadIdx.x) < 33) {
    ex = ids[((int)threadIdx.x)];
  }
  rank = horizon_pair_rank(ex);
  if (((((int)threadIdx.x) < 33) && (0 <= ex)) && (ex < 32)) {
    rank = (rank + ((int*)hist)[ex]);
  }
  if (((((int)threadIdx.x) < 33) && (0 <= ex)) && (ex < 32)) {
    if (0 <= (((int*)offsets)[ex] + rank)) {
      if ((((int*)offsets)[ex] + rank) < 1025) {
        sorted_ids[(((int*)offsets)[ex] + rank)] = ((int)threadIdx.x);
      }
    }
  }
}

