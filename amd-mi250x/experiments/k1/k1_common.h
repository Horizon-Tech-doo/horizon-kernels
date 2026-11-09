// K1: common helpers for direct global-to-LDS experiments on gfx90a.
#include <hip/hip_runtime.h>
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
typedef int i32x4_t __attribute__((ext_vector_type(4)));
#define FI __device__ inline __attribute__((always_inline))
#define LDSDECL extern __shared__ __align__(1024) uchar buf_dyn_shmem[]
FI uint lane_id(){ return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)); }
FI void wait_vm0(){ asm volatile("s_waitcnt vmcnt(0)" : : : "memory"); }
template<int N> FI void wait_vm(){ asm volatile("s_waitcnt vmcnt(%0)" : : "n"(N) : "memory"); }
FI void hz5(){ asm volatile("s_nop 4" : : : "memory"); }   // VALU-writes-SGPR -> VMEM-reads-SGPR: 5 wait states
// One direct load with the documented safe form: one wait state between the M0 write and the load.
FI void ldl(uint m0v, uint voff, const i32x4_t& rs){
  asm volatile("s_mov_b32 m0, %0\n\ts_nop 0\n\tbuffer_load_dword %1, %2, 0 offen lds" : : "s"(m0v), "v"(voff), "s"(rs) : "memory");
}
