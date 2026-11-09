# Does a normal T.Kernel with prelude + extern call give the launch we need (args, threads, dyn LDS)? Compile only, HIP.
import sys, tilelang, tilelang.rocm.language as T
src = r'''
__device__ inline __attribute__((always_inline)) void tk(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C, uchar* L) {
  C[blockIdx.x+blockIdx.y*256+threadIdx.x] = A[0]; }
'''
@tilelang.jit(out_idx=[-1])
def f1(M,N,K):
    @T.prim_func
    def main(A:T.Tensor((M,K),T.float16),B:T.Tensor((N,K),T.float16),C:T.Tensor((M,N),T.float16)):
        with T.Kernel(T.ceildiv(N,128),T.ceildiv(M,256),threads=256,prelude=src) as (bx,by):
            L=T.alloc_shared((61440,),T.uint8)
            T.call_extern("handle","tk",T.address_of(A[0,0]),T.address_of(B[0,0]),T.address_of(C[0,0]),T.address_of(L[0]))
    return main
k=f1(512,512,128)
print(type(k))
print(k.get_kernel_source() if hasattr(k,'get_kernel_source') else '')
print(getattr(k,'kernel_source',None))
for a in dir(k):
    if 'source' in a: print('ATTR',a)
