# Compile-only: staging locals as one small array per 16-byte vector, written only with TileLang. Looks at ScratchSize of generated code.
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch,tilelang
import tilelang.rocm.language as T
@tilelang.jit(out_idx=[-1])
def rs2(M,N,K,bM,bN,bK,threads,kp):
    dtype,acc=T.float16,T.float32
    nk=K//bK
    VA=max(1,bM*bK//8//threads); VB=max(1,bN*bK//8//threads)
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype),B:T.Tensor((N,K),dtype),C:T.Tensor((M,N),dtype)):
        with T.Kernel(T.ceildiv(N,bN),T.ceildiv(M,bM),threads=threads) as (bx,by):
            A_s=T.alloc_shared((bM,bK),dtype); B_s=T.alloc_shared((bN,bK),dtype)
            a0=T.alloc_local((8,),dtype); a1=T.alloc_local((8,),dtype); b0=T.alloc_local((8,),dtype)
            C_l=T.alloc_fragment((bM,bN),acc)
            tx=T.get_thread_binding()
            T.clear(C_l)
            for j in T.vectorized(8): a0[j]=A[by*bM+((0*threads+tx)*8)//bK,((0*threads+tx)*8)%bK+j]
            for j in T.vectorized(8): a1[j]=A[by*bM+((1*threads+tx)*8)//bK,((1*threads+tx)*8)%bK+j]
            for j in T.vectorized(8): b0[j]=B[bx*bN+((0*threads+tx)*8)//bK,((0*threads+tx)*8)%bK+j]
            for k in T.serial(nk):
                for j in T.vectorized(8): A_s[((0*threads+tx)*8)//bK,((0*threads+tx)*8)%bK+j]=a0[j]
                for j in T.vectorized(8): A_s[((1*threads+tx)*8)//bK,((1*threads+tx)*8)%bK+j]=a1[j]
                for j in T.vectorized(8): B_s[((0*threads+tx)*8)//bK,((0*threads+tx)*8)%bK+j]=b0[j]
                if k+1<nk:
                    for j in T.vectorized(8): a0[j]=A[by*bM+((0*threads+tx)*8)//bK,(k+1)*bK+((0*threads+tx)*8)%bK+j]
                    for j in T.vectorized(8): a1[j]=A[by*bM+((1*threads+tx)*8)//bK,(k+1)*bK+((1*threads+tx)*8)%bK+j]
                    for j in T.vectorized(8): b0[j]=B[bx*bN+((0*threads+tx)*8)//bK,(k+1)*bK+((0*threads+tx)*8)%bK+j]
                T.gemm(A_s,B_s,C_l,transpose_B=True,k_pack=kp)
            T.copy(C_l,C[by*bM,bx*bN])
    return main
k=rs2(8192,8192,8192,256,128,32,512,2)
s=k.get_kernel_source(); open('rs2_8k.cu','w').write(s)
print(s[:1800])
