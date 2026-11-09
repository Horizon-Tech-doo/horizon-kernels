# Compile-only: can TileLang itself emit a padded LDS row? Prints the LDS addressing of a generated K loop, or the error.
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch,tilelang
import tilelang.rocm.language as T
print([x for x in dir(T) if 'view' in x.lower() or 'layout' in x.lower() or 'pad' in x.lower() or 'align' in x.lower()])
@tilelang.jit(out_idx=[-1])
def p1(M,N,K,bM,bN,bK,th,kp,pk,form):
    dtype,acc=T.float16,T.float32; nk=K//bK
    @T.prim_func
    def a(A:T.Tensor((M,K),dtype),B:T.Tensor((N,K),dtype),C:T.Tensor((M,N),dtype)):
        with T.Kernel(T.ceildiv(N,bN),T.ceildiv(M,bM),threads=th) as (bx,by):
            A_s=T.alloc_shared((bM,bK+pk),dtype); B_s=T.alloc_shared((bN,bK+pk),dtype)
            C_l=T.alloc_fragment((bM,bN),acc)
            T.use_swizzle(panel_size=8,order="column")
            T.clear(C_l)
            for k in T.serial(nk):
                T.copy(A[by*bM:by*bM+bM,k*bK:k*bK+bK],A_s[0:bM,0:bK])
                T.copy(B[bx*bN:bx*bN+bN,k*bK:k*bK+bK],B_s[0:bN,0:bK])
                T.gemm(A_s[0:bM,0:bK],B_s[0:bN,0:bK],C_l,transpose_B=True,k_pack=kp)
            T.copy(C_l,C[by*bM,bx*bN])
    return a
try:
    k=p1(8192,8192,8192,256,128,32,512,2,8,'a'); s=k.get_kernel_source(); open('pad_8k.cu','w').write(s)
    import re
    print(s[s.index('for (int k'):][:1500])
except Exception as e:
    print('ERR',type(e).__name__,str(e)[-600:])
