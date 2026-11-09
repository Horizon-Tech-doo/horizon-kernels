# hipBLASLt in the same allocation, fp16 and cross-check through the TileLang wrapper. Timing only here.
import json,sys,time,os
import torch,tilelang
sys.path.insert(0,'.')
from gemm_prefetch import gemm
round_=int(sys.argv[1]); which=sys.argv[2] if len(sys.argv)>2 else 'std'
def bench(fn,iters,warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/iters
torch.manual_seed(0)
for M,N,K,it in [(4096,4096,4096,40),(8192,8192,8192,20),(32768,1024,2048,40),(4096,17408,5120,20)]:
    a=torch.randn(M,K,device='cuda',dtype=torch.float16); b=torch.randn(N,K,device='cuda',dtype=torch.float16)
    torch.backends.cuda.preferred_blas_library('cublaslt')
    ms=bench(lambda:a@b.T,it)
    print(json.dumps(dict(impl='hipblaslt',round=round_,M=M,N=N,K=K,ms=round(ms,4),TF=round(2*M*N*K/ms*1e-9,2))),flush=True)
    k=gemm(M,N,K,128,128,64,256,2,'base')
    ms=bench(lambda:k(a,b),it)
    print(json.dumps(dict(impl='tl_wrapper_b12864',round=round_,M=M,N=N,K=K,ms=round(ms,4),TF=round(2*M*N*K/ms*1e-9,2))),flush=True)
    if round_==0:
        ref=(a.float()@b.float().T); rel=((k(a,b).float()-ref).norm()/ref.norm()).item(); print(json.dumps(dict(impl='tl_wrapper_b12864_check',M=M,N=N,K=K,rel_err=rel)),flush=True)
    del a,b
    torch.cuda.empty_cache()
