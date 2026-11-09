# step 5: fp16 vs bf16, same tile, TileLang wrapper and hipBLASLt, interleaved rounds.
import json,sys,os,glob
import torch,tilelang
sys.path.insert(0,'.')
from bench_gemm import matmul,bench
torch.manual_seed(0)
CFGS=[(128,128,64,256,0,2,0),(256,128,32,256,0,2,0),(256,128,32,256,0,2,8)]
for (M,N,K),its in [((8192,8192,8192),15),((16384,16384,16384),4)]:
    data={d:(torch.randn(M,K,device='cuda',dtype=getattr(torch,d)),torch.randn(N,K,device='cuda',dtype=getattr(torch,d))) for d in ['float16','bfloat16']}
    ker={}
    for d in data:
        for c in CFGS:
            ker[(d,c)]=matmul(M,N,K,c[0],c[1],c[2],d,c[3],c[4],c[5],c[6])
    for d,(a,b) in data.items():
        ref=a.float()@b.float().T
        for c in CFGS:
            r=ker[(d,c)](a,b); print(json.dumps(dict(check=d,M=M,cfg=c,rel=(r.float()-ref).norm().item()/ref.norm().item())),flush=True)
        del ref,r
    for rnd in range(3):
        for d in (['float16','bfloat16'] if rnd%2==0 else ['bfloat16','float16']):
            a,b=data[d]
            torch.backends.cuda.preferred_blas_library('cublaslt')
            ms=bench(lambda:a@b.T,iters=its,warmup=3)
            print(json.dumps(dict(impl='hipblaslt',dtype=d,round=rnd,M=M,N=N,K=K,ms=round(ms,4),TF=round(2*M*N*K/ms*1e-9,2))),flush=True)
            for c in CFGS:
                k=ker[(d,c)]
                ms=bench(lambda:k(a,b),iters=its,warmup=3)
                print(json.dumps(dict(impl='tl',dtype=d,round=rnd,M=M,N=N,K=K,cfg=c,ms=round(ms,4),TF=round(2*M*N*K/ms*1e-9,2))),flush=True)
    del data,ker
    torch.cuda.empty_cache()
