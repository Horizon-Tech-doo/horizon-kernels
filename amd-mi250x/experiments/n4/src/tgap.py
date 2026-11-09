# Torch side of the same-job gap series. One process per shape. randn data, hipBLASLt through torch, wall clock around 'it' calls.
import json,sys,time,torch
M,N,K,rounds,it,warm=map(int,sys.argv[1:7])
torch.manual_seed(0)
a=torch.randn(M,K,device='cuda',dtype=torch.float16); b=torch.randn(N,K,device='cuda',dtype=torch.float16)
torch.backends.cuda.preferred_blas_library('cublaslt')
c=torch.zeros(M,N,device='cuda',dtype=torch.float16); bt=b.T
fl=2*M*N*K
def arm_at(): return a@b.T
def arm_bt(): return a@bt
def arm_out(): return torch.matmul(a,bt,out=c)
arms=[('torch_at',arm_at),('torch_out',arm_out),('torch_bt',arm_bt)]
for r in range(rounds):
    for i in range(len(arms)):
        nm,fn=arms[(i+r)%len(arms)]
        for _ in range(warm): fn()
        torch.cuda.synchronize(); s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t0=time.perf_counter(); s.record()
        for _ in range(it): fn()
        e.record(); torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/it*1e3
        print(json.dumps(dict(impl=nm,round=r,pos=i,M=M,N=N,K=K,data='randn',ms=round(ms,4),TF=round(fl/ms*1e-9,2),ev_ms=round(s.elapsed_time(e)/it,4))),flush=True)
