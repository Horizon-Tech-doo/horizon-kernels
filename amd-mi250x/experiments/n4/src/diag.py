# Why does the rule kernel read 94 TF in 'auto' and 104 TF in 'sweep' at 8192^3? arms from argv, one process, same data, many rounds.
import json,sys,torch
from gemm_prefetch import gemm,bench
M,N,K=8192,8192,8192
mode=sys.argv[1]
torch.manual_seed(0)
a=torch.randn(M,K,device="cuda",dtype=torch.float16); b=torch.randn(N,K,device="cuda",dtype=torch.float16)
ref=a.float()@b.float().T
if mode=='noref': del ref
k=gemm(M,N,K,128,128,64,256,2,"base",16,"column")
k0=gemm(M,N,K,128,128,64,256,2,"base")
keep=None
if mode=='keepc': keep=k(a,b)
arms={'auto':[('hb',lambda:a@b.T),('rule',lambda:k(a,b))],'noref':[('hb',lambda:a@b.T),('rule',lambda:k(a,b))],'solo':[('rule',lambda:k(a,b))],'three':[('hb',lambda:a@b.T),('b0',lambda:k0(a,b)),('rule',lambda:k(a,b))],'keepc':[('hb',lambda:a@b.T),('rule',lambda:k(a,b))]}[mode]
out=torch.zeros(M,N,device="cuda",dtype=torch.float16)
for r in range(8):
    for i in range(len(arms)):
        nm,fn=arms[(i+r)%len(arms)]
        ms=bench(fn,10,3)
        print(json.dumps(dict(impl=nm,mode=mode,round=r,pos=i,M=M,N=N,K=K,ms=round(ms,4),TF=round(2*M*N*K/ms*1e-9,2))),flush=True)
