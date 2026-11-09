import sys; sys.path.insert(0,"../../scripts")
import gemm_prefetch as g
for sn,(M,N,K) in {"4k":(4096,4096,4096),"8k":(8192,8192,8192),"17k":(4096,17408,5120),"32k":(32768,1024,2048),"4kk":(16384,4096,4096)}.items():
    n=f"h10_o128c8_{sn}"; src,dyn=g.hand_source(M,N,K,entry=n,bM=128,bN=256); open(f"h10/{n}.cu","w").write(src); print(n,dyn)
