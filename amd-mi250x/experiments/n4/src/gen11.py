import sys; sys.path.insert(0,"../../scripts")
import gemm_prefetch as g
for sn,(M,N,K) in {"6k":(6144,6144,6144),"16k":(16384,16384,16384)}.items():
    n=f"h10_p16_{sn}"; src,dyn=g.hand_source(M,N,K,entry=n,pad=16); open(f"h10/{n}.cu","w").write(src)
    a=open(f"arms10_{sn}.txt").read().strip().split(" "); a=[x for x in a if not x.startswith(n+",")]+[f"{n},h10/{n}.co,256,128,256,{dyn}"]; open(f"arms10_{sn}.txt","w").write(" ".join(a)+"\n"); print(n,dyn)
