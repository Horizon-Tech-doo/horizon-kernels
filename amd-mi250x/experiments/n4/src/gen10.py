# Writes the step 2 arms as standalone sources in h10/ and arms_<shape>.txt (harness arm strings). Generator: hand_source in scripts/gemm_prefetch.py.
import sys; sys.path.insert(0,"../../scripts")
import gemm_prefetch as g
F={"c8":{}, "c4":dict(sw="c4"), "r8":dict(sw="r8"), "c16":dict(sw="c16"),
   "o128c8":dict(bM=128,bN=256), "o128r8":dict(bM=128,bN=256,sw="r8"),
   "bb16":dict(b_base=16), "bb32":dict(b_base=32), "bb64":dict(b_base=64), "bb1024":dict(b_base=1024),
   "a144b176":dict(pad=16,pad_b=48), "a176b144":dict(pad=48,pad_b=16), "a160b144":dict(pad=32,pad_b=16), "a144b160":dict(pad=16,pad_b=32), "a160b176":dict(pad=32,pad_b=48),
   "guard":dict(guard=True)}
SH={"6k":(6144,6144,6144),"12k":(12288,12288,12288),"16k":(16384,16384,16384),"32k":(32768,1024,2048)}
for sn,(M,N,K) in SH.items():
    arms=[]
    for f,kw in F.items():
        if sn=="32k" and f not in ("c8","guard"): continue
        n=f"h10_{f}_{sn}"; src,dyn=g.hand_source(M,N,K,entry=n,**kw); open(f"h10/{n}.cu","w").write(src)
        bM,bN=kw.get("bM",256),kw.get("bN",128); arms.append(f"{n},h10/{n}.co,{bM},{bN},256,{dyn}")
    open(f"arms10_{sn}.txt","w").write(" ".join(arms)+"\n")
print("arms ok")
