import json
def arm(n,t,d,S=13,TH=128,BM=32,**k): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM,**k)
G="gen/form_%s.cu"
P=[arm("base","base",17408),arm("f643","b+nob",21504,TH=64,BM=16),arm("bnb3","b+nb",17408)]
for n,S,sw in [("a13",13,0),("a13rc",13,0),("a20rc",20,0),("a26rc",26,0),("s13rc",13,1),("tp12_4",16,1),("tp10_6",16,1),("tp11_8",19,1),("tp8_16",24,1),("tpn12_4",16,0)]:
    P.append(arm(n,"base+swap" if sw else "base",17408,S=S,gen=G%n))
print(json.dumps(P))
