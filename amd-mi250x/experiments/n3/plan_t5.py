import json
def arm(n,t,d,S=13,TH=128,BM=32,**k): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM,**k)
G="gen/form_%s.cu"
P=[arm("base","base",17408),arm("bnb3","b+nb",17408),arm("f643","b+nob",21504,TH=64,BM=16),
   arm("v2_t64if","base",16640+256,TH=64,gen=G%"t64_if"),arm("v2_t64plain","base",16640+256,TH=64,gen=G%"t64_plain"),arm("v2_t128if","base",17408,gen=G%"t128_if"),arm("v2_t128plain","base",17408,gen=G%"t128_plain")]
print(json.dumps(P))
