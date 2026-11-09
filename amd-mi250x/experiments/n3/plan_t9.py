import json
def arm(n,t,d,S=13,TH=128,BM=32,**k): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM,**k)
P=[arm("base","base",17408),arm("bnb3rc","b+nb+rc",17408),arm("bnb4rc","b+nb+rc",16384),arm("bnb3vf","b+nb+rc+vf",17408),arm("bnb4vf","b+nb+rc+vf",16384),
   arm("f643rc","b+nob+rc",21504,TH=64,BM=16),arm("f644rc","b+nob+rc",16384,TH=64,BM=16),arm("f643vf","b+nob+rc+vf",21504,TH=64,BM=16),arm("f644vf","b+nob+rc+vf",16384,TH=64,BM=16),
   arm("ld3","ld",17408),arm("ld4","ld",16384)]
print(json.dumps(P))
