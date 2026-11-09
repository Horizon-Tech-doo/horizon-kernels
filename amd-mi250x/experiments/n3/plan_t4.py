import json
def arm(n,t,d,S=13,TH=128,BM=32): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM)
P=[]
for S in (13,20,26,39,52,80,104):
    P+=[arm(f"bnb3_{S}","b+nb",17408,S=S),arm(f"f643_{S}","b+nob",21504,S=S,TH=64,BM=16),arm(f"f64s3_{S}","b+nob+swap",21504,S=S,TH=64,BM=16),arm(f"ld4_{S}","ld",16384,S=S),arm(f"l64x_{S}","ld",0,S=S,TH=64,BM=16),arm(f"f644_{S}","b+nob",16384,S=S,TH=64,BM=16),arm(f"bnb4_{S}","b+nb",16384,S=S)]
P+=[arm("base","base",17408),arm("f643c","b+nob+ck",21504,TH=64,BM=16),arm("bnb3c","b+nb+ck",17408),arm("basec","base+ck",17408)]
print(json.dumps(P))
