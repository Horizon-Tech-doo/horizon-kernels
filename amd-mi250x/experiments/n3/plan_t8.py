import json
def arm(n,t,d,S=13,TH=128,BM=32,**k): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM,**k)
G="gen/form_%s.cu"
P=[arm("base","base",17408)]
for S in (13,20,26):
    P+=[arm(f"bnb3rc_{S}","b+nb+rc",17408,S=S),arm(f"bnb4rc_{S}","b+nb+rc",16384,S=S),arm(f"bw14rc_{S}","b+w1+rc",16384,S=S),arm(f"f643rc_{S}","b+nob+rc",21504,S=S,TH=64,BM=16),arm(f"f644rc_{S}","b+nob+rc",16384,S=S,TH=64,BM=16)]
P+=[arm("tp10_6",  "base+swap",17408,S=16,gen=G%"tp10_6"),arm("tp10_6d4","base+swap",16384,S=16,gen=G%"tp10_6"),arm("a13rcd4","base",16384,S=13,gen=G%"a13rc")]
print(json.dumps(P))
