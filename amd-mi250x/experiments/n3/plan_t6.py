import json,sys
B=int(sys.argv[1]); SS=json.loads(sys.argv[2]); S0=SS[0]
def arm(n,t,d,S,TH=128,BM=32): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM)
P=[arm("base","base",17408,S0)]
for S in SS: P+=[arm(f"bnb3_{S}","b+nb",17408,S),arm(f"bnb4_{S}","b+nb",16384,S),arm(f"f643_{S}","b+nob",21504,S,64,16),arm(f"ld4_{S}","ld",16384,S)]
print(json.dumps(P))
