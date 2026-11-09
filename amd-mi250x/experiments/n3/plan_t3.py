import json
def arm(n,t,d,S=13,TH=128,BM=32): return dict(name=n,toks=t,dyn=d,S=S,TH=TH,BM=BM)
P=[arm("base","base",17408),arm("bnb","b+nb",17408),arm("bnbs","b+nb+swap",17408),arm("bnb2","b+nb",32768),arm("bnbs2","b+nb+swap",32768)]
for d,tag in [(32768,"2"),(17408,"3"),(16384,"4"),(0,"x")]:
    P+= [arm("ld"+tag,"ld",d),arm("lds"+tag,"ld+swap",d),arm("ldh"+tag,"ld+hnd",d),arm("ldhs"+tag,"ld+hnd+swap",d)]
for d,tag in [(32768,"2"),(17408,"3"),(16384,"4"),(8192,"8"),(0,"x")]:
    P+= [arm("l64"+tag,"ld",d,TH=64,BM=16),arm("l64s"+tag,"ld+swap",d,TH=64,BM=16)]
for d,tag in [(32768,"2"),(17408,"3"),(0,"x")]:
    P+= [arm("l256"+tag,"ld",d,TH=256,BM=64),arm("l256s"+tag,"ld+swap",d,TH=256,BM=64)]
P+= [arm("f64","b+nob",16384,TH=64,BM=16),arm("f64s","b+nob+swap",16384,TH=64,BM=16),arm("f643","b+nob",21504,TH=64,BM=16),arm("f64s3","b+nob+swap",21504,TH=64,BM=16),arm("f642","b+nob",32768,TH=64,BM=16),
     arm("f256","b+nb",17408,TH=256,BM=64),arm("f2562","b+nb",32768,TH=256,BM=64),arm("f256s2","b+nb+swap",32768,TH=256,BM=64)]
print(json.dumps(P))
