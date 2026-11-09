import json
c={"S":13,"TH":128,"BM":32}
P=[("base","base",17408),("base16","base",16384),("ns","ns",17408),("b","b",17408),("bnb","b+nb",17408),("bnb16","b+nb",16384),("w1","w1",17408),("bw1","b+w1",17408),("bw116","b+w1",16384),("bbt1","b+nb+bt1",16384),("ld","ld",0),("ld16","ld",16384),("ldl","ldl",16384),("basec","base+ck",17408),("bnb16c","b+nb+ck",16384),("bw116c","b+w1+ck",16384),("ld16c","ld+ck",16384)]
print(json.dumps([dict(name=n,toks=t,dyn=d,**c) for n,t,d in P]))
