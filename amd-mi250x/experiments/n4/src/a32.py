import json,glob,re,sys,statistics as st,collections as cc
O=sys.argv[1]
def load(tag):
    out={}
    for f in sorted(glob.glob(f"{O}/{tag}_d*.jsonl")):
        d=re.search(r"_d(\d)",f).group(1); out[d]=[json.loads(l) for l in open(f) if l.startswith("{")]
    return out
print("PHASE a: step q ratio o128/hb per card; hb read at same step")
for d,r in load("a16").items():
    hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
    print(" d"+d," ".join(f"{x['q']}{x['TF']/hb[x['step']]:.3f}" for x in r if "o128" in x.get("impl","")))
tr=cc.defaultdict(lambda:[0,0])
for d,r in load("a16").items():
    hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}; a=[x for x in r if "o128" in x.get("impl","")]
    for p,x in zip(a,a[1:]):
        s0=p["TF"]/hb[p["step"]]; s1=x["TF"]/hb[x["step"]]; tr[x["q"]][0]+=1; tr[x["q"]][1]+= (s0<1.0)!=(s1<1.0)
print("flips across 1.0 by step type",dict(tr))
for tag,sn in (("b16","16k"),("c6","6k")):
    print("PHASE",tag)
    D=load(tag); rows=[]
    for d,r in D.items():
        steps=sorted({x["step"] for x in r if "impl" in x})
        for s in steps:
            v={re.sub(r"h11_|_"+sn,"",x["impl"]):x["TF"] for x in r if x.get("step")==s and "impl" in x}
            rows.append((d,s,v))
    arms=list(rows[0][2].keys())
    thr=1.0
    lo=[v for _,_,v in rows if v["o128"]/v["hb"]<thr]; hi=[v for _,_,v in rows if v["o128"]/v["hb"]>=thr]
    print(" allocations low",len(lo),"high",len(hi))
    for a in arms:
        L=[v[a]/v["hb"] for v in lo]; H=[v[a]/v["hb"] for v in hi]
        print(f"  {a:8s} /hb in o128-low {st.median(L) if L else 0:.3f} [{min(L) if L else 0:.3f}-{max(L) if L else 0:.3f}] in o128-high {st.median(H) if H else 0:.3f} [{min(H) if H else 0:.3f}-{max(H) if H else 0:.3f}]")
    print(" per card o128/hb:")
    for d in D: print("  d"+d," ".join(f"{v['o128']/v['hb']:.3f}" for dd,_,v in rows if dd==d))
    if "c8" in arms:
        print(" per card c8/hb:")
        for d in D: print("  d"+d," ".join(f"{v['c8']/v['hb']:.3f}" for dd,_,v in rows if dd==d))
    print(" checks",{x["check"]:(x["fnv"],x["differ_from_hb"]) for x in next(iter(D.values())) if "check" in x})
