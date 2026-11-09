# N4 round 3, job 35. Ratio ours over hb, same step, throughput. argv: dir
import json,sys,glob,re,statistics as st,collections as cc
O=sys.argv[1]
for f in sorted(glob.glob(f"{O}/*.jsonl")):
    r=[json.loads(l) for l in open(f) if l.startswith("{")]
    off={x["step"]:x["boff"] for x in r if "boff" in x}; hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
    print("==",f.split("/")[-1],"fnvs",{x["fnv"][:8] for x in r if "check" in x})
    T=cc.defaultdict(lambda:cc.defaultdict(list)); H=cc.defaultdict(list); Tf=cc.defaultdict(lambda:cc.defaultdict(list))
    for x in r:
        if "impl" in x:
            H[off[x["step"]]].append(hb[x["step"]])
            if x["impl"]!="hb":
                n=re.sub(r"h11_|_(16k|6k|8k)$","",x["impl"]); T[n][off[x["step"]]].append(round(x["TF"]/hb[x["step"]],3)); Tf[n][off[x["step"]]].append(x["TF"])
    if len(H)>3:
        for d in sorted(H): print(f"  {d:8d} hb {st.median(H[d]):7.2f} "+" ".join(f"{n} {' '.join('%.3f'%v for v in T[n][d])}" for n in T))
    else:
        for n in T: print(f"  {n:14s} "+"  ".join(f"off {d}: {st.median(T[n][d]):.3f} ({min(T[n][d]):.3f}-{max(T[n][d]):.3f}) TF {st.median(Tf[n][d]):.2f}" for d in sorted(T[n])))
