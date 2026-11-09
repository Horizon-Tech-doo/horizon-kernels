import json,glob,re,statistics as st,collections as cc
for f in sorted(glob.glob("raw/j37_23508861/*s_*.jsonl")+glob.glob("raw/j37_23508861/comp_*.jsonl")):
    r=[json.loads(l) for l in open(f) if l.startswith("{")]
    off={x["step"]:(x["q"],x["boff"]) for x in r if "boff" in x}; hb={x["step"]:x["TF"] for x in r if x.get("impl")=="hb"}
    T=cc.defaultdict(lambda:cc.defaultdict(list))
    for x in r:
        if "impl" in x and x["impl"]!="hb" and off[x["step"]][0]=="b": T[re.sub(r"h11_|_(16k|6k|8k)$","",x["impl"])][off[x["step"]][1]].append((x["TF"]/hb[x["step"]],x["TF"]))
    print("==",f.split("/")[-1],"fnv",{x["fnv"][:8] for x in r if "check" in x}, "hb",round(st.median(hb.values()),2))
    for n,v in T.items(): print(f"  {n:14s}"+"  ".join(f"off {d}: {st.median(z[0] for z in v[d]):.3f} ({min(z[0] for z in v[d]):.3f}-{max(z[0] for z in v[d]):.3f}) TF {st.median(z[1] for z in v[d]):.2f}" for d in sorted(v)))
