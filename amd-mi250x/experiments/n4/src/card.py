import json,glob,re,sys
for O,sn in (("raw/j24_23506098","16k"),("raw/j26_23506132","16k"),("raw/j24_23506098","6k"),("raw/j26_23506132","6k"),("raw/j26_23506132","8k")):
    print(O,sn)
    for f in sorted(glob.glob(f"{O}/all_d*_{sn}.jsonl")):
        d=re.search(r"_d(\d)_",f).group(1); a=[json.loads(l) for l in open(f) if l.startswith("{")]; a=[x for x in a if "impl" in x]
        hb={x["round"]:x["TF"] for x in a if x["impl"]=="hb"}
        o=[]
        for x in a:
            n=x["impl"]
            if "o128c8" in n: o.append("o128 r%d %.3f"%(x["round"],x["TF"]/hb[x["round"]]))
            elif "_c8_" in n or "25664c8" in n: o.append("c8 r%d %.3f"%(x["round"],x["TF"]/hb[x["round"]]))
        print(" d"+d," ".join(o)," hb r0 %.2f r1 %.2f"%(hb[0],hb[1]))
