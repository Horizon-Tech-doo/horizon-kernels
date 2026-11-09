# Per shape and form at the cap: ratio to hb per card per round, UMC and delivered clock, FNV against c8.
import json,sys,gzip,glob,re,statistics as st
O=sys.argv[1]
m={int(l.split()[1]):l.split()[5] for l in open(f"{O}/map.txt") if l.startswith("rocr")}
ms=[json.loads(l) for l in gzip.open(f"{O}/metrics.jsonl.gz","rt") if l.strip().startswith("{") and l.strip().endswith("}")]
for sn in sys.argv[2].split():
    res={}; fnv={}
    for f in sorted(glob.glob(f"{O}/all_d*_{sn}.jsonl")):
        d=int(re.search(r"_d(\d+)_",f).group(1)); r=[json.loads(l) for l in open(f) if l.startswith("{")]
        a=[x for x in r if "impl" in x]; hb={x["round"]:x["TF"] for x in a if x["impl"]=="hb"}
        for x in a:
            s=[q[m[d]] for q in ms if x["t0"]+0.5<=q["t"]<=x["t1"] and m[d] in q]
            e=res.setdefault(x["impl"],dict(tf=[],ratio=[],umc=[],agfx=[],pw=[]))
            e["tf"].append(x["TF"]); e["ratio"].append(x["TF"]/hb[x["round"]]); e["umc"]+= [z["umc"] for z in s]; e["agfx"]+=[z["agfx"] for z in s]
        for x in r:
            if "check" in x: fnv.setdefault(x["check"],set()).add(x["fnv"])
    print("==",sn)
    for k,e in sorted(res.items(),key=lambda kv:-st.median(kv[1]["ratio"])):
        print(f"{k:20s} TF {st.median(e['tf']):7.2f} ratio med {st.median(e['ratio']):.3f} [{min(e['ratio']):.3f}-{max(e['ratio']):.3f}] n{len(e['ratio'])} umc {st.mean(e['umc']) if e['umc'] else 0:.1f} agfx {st.median(e['agfx']) if e['agfx'] else 0:.0f} fnv {sorted(fnv.get(k,[]))}")
