import json,sys,gzip,statistics as st
O=sys.argv[1]
ms=[json.loads(l) for l in gzip.open(f"{O}/metrics.jsonl.gz","rt") if l.strip().startswith("{") and l.strip().endswith("}")]
print({k:0 for k in []})
for sn in "4k 8k 16k 17k 32k 6k 12k 4kk".split():
    rows=[json.loads(l) for l in open(f"{O}/hb_{sn}.jsonl") if l.startswith("{")]
    a=[r for r in rows if "impl" in r]; hb={r["round"]:r["TF"] for r in a if r["impl"]=="hb"}
    o=[]
    for nm in sorted(set(r["impl"] for r in a)):
        x=[r for r in a if r["impl"]==nm]; rr=[r["TF"]/hb[r["round"]] for r in x]
        cs=[k for k in ms[0] if k.startswith("card")]
        act={c:st.mean(q[c].get("act",0) for q in ms if x[0]["t0"]<=q["t"]<=x[-1]["t1"]) for c in cs} if cs else {}
        c=max(act,key=act.get) if act else None
        s=[q[c] for r in x for q in ms if r["t0"]+.5<=q["t"]<=r["t1"]]
        o.append(f"{nm.split('_')[1] if nm!='hb' else 'hb'}: TF {st.median(r['TF'] for r in x):.2f} /hb {st.median(rr):.3f} [{min(rr):.3f}-{max(rr):.3f}] umc {st.mean(z['umc'] for z in s):.0f} agfx {st.median(z['agfx'] for z in s):.0f} W {st.median(max(z.get('pw',0),0) for z in s):.0f}")
    ck={r['check']:r['fnv'] for r in rows if 'check' in r}
    print(sn," | ".join(o),"fnv_equal",len(set(ck.values()))==1)
