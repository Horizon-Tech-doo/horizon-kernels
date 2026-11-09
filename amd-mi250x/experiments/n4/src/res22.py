# Per shape: median TF, ratio to hb and to c8 by same-round pairs, FNV against c8.
import json,sys,statistics as st
O=sys.argv[1]; names=sys.argv[2].split() if len(sys.argv)>2 else ["6k","12k","16k","32k"]
for sn in names:
    rows=[json.loads(l) for l in open(f"{O}/hb_{sn}.jsonl") if l.startswith("{")]
    tf={}; 
    for r in rows:
        if "impl" in r: tf.setdefault(r["impl"],{})[r["round"]]=r["TF"]
    ck={r["check"]:r for r in rows if "check" in r}
    c8=[k for k in tf if k.startswith("h10_c8")][0]; fb=ck.get(c8,{}).get("fnv")
    print("==",sn,"hb",st.median(tf["hb"].values()), "rounds",len(tf["hb"]))
    for k,v in sorted(tf.items(),key=lambda kv:-st.median(kv[1].values())):
        if k=="hb": 
            continue
        pr=[v[r]/tf[c8][r] for r in v if r in tf[c8]]; ph=[v[r]/tf["hb"][r] for r in v if r in tf["hb"]]
        c=ck.get(k,{})
        print(f"{k:22s} TF {st.median(v.values()):7.2f} /hb {st.median(ph):.3f} [{min(ph):.3f}-{max(ph):.3f}] /c8 {st.median(pr):.3f} rel {c.get('rel')} fnvsame {c.get('fnv')==fb}")
