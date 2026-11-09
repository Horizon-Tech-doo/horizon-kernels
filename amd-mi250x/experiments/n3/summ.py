import json,sys,statistics as st,collections
T=collections.defaultdict(list); E={}
for fn in sys.argv[1:]:
  for l in open(fn):
    if not l.startswith('{"stack'): continue
    d=json.loads(l); k=(d["stack"],d["B"],d["impl"]); T[k].append(d)
    if "rel_vs_fp64" in d: E[k]=d["rel_vs_fp64"]
for k,v in sorted(T.items()):
    ms=[x["ms"] for x in v]; print(k[0],k[1],f"{k[2]:12s}","ms",st.median(ms),[min(ms),max(ms)],"k1",st.median(x.get("k1",0) for x in v),"GB/s",st.median(x["gbps"] for x in v),"%",st.median(x["pct"] for x in v),"err",E.get(k))
