"""sum.py run.jsonl [filter]: median TF per (stack, patch, name, input) with range; ratios to CK of the same job and input."""
import json,sys,statistics as st,collections as c
d=c.defaultdict(list); ex={}
for l in open(sys.argv[1]):
    r=json.loads(l)
    if "info" in r: continue
    k=(r.get("case",""),r.get("input",""),r["name"].replace(r.get("case","")+"_",""),r.get("stack","")[:1],r.get("patch",""))
    if r.get("err"): ex[k]=r["err"][-60:]; continue
    d[k].append(r["tflops"])
    for f in ("rel_vs_fp64","rel_vs_sdpa","rel_vs_sdpa32","repeat_equal"):
        if f in r: ex.setdefault(k+(f,),r[f])
ck={}
for k,v in d.items():
    if k[2]=="ck": ck[(k[0],k[1])]=st.median(v)
filt=sys.argv[2] if len(sys.argv)>2 else ""
rows=sorted(d.items(),key=lambda kv:(kv[0][0],kv[0][1],-st.median(kv[1])))
for k,v in rows:
    s=" ".join(k)
    if filt and filt not in s: continue
    m=st.median(v); q=ck.get((k[0],k[1]))
    print(f"{s:62s} {m:7.2f} [{min(v):.2f},{max(v):.2f}] {'%.3f'%(m/q) if q else '     '} e={ex.get(k+('rel_vs_fp64',),'')} s={ex.get(k+('rel_vs_sdpa',),ex.get(k+('rel_vs_sdpa32',),''))} rep={ex.get(k+('repeat_equal',),'')}")
for k,e in ex.items():
    if len(k)==5 and k not in d and (not filt or filt in " ".join(k)): print("ERR"," ".join(k),e)
