import json,sys,statistics as st,collections
rows=[]
for fn in sys.argv[1:]:
    job=fn.split('-')[-1].split('.')[0]
    for l in open(fn):
        if l.startswith('{'):
            try: d=json.loads(l)
            except: continue
            d['job']=job; rows.append(d)
T=collections.defaultdict(list); P=collections.defaultdict(list)
for d in rows:
    if d.get('t')=='time':
        k=(d['job'],d['stack'],d['B'],d['L'],d['mode'],d['impl']); T[k].append(d['ms']); P[k].append(d['pct'])
keys=sorted({k[:5] for k in T},key=lambda x:(x[0],x[1],x[3]*x[2],x[2],x[3],x[4]))
for kk in keys:
    arms={k[5]:(st.median(T[k]),max(T[k])/min(T[k])-1,st.median(P[k])) for k in T if k[:5]==kk}
    vp=arms.get('vllm_p',(None,))[0]; vs=arms.get('vllm_s',(None,))[0]
    print(*kk, ' '.join(f"{n}={m:.3f}/{p:.0f}%"+(f"/x{(vs if n in ('v2f','v2n','v3') else vp)/m:.2f}" if (vs if n in ('v2f','v2n','v3') else vp) else "") for n,(m,sp,p) in sorted(arms.items(),key=lambda x:x[1][0])))
print("== err max per impl,mode")
E=collections.defaultdict(list)
for d in rows:
    if d.get('t')=='err': E[(d['stack'],d['mode'],d['impl'])].append((d['rel'],d.get('nan')))
Em=collections.defaultdict(list)
for k,v in E.items():
    Em[(k[0],k[1])].append((k[2],max(x[0] for x in v) if all(x[0]==x[0] for x in v) else 'NaN',max((x[1] or 0) for x in v)))
for k,v in sorted(Em.items()): print(*k,' '.join(f"{a}:{(f'{b:.3g}' if b!='NaN' else b)}"+(f"[n{c}]" if c else "") for a,b,c in v))
for d in rows:
    if d.get('t')=='compile_err': print(d)
