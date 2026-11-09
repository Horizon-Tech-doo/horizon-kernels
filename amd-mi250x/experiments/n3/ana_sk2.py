import json,statistics as st,collections as C,sys
J=('23504738','23504743')
T=C.defaultdict(list);Pc=C.defaultdict(list);E=C.defaultdict(list);S=C.defaultdict(list)
for j in J:
    for l in open(f'raw/sk2-{j}.jsonl'):
        if not l.startswith('{'): print(j,l.strip()[:200]) if 'gfx' in l or 'rror' in l else None; continue
        d=json.loads(l)
        if d['t']=='time': T[(j,d['B'],d['L'],d['mode'],d['impl'])].append(d['ms']);Pc[(j,d['B'],d['L'],d['mode'],d['impl'])].append(d['pct'])
        elif d['t']=='err': E[d['impl']].append(d['rel'])
        elif d['t']=='seed': S[(j,d['B'],d['L'],d['seed'],d['mode'],d['impl'])].append(d['ms'])
        elif d['t']=='meta' and j==J[0]: print(d)
print('err max',{k:f'{max(v):.3g}' for k,v in E.items()})
sh=[(8,32768),(16,16384),(32,8192),(64,4096),(32,1024),(64,1024)]
for B,L in sh:
  for m in ('base','full','part','skew'):
    o=[]
    for j in J:
        g=lambda a:st.median(T[(j,B,L,m,a)]) if T.get((j,B,L,m,a)) else None
        la,ls,vp,vr,vt,vs=[g(a) for a in('landed','landed_s','vllm_p','vllm_rocm','vllm_tri','vllm_s')]
        sp=max(T[(j,B,L,m,'vllm_rocm')])/min(T[(j,B,L,m,'vllm_rocm')])-1
        s=f"la={la:.3f} {st.median(Pc[(j,B,L,m,'landed')]):.0f}% uni={vp:.3f} roc={vr:.3f}({st.median(Pc[(j,B,L,m,'vllm_rocm')]):.0f}%,sp{sp*100:.1f}) tri={vt:.3f} roc/la=x{vr/la:.2f} uni/la=x{vp/la:.2f} tri/la=x{vt/la:.2f}"
        if ls: s+=f" ls={ls:.3f} roc/ls=x{vr/ls:.2f}"
        o.append(s)
    print(f"{B}x{L} {m}: "+"\n     ".join(o))
print('== seeds, landed form, ratios per length draw, vendor/landed')
for B,L in [(64,1024),(32,1024),(64,2048)]:
  for m in ('part','skew'):
    for v in ('vllm_p','vllm_rocm','vllm_tri'):
      for j in J:
        r=[st.median(S[(j,B,L,s,m,v)])/st.median(S[(j,B,L,s,m,'landed')]) for s in range(8)]
        print(B,L,m,v,j,'min %.2f med %.2f max %.2f'%(min(r),st.median(r),max(r)),[round(x,2) for x in r])
