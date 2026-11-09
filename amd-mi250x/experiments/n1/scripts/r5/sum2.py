import json,sys,collections
R=collections.defaultdict(list);ER={}
for l in open(sys.argv[1]):
    if not l.startswith('{'):continue
    try:d=json.loads(l)
    except:continue
    k=(d.get('layer','syn'),d.get('T'),d.get('route'),d.get('stack'),d.get('arm'))
    if d.get('kind')=='time':R[k].append(d)
    if d.get('kind')=='err':ER[k]=(d['rel_vs_fp32'],d['nonfinite'])
    if 'failed' in str(d.get('kind')):print('FAILED',d.get('T'),d.get('arm'),str(d.get('err'))[-200:])
def m(k,f='tot_med'):
    v=R.get(k);return sum(x[f] for x in v)/len(v) if v else None
for k in sorted({k[:4] for k in R},key=str):
    if k[3] != 'vllm' and (k[0],k[1],k[2],'vllm') in {x[:4] for x in R}: pass
for L,T,r in sorted({k[:3] for k in R},key=lambda x:(str(x[0]),x[1],x[2])):
    v=m((L,T,r,'vllm','vllm_total'))
    out=[f"{L} T{T} {r} v {v:.3f}"]
    for s in ('vllm','lumi'):
        for a in ('ship','grid'):
            t=m((L,T,r,s,a))
            if t: 
              st=R[(L,T,r,s,a)][0].get('stages')
              n=len(R[(L,T,r,s,a)]); st=[sum(x['stages'][i] for x in R[(L,T,r,s,a)])/n for i in range(4)]
              out.append(f"{s[:2]}/{a} {t:.3f} r={t/v:.3f} [{min(x['tot_med'] for x in R[(L,T,r,s,a)])/v:.3f}-{max(x['tot_med'] for x in R[(L,T,r,s,a)])/v:.3f}] n={n} st={' '.join(f'{x:.3f}' for x in st)}")
    print(' | '.join(out))
o=[(e[0],k) for k,e in ER.items() if 'vllm_' not in k[4]]; w=[(e[0],k) for k,e in ER.items() if 'vllm_' in k[4]]
print('checks',len(o),len(w),'worst ours',max(o),'worst vllm',max(w),'nonfinite',sum(e[1] for e in ER.values()))
