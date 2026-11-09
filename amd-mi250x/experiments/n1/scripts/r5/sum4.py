import json,sys,collections
R=collections.defaultdict(list);ER={}
for l in open(sys.argv[1]):
    if not l.startswith('{'):continue
    try:d=json.loads(l)
    except:continue
    k=(d.get('layer','syn'),d.get('grid',0),d.get('T'),d.get('route'),d.get('arm'))
    if d.get('kind')=='time':R[k].append(d)
    if d.get('kind')=='err':ER[k]=(d['rel_vs_fp32'],d['nonfinite'])
    if 'failed' in str(d.get('kind')):print('FAILED',d.get('T'),d.get('arm'),str(d.get('err'))[-200:])
A=lambda v,f:sum(x[f] for x in v)/len(v)
for K in sorted({k[:4] for k in R},key=str):
    v=R[K+('vllm_total',)]; vm=A(v,'tot_med'); vs=R[K+('vllm_stages',)]
    out=[f"{K[0]} g{K[1]} T{K[2]} {K[3]} v {vm:.3f} k1 {sum(x['stages'][1] for x in vs)/len(vs):.3f}"]
    for a in ('ship','M_none','grid'):
        w=R.get(K+(a,))
        if w:
            t=A(w,'tot_med');
            out.append(f"{a} {t:.3f} r={t/vm:.3f} [{min(x['tot_med'] for x in w)/vm:.3f}-{max(x['tot_med'] for x in w)/vm:.3f}] min/vmin {min(x['tot_min'] for x in w)/min(x['tot_min'] for x in v):.3f} k1 {sum(x['stages'][1] for x in w)/len(w):.3f} s {sum(x['stages'][3] for x in w)/len(w):.3f}")
    print(' | '.join(out))
o=[(e[0],k) for k,e in ER.items() if 'vllm_' not in k[4]]; w=[(e[0],k) for k,e in ER.items() if 'vllm_' in k[4]]
print('checks',len(o),len(w),'worst ours',max(o),'worst vllm',max(w),'nonfinite',sum(e[1] for e in ER.values()))
