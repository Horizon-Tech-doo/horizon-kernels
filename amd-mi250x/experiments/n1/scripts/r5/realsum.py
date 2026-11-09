import json,sys,collections
R=collections.defaultdict(list);ER={}
for l in open(sys.argv[1]):
    if not l.startswith('{'):continue
    try:d=json.loads(l)
    except:continue
    if d.get('kind')=='time':R[(d['layer'],d['T'],d['stack'],d['arm'])].append(d)
    if d.get('kind')=='err':ER[(d['layer'],d['T'],d['stack'],d['arm'])]=(d['rel_vs_fp32'],d['nonfinite'])
def m(k,f='tot_med'):
    v=R.get(k);return sum(x[f] for x in v)/len(v) if v else None
for L in (0,19,39):
  for T in sorted({k[1] for k in R if k[0]==L},reverse=True):
    v=m((L,T,'vllm','vllm_total'))
    ks=lambda s,a:(sum(x['stages'][1] for x in R[(L,T,s,a)])/len(R[(L,T,s,a)])) if (L,T,s,a) in R else None
    vk=ks('vllm','vllm_stages')
    out=[f"L{L} T{T} v{v:.3f} vk1 {vk:.3f}"]
    for s,a in (('vllm','ship'),('vllm','LS_ilp'),('lumi','ship'),('lumi','LS_u20k')):
        t=m((L,T,s,a))
        if t: out.append(f"{s[:2]}/{a} {t:.3f} r={t/v:.3f} k1 {ks(s,a):.3f} [{min(x['tot_med'] for x in R[(L,T,s,a)])/v:.3f}-{max(x['tot_med'] for x in R[(L,T,s,a)])/v:.3f}]")
    print(' | '.join(out))
print('worst err ours',max((e[0],k) for k,e in ER.items() if 'vllm_' not in k[3]),'vllm',max((e[0],k) for k,e in ER.items() if 'vllm_' in k[3]),'nf',sum(e[1] for e in ER.values()))
