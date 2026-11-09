import json,sys,collections
for f in sys.argv[1:]:
    R=collections.defaultdict(list)
    for l in open(f):
        if not l.startswith('{'):continue
        try:d=json.loads(l)
        except:continue
        if d.get('kind')=='time' and 'stages' in d:R[(d['T'],d['route'],d['stack'],d['arm'])].append(d)
    print(f)
    for (T,r,s,a),v in sorted(R.items()):
        if a=='vllm_stages' or s=='vllm' and a!='ship' and a not in('LS_ilp','LS_none','B6S_ilp','B6SS_ilp','LDS_ilp'): continue
        if s=='lumi' and a not in('LS_u20k','LS_none','B6S_u20k','B6SS_u20k','LDS_u20k'): continue
        if a=='ship':continue
        sh=R.get((T,r,s,'ship'))
        if not sh:continue
        k=lambda vv,i:sum(x['stages'][i] for x in vv)/len(vv)
        print(f"T{T} {r} {s} {a} k1 {k(v,1):.3f} vs ship {k(sh,1):.3f} = {100*(k(v,1)/k(sh,1)-1):+.1f}%  k2 {k(v,2):.3f}/{k(sh,2):.3f}")
