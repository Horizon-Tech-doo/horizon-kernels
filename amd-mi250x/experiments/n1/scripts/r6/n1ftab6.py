import json,sys,statistics as st,collections as C
for j in sys.argv[1:]:
    rs=[]
    for l in open(f'raw/fp_{j}.jsonl'):
        if l.startswith('{'):
            try: rs.append(json.loads(l))
            except: pass
    print('job',j,'failed',[(r.get('T'),r.get('arm')) for r in rs if 'fail' in r.get('kind','')][:6])
    d=C.defaultdict(lambda:C.defaultdict(list)); e={}
    for r in rs:
        if r['kind']=='time': d[(r['layer'],r['T'],r['route'])][r['arm']].append(r)
        if r['kind']=='err' : e[(r['layer'],r['T'],r['route'],r['arm'])]=r
    for k,a in d.items():
        b=st.median(x['tot_med'] for x in a['fp16'])
        line=f"{k} fp16={b:.3f}"
        for f in ('fp8','fp4'):
            if f in a:
                m=st.median(x['tot_med'] for x in a[f]); s=[st.median(x['stages'][i] for x in a[f]) for i in range(4)]
                er=e[k+(f,)]
                line+=f" | {f} {m/b:.3f} k1k2 {s[1]:.2f},{s[2]:.2f} rel={er['rel_vs_fp16w']:.4f} kd={er['kernel_vs_dq']:.1e} cos={er['cos']:.5f} nf={er['nonfinite']}"
        print(line)
