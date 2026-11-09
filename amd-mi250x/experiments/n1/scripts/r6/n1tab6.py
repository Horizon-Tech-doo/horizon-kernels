import json,sys,statistics as st,collections as C
def load(j,pre=''):
    out=[]
    for l in open(f'raw/run_{j}.jsonl'):
        if l.startswith('{'):
            try: out.append(json.loads(l))
            except: pass
    return out
def med(x): return st.median(x)
def tab(rs,arms=None,stage=None,cond=lambda r:True,ref='vllm_total'):
    d=C.defaultdict(lambda:C.defaultdict(list))
    for r in rs:
        if r.get('kind')=='time' and cond(r):
            d[(r.get('layer',''),r['T'],r['route'])][r['arm']].append(r)
    return d
for j in sys.argv[1:]:
    rs=load(j)
    print('== job',j,'nonfinite',sum(1 for r in rs if r.get('kind')=='err' and r.get('nonfinite')),'worst err',max([r['rel_vs_fp32'] for r in rs if r.get('kind')=='err'] or [0]), 'fails',[r['kind'] for r in rs if 'fail' in r.get('kind','')][:5])
    d=tab(rs)
    sp=[]
    for k in d:
        aa=d[k]; base=aa.get('vllm_total')
        line=[]
        for a,v in aa.items():
            m=[x['tot_med'] for x in v]; sp.append((max(m)-min(m))/min(m))
        if not base: continue
        vm=med([x['tot_med'] for x in base]); vmin=min(x['tot_min'] for x in base)
        for a,v in aa.items():
            if a.startswith('vllm'): continue
            mm=med([x['tot_med'] for x in v]); mi=min(x['tot_min'] for x in v)
            s=[med([x['stages'][i] for x in v]) for i in range(4)]
            line.append(f"{a} {mm/vm:.3f}/{mi/vmin:.3f} st{[round(q,3) for q in s]}")
        print(k,f"v={vm:.3f}",' | '.join(line))
    sp.sort(); print('spread p50 %.3f p90 %.3f max %.3f n=%d'%(sp[len(sp)//2],sp[int(len(sp)*.9)],sp[-1],len(sp)))
