"""Per-call test: are fast calls transient? For each arm, case, round of a job: min, median, share of calls within 1% of the arm-case job minimum, lag-1 correlation, longest run within 1%, mean position of fast calls. Also a same-process state check: across rounds, spread of round medians vs within-round p10-p90."""
import json,sys,statistics as st,collections as C
for j in sys.argv[1:]:
    d=C.defaultdict(dict)
    for l in open(f'raw/run_{j}.jsonl'):
        if not l.startswith('{'): continue
        try: r=json.loads(l)
        except: continue
        if r.get('kind')=='time' and 'calls' in r: d[(r.get('layer',''),r['T'],r['route'],r['arm'])][r['round']]=r['calls']
    agg=C.defaultdict(list)
    for k,rr in d.items():
        allc=[x for v in rr.values() for x in v]; m0=min(allc); fam='vllm' if k[3].startswith('vllm') else 'ours'
        rm=[st.median(v) for v in rr.values()]
        within=[];lags=[];fast=[]
        for v in rr.values():
            s=sorted(v); within.append((s[int(.9*len(s))]-s[int(.1*len(s))])/s[len(s)//2])
            f=[x<=m0*1.01 for x in v]; fast.append(sum(f)/len(f))
            mu=st.mean(v); den=sum((x-mu)**2 for x in v) or 1
            lags.append(sum((v[i]-mu)*(v[i+1]-mu) for i in range(len(v)-1))/den)
        agg[(fam,'T>=6144' if k[1]>=6144 else 'T<6144')].append(dict(between=(max(rm)-min(rm))/min(rm),within=st.median(within),lag=st.median(lags),fast=st.mean(fast),fastmax=max(fast),fastmin=min(fast),med_over_min=st.median(allc)/m0))
    print('job',j)
    for k,v in sorted(agg.items()):
        print(k,'n',len(v),{q:round(st.median(x[q] for x in v),3) for q in v[0]})
