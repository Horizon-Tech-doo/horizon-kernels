import json,glob,statistics as st,collections as C,re,sys
rows=[];chk=[];direct=[];hdr=[]
src={}
for f in sorted(glob.glob('logs/ep-*.out')):
    job=re.search(r'(\d{8})',f).group(1)
    for l in open(f,errors='replace'):
        if not l.startswith('{'):continue
        try:o=json.loads(l)
        except: continue
        o['job']=job
        if 'header' in o: hdr.append(o)
        if o.get('arm')=='direct': direct.append(o)
        elif 'arm' in o and 'round' in o: rows.append(o)
        elif 'check' in o: chk.append(o)
print('rows',len(rows),'checks',len(chk),'direct',len(direct),'hdr',len(hdr))
print('rows by job',C.Counter(o['job'] for o in rows)); print('chk by job',C.Counter(o['job'] for o in chk))
print('md5s',C.Counter(h['moe_md5'] for h in hdr))
print('dropped',max(o['dropped_max'] for o in rows))
print('worst',max(c['rel_y_vs_fp32'] for c in chk),max(c['rel_out_vs_fp32'] for c in chk),sum(c['round_trip_diff'] for c in chk),sum(c['nonfinite'] for c in chk))
print('chk arm',C.Counter(c['check'] for c in chk))
print('direct',[(d['job'],d['tokens_per_rank'],d['compute_ms'],d['rel_vs_fp32']) for d in direct])
g=C.defaultdict(list)
for o in rows:g[o['job'],o['tokens_per_rank'],o['arm'],o['world']].append(o)
print('rounds per cell',C.Counter(len(v) for v in g.values()), [k for k,v in g.items() if len(v)!=4])
m=lambda k,x:st.median(r[x] for r in g[k])
J='23516765'
for T in(2048,512):
  for a in('old','land','skip'):
    b=m((J,T,a,1),'tokens_per_s_total')
    for W in(1,8,16,32):
      k=(J,T,a,W);d,c,cb,s,tp=[m(k,x) for x in('dispatch_ms','compute_ms','combine_ms','step_ms','tokens_per_s_total')]
      print(T,a,W,f'{d:.3f} {c:.3f} {cb:.3f} {s:.3f} share {(d+cb)/(d+c+cb):.1%} {tp/1e6:.3f} eff {tp/W/b:.1%} rng {min(r["tokens_per_s_total"] for r in g[k])/1e6:.3f}-{max(r["tokens_per_s_total"] for r in g[k])/1e6:.3f}')
print('--same job vs pooled')
pool={}
for T in(2048,512):
  for a in('old','land','skip'):
    vals=[st.median(r['tokens_per_s_total'] for r in g[k]) for k in g if k[1]==T and k[2]==a and k[3]==1]
    pool[T,a]=st.median(vals);print('W1',T,a,[(k[0],round(st.median(r['tokens_per_s_total'] for r in g[k])/1e6,4)) for k in g if k[1]==T and k[2]==a and k[3]==1],'pool',pool[T,a])
print('--cells by job, spread')
worst=[]
for T in(2048,512):
  for a in('old','land','skip'):
    for W in(1,8,16,32):
      ks=[k for k in g if k[1]==T and k[2]==a and k[3]==W]
      v=[st.median(r['tokens_per_s_total'] for r in g[k]) for k in ks]
      if v:
        sp=max(v)/min(v)-1; worst.append((sp,T,a,W))
        js=[(k[0][-4:],round(x/1e6,3),f'{x/W/pool[T,a]:.1%}', (f'{x/W/st.median(r["tokens_per_s_total"] for r in g[k[0],T,a,1]):.1%}' if (k[0],T,a,1) in g else '-')) for k,x in zip(ks,v)]
        print(T,a,W,f'{sp:.1%}',js)
print(sorted(worst)[-5:])
print('--ratio old/land by step')
for T in(2048,512):
  for W in(1,8,16,32):
    out=[]
    for j in sorted({k[0] for k in g}):
      if (j,T,'old',W) in g and (j,T,'land',W) in g: out.append((j[-4:],round(m((j,T,'old',W),'step_ms')/m((j,T,'land',W),'step_ms'),3)))
    print(T,W,out)
print('--skip/land')
for W in(1,8,16,32): print(W,round(m((J,2048,'land',W),'step_ms')/m((J,2048,'skip',W),'step_ms'),3))
print('direct-based')
for T in(2048,512):
  dm=st.median(d['compute_ms'] for d in direct if d['tokens_per_rank']==T); base=T/dm*1e3
  print(T,dm,'land c/direct',m((J,T,'land',1),'compute_ms')/dm,'land step/direct',m((J,T,'land',1),'step_ms')/dm,[round(m((J,T,'land',W),'tokens_per_s_total')/W/base,3) for W in(8,16,32)],'8->32',m((J,T,'land',32),'tokens_per_s_total')/m((J,T,'land',8),'tokens_per_s_total'))
