import json,statistics as st,collections as C,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
h=C.defaultdict(lambda:[0,0,0])
for r in rows:
    if r['t']=='haz':
        x=h[(r['mode'],r['blocks'])]; x[0]+=r['dwords_checked'];x[1]+=r['errors'];x[2]+=r['sentinel']
for k in sorted(h): print(k,h[k])
print("ords"); h=C.defaultdict(lambda:[0,0,0])
for r in rows:
    if r['t']=='ords':
        x=h[(r['k'],r['blocks'])]; x[0]+=r['lane_checks'];x[1]+=r['errors'];x[2]+=r['sentinel']
for k in sorted(h): print(k,h[k])
for r in rows:
    if r['t']=='clk1': print(r)
print("gemm")
g=C.defaultdict(list); cs=C.defaultdict(set)
for r in rows:
    if r['t']=='g': g[(r['M'],r['k'],r['dyn'])].append(r['TF']); cs[r['M']].add(r['cs'])
print({k:v for k,v in cs.items()})
for k in sorted(g): v=g[k]; print(k,v,'mean %.2f spread %.1f%%'%(st.mean(v),(max(v)-min(v))/st.mean(v)*100))
for r in rows:
    if r['t']=='hipblaslt': print(r)
print("parts")
p=C.defaultdict(list)
for r in rows:
    if r['t']=='p': p[(r['M'],r['hot'],r['k'],r['dyn'])].append(r)
for k in sorted(p):
    v=p[k]; ms=[x['ms'] for x in v]; print(k,'ms',[round(m,3) for m in ms],'GB/s %.0f TFm %.1f'%(st.mean([x['GBps_if_load'] for x in v]),st.mean([x['TF_if_math'] for x in v])))
