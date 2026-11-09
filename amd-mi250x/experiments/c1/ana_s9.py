import json,sys,collections as C,statistics as st
R=C.defaultdict(list);CS=C.defaultdict(set);HB=C.defaultdict(list)
for f in sys.argv[1:]:
  for l in open(f):
    if not l.startswith('{') or 'TF' not in l: continue
    r=json.loads(l)
    if 'label' in r:
        n,s=r['label'].rsplit('_',1); R[(s,n)].append(r['TF']); CS[(s)].add(r['cs'])
    elif r.get('impl')=='hipblaslt_direct': HB[r['shape']].append(r['TF'])
for s in ['4k','8k','32k','17k','16k']:
    print('==',s,'hipBLASLt direct',[round(x,1) for x in HB.get(s,[])],'checksums',len(CS[s]))
    for (a,n),v in sorted(((k,v) for k,v in R.items() if k[0]==s),key=lambda kv:-st.median(kv[1])):
        print(f'  {n:22s} med {st.median(v):7.2f}  {[round(x,1) for x in v]}')
