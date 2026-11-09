import json,sys,glob,statistics as st,collections as C
d=sys.argv[1]
R=C.defaultdict(list)
for l in open(d+'/py.jsonl'):
    if not l.startswith('{'): continue
    r=json.loads(l); R[(r['shape'],r['arm'])].append(r['TF'])
    if r.get('max_rel_err') is not None: R[('err',r['arm'])].append(r['max_rel_err'])
for l in open(d+'/gmain.jsonl'):
    if not l.startswith('{'): continue
    r=json.loads(l); s=r['label'].rsplit('_',1); R[(s[1],s[0])].append(r['TF']); R[('cs',r['label'])].append(r['cs'])
for l in open(d+'/hbm2.jsonl'):
    if not l.startswith('{') or 'TF' not in l: continue
    r=json.loads(l); sh=r.get('shape'); R[(sh,'hbm2_'+str(r.get('impl')))].append(r['TF'])
for sh in ['4k','8k','32k','17k','16k']:
    print('==',sh)
    for (s,a),v in sorted(R.items(),key=lambda kv:-st.mean(kv[1]) if kv[0][0]==sh else 0):
        if s==sh: print(f'  {a:28s} {[round(x,2) for x in v]}  mean {st.mean(v):.2f}')
print({k[1]:(len(set(v)),v[0][:8]) for k,v in R.items() if k[0]=='cs'})
print({k[1]:max(v) for k,v in R.items() if k[0]=='err'})
