import json,sys,statistics as st,collections as C
R=C.defaultdict(list);M={}
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    r=json.loads(l)
    if 'error' in r: print('ERR',r); continue
    k=(r['shape'],tuple(r['arm']) if isinstance(r['arm'],list) else r['arm']); R[k].append(r['TF']); M[k]=r
for sh in ['4k','8k','32k','17k','16k']:
    print('==',sh,'(TF per round; arm = bM,bN,bK,threads,blocks)')
    rows=[(st.mean(v),k,v) for k,v in R.items() if k[0]==sh]
    for m,k,v in sorted(rows,key=lambda t:-t[0]):
        r=M[k]; print(f"  {str(k[1]):28s} {r.get('bounds','')!s:28s} V{r.get('V','')} A{r.get('A','')} scr{r.get('scr','')} occ{r.get('occ','')} {[round(x,1) for x in v]} mean {m:.2f}")
