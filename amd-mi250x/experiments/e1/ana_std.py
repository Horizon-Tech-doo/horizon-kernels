import json,collections,statistics as st,sys
SH={(4096,4096,4096):'4k',(8192,8192,8192):'8k',(32768,1024,2048):'32k',(4096,17408,5120):'17k'}
D=collections.defaultdict(list); CS=collections.defaultdict(set)
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    r=json.loads(l); D[r['label']].append(r['TF']); CS[(r['M'],r['N'],r['K'])].add((r['cs'],r['label']))
H=collections.defaultdict(list);W=collections.defaultdict(list)
for fn in sys.argv[2:]:
  for l in open(fn):
    if l.startswith('{"impl": "hipblaslt"'): r=json.loads(l); H[(r['M'],r['N'],r['K'])].append(r['TF'])
    if l.startswith('{"impl": "tl_wrapper_b12864",'): r=json.loads(l); W[(r['M'],r['N'],r['K'])].append(r['TF'])
def f(v): return f"{st.median(v):6.2f} [{min(v):.2f},{max(v):.2f}] n={len(v)}"
for s,sn in SH.items():
    h=st.median(H[s]); print(f"== {sn} {s} hipBLASLt {f(H[s])} wrapper b12864 {f(W[s])} ratio wr/hb {st.median(W[s])/h:.3f}")
    for v in ['b12864','b25632w2','rr12864w3','rr25632w2','b25632','rr12864','rr25632','b12864m32','rr12864m32w3','rr12864m32']:
        for tu in ['solo','comb','mbtu']:
            k=f"{v}_{sn}|{tu}"
            if k in D: print(f"  {v:14s} {tu} {f(D[k])}  /hb {st.median(D[k])/h:.3f}")
    cs=collections.defaultdict(list)
    for c,lab in CS[s]: cs[c].append(lab)
    print("  checksums:",{c:len(v) for c,v in cs.items()}, [v for c,v in cs.items() if any('m32' not in x for x in v)][0][:0])
    for c,v in cs.items():
        nm=[x for x in v if 'm32' not in x]
        if nm and len(cs)>1: print('   non-m32 group',c,len(nm),'of',sum(1 for x in sum(cs.values(),[]) if 'm32' not in x))
