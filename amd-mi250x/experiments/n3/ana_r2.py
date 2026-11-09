import json,sys,statistics as st,collections
rows=[]
for fn in sys.argv[1:]:
    job=fn.split('-')[-1].split('.')[0]
    for l in open(fn):
        if l.startswith('{'):
            try: d=json.loads(l)
            except: continue
            d['job']=job; rows.append(d)
def agg(tag,keys):
    T=collections.defaultdict(list)
    for d in rows:
        if d.get('t')==tag: T[tuple(d.get(k) for k in keys)].append(d['ms'])
    return T
print("== kern");[print(d) for d in rows if d.get('t')=='kern']
print("== err")
E=collections.defaultdict(dict)
for d in rows:
    if d.get('t')=='err': E[(d['job'],d['stack'],d['mode'])][d['impl']]=(f"{d['rel']:.3g}" if 'rel' in d else None,d.get('nan'))
for k,v in E.items(): print(k,v)
print("== tail");
for k,v in sorted(agg('tail',['job','stack','mode','impl']).items()): print(*k,'med',st.median(v),'range',min(v),max(v))
print("== realloc")
T=collections.defaultdict(dict)
for d in rows:
    if d.get('t')in('realloc','reload'):
        k=(d['job'],d['stack'],d['t'],d['iter']);T[k].setdefault(d['impl'],[]).append(d['ms']);T[k]['delta']=d.get('delta');T[k]['maps']=d.get('maps')
for k,v in T.items(): print(*k,{a:(b if isinstance(b,(int,type(None))) else [round(x,3) for x in b]) for a,b in v.items()})
print("== place")
T=collections.defaultdict(dict)
for d in rows:
    if d.get('t')=='place':
        T[(d['job'],d['stack'],d['place'],d['rep'])].setdefault(d['impl'],[]).append(d['ms'])
for k,v in T.items(): print(*k,{a:[round(x,3) for x in b] for a,b in v.items()})
