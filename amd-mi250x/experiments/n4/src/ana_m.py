# Join gpu_metrics samples with timed arm windows. Usage: ana_m.py <run dir> [card]
import json,glob,os,sys,statistics as st,bisect
d=sys.argv[1]
S=[json.loads(l) for l in open(d+'/metrics.jsonl') if l.endswith('}\n')]
T=[s['t'] for s in S]
cards=[k for k in S[0] if k.startswith('card')]
W=[]
for f in sorted(glob.glob(d+'/*.jsonl')):
    if 'metrics' in f: continue
    for l in open(f,errors='replace'):
        try:r=json.loads(l)
        except: continue
        if 't0' in r: W.append((os.path.basename(f),r))
# which card is ours: highest mean activity correlation during windows vs gaps; print per card mean act in windows
def win(r):
    i,j=bisect.bisect_left(T,r['t0']),bisect.bisect_right(T,r['t1']); return S[i:j]
ma={c:st.mean(s[c]['act'] for w in W for s in win(w[1]) if c in s) for c in cards}
print('mean act in windows',ma)
ours=sys.argv[2] if len(sys.argv)>2 else None
if not ours:
    # ours: gfxclk goes idle right after the job's end / lowest activity outside windows
    inwin=set()
    for w in W:
        for s in win(w[1]): inwin.add(s['t'])
    oa={c:st.mean(s[c]['act'] for s in S if s['t'] not in inwin) for c in cards}
    print('mean act outside',oa)
    ours=max(cards,key=lambda c:ma[c]-oa[c])
print('ours',ours)
twin={'card0':'card1','card1':'card0','card2':'card3','card3':'card2','card4':'card5','card5':'card4','card6':'card7','card7':'card6'}[ours]
lead=ours if int(ours[4:])%2==0 else twin
agg={}
for fn,r in W:
    X=win(r)
    if not X: continue
    k=(fn.split('_')[1] if fn.startswith('hb') else fn.split('_')[-1],r['data'],r['impl'])
    a=agg.setdefault(k,dict(tf=[],pw=[],ag=[],cg=[],thr=set(),ith=set(),act=[],umc=[],tw=[],pr=[],th=[]))
    a['tf'].append(r['TF'])
    if 'mhz_med' in r:a['pr'].append(r['mhz_med'])
    for s in X:
        c=s[ours];a['ag'].append(c['agfx']);a['cg'].append(c['cgfx']);a['act'].append(c['act']);a['umc'].append(c['umc']);a['thr'].add(c['thr']);a['ith'].add(c['ithr']);a['th'].append(c['th']);a['pw'].append(s[lead]['pw']);a['tw'].append(s[twin]['act'])
print('shape data arm TF probeMHz cgfx agfx pw act umc hotspot twinact thr')
for k,a in sorted(agg.items()):
    print(*k,f"{st.median(a['tf']):.2f} {st.median(a['pr']) if a['pr'] else 0:.0f} {st.median(a['cg']):.0f} {st.median(a['ag']):.0f} {st.median(a['pw']):.0f} {st.median(a['act']):.0f} {st.median(a['umc']):.0f} {max(a['th'])} {st.mean(a['tw']):.0f}",sorted(a['thr']),sorted(a['ith']))
