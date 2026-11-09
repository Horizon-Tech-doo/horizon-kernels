import json,collections,statistics as st,math,sys
D=collections.defaultdict(list);I={}
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    r=json.loads(l); D[r['label']].append(r['TF']); I[r['label']]=r
for k in D:
    r=I[k];B=r['B'];v=D[k]
    print(f"{k:24s} M={r['M']} N={r['N']} MN={r['M']*r['N']/1e6:.2f}M B={B} ceil={math.ceil(B/110)} eff={(B/110)/math.ceil(B/110):.3f} TF {st.median(v):.2f} [{min(v):.2f},{max(v):.2f}] n={len(v)}")
