import json,collections,statistics as st,math,sys
D=collections.defaultdict(list)
for l in open(sys.argv[1]):
    if l.startswith('{"m":"scan"') or l.startswith('{"m":"long"'):
        r=json.loads(l); D[(r['m'],r['k'],r['P'],r['dyn'],r['B'])].append(r['TF'])
out=[]
for (m,k,P,dyn) in sorted({x[:4] for x in D}):
    rows={x[4]:D[x] for x in D if x[:4]==(m,k,P,dyn)}
    sat=st.median([st.median(v) for B,v in rows.items() if B%110==0 and B>=880])
    print(f"## {m} {k} P={P} dyn={dyn} sat={sat:.2f} (median of exact multiples B>=880)")
    errA=[];errB=[]
    for B in sorted(rows):
        v=rows[B]; med=st.median(v); s=(max(v)-min(v))/med*100
        a=(B/110)/math.ceil(B/110)*sat
        slots=110*P; b=(B/slots)/math.ceil(B/slots)*191.5
        errA.append(abs(a-med)/med*100);errB.append(abs(b-med)/med*100)
        print(f"B={B:5d} n={len(v)} med={med:7.2f} spread%={s:4.1f} ceil110={a:7.2f} ({(a-med)/med*100:+5.1f}%)  sec5.5(slots=110P,peak191.5)={b:7.2f} ({(b-med)/med*100:+5.1f}%)")
    print(f"   max|err| ceil110 {max(errA):.1f}%  sec5.5 {max(errB):.1f}%")
