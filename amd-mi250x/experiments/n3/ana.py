import json,sys,statistics as st,collections
for fn in sys.argv[1:]:
    seq=None; T=collections.defaultdict(list); C={}; CK=collections.defaultdict(list)
    print("==",fn)
    for l in open(fn):
        if l.startswith("## seq"): seq=l.strip()[-1]; continue
        if not l.startswith("{"): print(l.strip()[:150]) if ("error" in l.lower() or "HIPERR" in l) else None; continue
        d=json.loads(l)
        if d["m"]=="setup": B=d["B"];L=d["L"];nb=d["bytes"];seq=d["seq"]
        if d["m"]=="time":
            T[(seq,d["k"])].append(d)
            if d["cs"]!="-": C[(seq,d["k"])]=d["cs"]
        if d["m"]=="clk": CK[(seq,d["k"])].append(d)
    print("B",B,"L",L,"bytes",nb)
    for (s,k),v in T.items():
        k1=[x["k1"] for x in v]; k2=[x["k2"] for x in v]; b=[x["both"] for x in v]
        t=st.median(b) if b[0]>0 else st.median(k1)
        print(f"seq{s} {k:10s} S{v[0]['S']:<4d} th{v[0]['th']:<4d} dyn{v[0]['dyn']:<6d} k1 {st.median(k1):7.4f} [{min(k1):.4f},{max(k1):.4f}] k2 {st.median(k2):.4f} both {st.median(b):.4f} GB/s {nb/t*1e-6:7.1f} %{nb/t*1e-6/16:5.1f} cs {C.get((s,k),'-')}")
    for (s,k),v in CK.items():
        print(f"clk seq{s} {k:10s} mhz_med {st.median(x['mhz_med'] for x in v):.0f} p5 {st.median(x['mhz_p5'] for x in v):.0f} p95 {st.median(x['mhz_p95'] for x in v):.0f} ticks {st.median(x['ticks_med'] for x in v):.0f} span_us {st.median(x['span_us'] for x in v):.0f} cus {v[0]['cus']} peak_blk_cu {st.median(x['peak_blk_cu_mean'] for x in v):.2f}")
