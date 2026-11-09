"""TRI3D graph step. raw/tri3g-*.jsonl. Suffix g = replayed from a torch.cuda graph. Ratios vLLM ms / ours ms, higher better for ours. V1 table at the end: 16 D256 layers x kernel ms, per rank, against V1 absolute step time."""
import json, glob, statistics as st, collections as C
ALL = {}
for f in sorted(glob.glob("raw/tri3g-*.jsonl")):
    J = f.split("-")[-1][:-6]; T = C.defaultdict(list); E = []; X = {}; chk = []
    for l in open(f):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r["t"] == "err": E.append(r); continue
        if r["t"] == "3dcheck": chk.append(r); continue
        k = (r["part"], r["Hkv"], r["B"], r["L"], r["mode"]); T[k + (r["impl"],)].append(r["ms"]); X[k] = r
    M = {k: st.median(v) for k, v in T.items()}; ALL[J] = M
    print("==", J, "err max", {i: f"{max(e['rel'] for e in E if e['impl']==i):.2e}" for i in sorted({e['impl'] for e in E})}, "nan", sum(e["rel"] != e["rel"] for e in E), "3D graph writes segm", all(c["segm_written_tri3"] > 0 for c in chk), len(chk))
    print(" cell | ms pick pickg tri3 tri3g trig rocg | tri3g/pickg trig/pickg rocg/pickg | tri3/pick eager | trig/tri3g | tri3/tri3g pick/pickg")
    for k in sorted(X):
        m = lambda n: M.get(k + (n,), float("nan"))
        print(f" {k[0]} H{k[1]} {k[2]:3d}x{k[3]:<6d}{k[4]:5s}| {m('pick'):.4f} {m('pickg'):.4f} {m('tri3'):.4f} {m('tri3g'):.4f} {m('trig'):.4f} {m('rocg'):.4f} | {m('tri3g')/m('pickg'):.3f} {m('trig')/m('pickg'):.3f} {m('rocg')/m('pickg'):.2f} | {m('tri3')/m('pick'):.3f} | {m('trig')/m('tri3g'):.3f} | {m('tri3')/m('tri3g'):.3f} {m('pick')/m('pickg'):.3f}" + (f" | kk {m('kk'):.4f} kkg {m('kkg'):.4f} tri3g/kkg {m('tri3g')/m('kkg'):.3f}" if k + ('kkg',) in M else ""))
    for part in "ZK":
        ks = [k for k in X if k[0] == part]; v = [M[k + ("tri3g",)] / M[k + ("pickg",)] for k in ks]
        print(f" {part} graph tri3g/pickg {min(v):.3f}-{max(v):.3f}, median {st.median(v):.3f}, below 1: {sum(x<1 for x in v)} of {len(v)}")
# V1: observed win_tok_s (V1/summary), medians over the 4 s_tri and s_rocm jobs and the two ip_ jobs
ST = {"s_tri":[32.9,180.6,648.5,32.8,178.5,630.2,32.1,174.0],"s_tri2":[33.8,182.0,668.2,33.4,180.1,649.0,32.6,174.5],"s_tri3":[33.0,181.7,667.7,32.8,180.2,646.8,32.1,175.2],"s_tri4":[33.0,179.6,648.4,32.9,177.3,627.0,32.1,173.1]}
RO = [[32.2,178.0,639.7,26.9,158.5,573.7,17.5,114.5],[32.2,177.5,652.0,26.9,158.3,582.7,17.6,115.0],[32.1,176.5,648.9,26.9,157.5,582.3,17.5,114.2],[32.1,175.3,634.0,26.8,156.8,570.2,17.5,114.0]]
IP = [[32.0,177.1,637.6,32.0,175.7,615.4,31.0,170.4],[32.9,177.9,659.4,32.6,176.7,636.3,31.8,172.1]]; ST2 = [ST["s_tri"], ST["s_tri2"]]
cells = [(1,1024),(8,1024),(32,1024),(1,2048),(8,2048),(32,2048),(1,8192),(8,8192)]; S = [1,8,32,1,8,32,1,8]
print("\nV1 closure, ms per step: lab 16 x kernel; observed step = streams / win_tok_s")
print("cell | step s_tri | obs rocm-tri | lab 16(rocg-tri3g) 16(rocg-trig) | lab headroom 16(tri3g-kkg) % of step, 16(tri3g-pickg) | obs ip_tri-s_tri ms (paired) | ratio obs")
for i, c in enumerate(cells):
    k = ("V", 2) + c + ("v1",); ms = {n: st.median(M[k + (n,)] for M in ALL.values()) for n in ("tri3g","trig","rocg","kkg","pickg","kk","tri3")}
    tri = [S[i] / ST[a][i] * 1000 for a in ST]; rocm = [S[i] / r[i] * 1000 for r in RO]; d = st.median(r - t for r, t in zip(rocm, tri))
    ip = [S[i] / x[i] * 1000 - S[i] / y[i] * 1000 for x, y in zip(IP, ST2)]; rt = [x[i] / y[i] for x, y in zip(IP, ST2)]
    h = 16 * (ms["tri3g"] - ms["kkg"]); h2 = 16 * (ms["tri3g"] - ms["pickg"]); s = st.median(tri)
    print(f" x{c[0]:<2d} L{c[1]:<5d}| {s:.1f} | {d:+.1f} | {16*(ms['rocg']-ms['tri3g']):+.1f} {16*(ms['rocg']-ms['trig']):+.1f} | {h:+.2f} ({100*h/s:+.1f}%) {h2:+.2f} ({100*h2/s:+.1f}%) | {min(ip):+.2f}..{max(ip):+.2f} | {min(rt):.3f}-{max(rt):.3f} | kernel ms tri3g {ms['tri3g']:.4f} kkg {ms['kkg']:.4f} pickg {ms['pickg']:.4f} trig {ms['trig']:.4f} rocg {ms['rocg']:.4f}")
print("\nTwo jobs side by side, graph: tri3g/pickg (trig/pickg); eager tri3/pick; rows where the jobs agree on side of 1 marked")
J = sorted(ALL); ks = sorted({k[:5] for k in ALL[J[0]]})
for k in ks:
    r = [ALL[j][k + ("tri3g",)] / ALL[j][k + ("pickg",)] for j in J]; t = [ALL[j][k + ("trig",)] / ALL[j][k + ("pickg",)] for j in J]; e = [ALL[j][k + ("tri3",)] / ALL[j][k + ("pick",)] for j in J]; c = [ALL[j][k + ("tri3g",)] / ALL[j][k + ("trig",)] for j in J]
    print(f" {k[0]} H{k[1]} {k[2]:3d}x{k[3]:<6d}{k[4]:5s} tri3g/pickg {r[0]:.3f} {r[1]:.3f} | trig/pickg {t[0]:.3f} {t[1]:.3f} | eager tri3/pick {e[0]:.3f} {e[1]:.3f} | tri3g/trig {c[0]:.3f} {c[1]:.3f}")
for p in "ZK":
    v = [ALL[j][k + ("tri3g",)] / ALL[j][k + ("pickg",)] for k in ks if k[0] == p for j in J]; print(p, "graph range", f"{min(v):.3f}-{max(v):.3f}", "median", f"{st.median(v):.3f}", "below 1", sum(x < 1 for x in v), "of", len(v))
for p, h in (("Z",4),("K",2)):
    base = [k for k in ks if k[0] == p and k[4] in ("base", "mid", "skew")]
    w = sorted(((max(ALL[j][k + ('tri3g',)] / ALL[j][k + ('pickg',)] for j in J)), k[2:5]) for k in base)[:4]; print(p, "lowest", [(round(a,3),b) for a,b in w])
