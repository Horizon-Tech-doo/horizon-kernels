# Order effects in Experiment A: each arm's TF over its own median, grouped by round, position and predecessor family; also paired arm ratios for the stop rule.
import json, sys, collections, statistics as st
for fn in sys.argv[1:]:
    print('##', fn.split('/')[-1])
    p = 0; rows = []
    for l in open(fn):
        if not l.startswith('{'): continue
        d = json.loads(l)
        if 'process' in d: p = d['process']; continue
        if 'TF' in d: rows.append((d['shape'], p, d['round'], d['pos'], d['label'] if d['label'] == 'hipblaslt_direct' else d['label'].rsplit('_', 1)[0], d['TF']))
    med = collections.defaultdict(list)
    for r in rows: med[(r[0], r[4])].append(r[5])
    med = {k: st.median(v) for k, v in med.items()}
    seq = {(r[0], r[1], r[2], r[3]): r for r in rows}
    fam = lambda a: a.split('.')[0]
    for s in sorted({r[0] for r in rows}):
        G = collections.defaultdict(list); R = collections.defaultdict(list); P = collections.defaultdict(list); Pos = collections.defaultdict(list)
        for r in rows:
            if r[0] != s or r[4] == 'hipblaslt_direct': continue
            x = r[5] / med[(s, r[4])]
            R[(r[1], r[2])].append(x); Pos[min(r[3] // 4, 4)].append(x)
            q = seq.get((s, r[1], r[2], r[3] - 1)); G[fam(q[4]) if q else 'first'].append(x)
            if r[4].startswith('b12864.fill') or r[4].startswith('b12864.cl'): P[fam(q[4]) if q else 'first'].append(x)
        print(s)
        print(' by process,round:', {k: round(st.median(v), 3) for k, v in sorted(R.items())})
        print(' by position/4:', {k: round(st.median(v), 3) for k, v in sorted(Pos.items())})
        print(' by predecessor, all arms:', {k: (round(st.median(v), 3), len(v)) for k, v in sorted(G.items())})
        print(' by predecessor, gapped b12864:', {k: (round(st.median(v), 3), len(v)) for k, v in sorted(P.items())})
    T = {(r[0], r[1], r[2], r[4]): r[5] for r in rows}
    for s in sorted({r[0] for r in rows}):
        for base, arms in [('rr25632w2.asis', ['rr25632w2.rep3', 'rr25632w2.rep7', 'rr25632w2.ctrl', 'rr25632w2.cl1', 'rr25632w2.hand']), ('rr12864w3.asis', ['rr12864w3.rep3', 'rr12864w3.rep7', 'rr12864w3.ctrl']), ('b12864.asis', ['b12864.fill0', 'b12864.fill3', 'b12864.fill7', 'b12864.ctrl', 'b12864.cl1', 'b12864.cl2', 'b12864.rep3', 'b12864.rep7']), ('b25632a2.asis', ['b25632a2.fill3'])]:
            o = []
            for a in arms:
                v = [T[(s, p, r, a)] / T[(s, p, r, base)] for (ss, p, r, aa) in T if ss == s and aa == base and (s, p, r, a) in T]
                if not v: continue
                o.append(f"{a.split('.')[1]} {st.median(v):.3f}[{min(v):.3f},{max(v):.3f}]")
            print(f"  {s[:8]} {base}: " + ' '.join(o))
