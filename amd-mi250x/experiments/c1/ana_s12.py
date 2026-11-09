# Experiment A table. Ratio = ours over hipBLASLt direct, throughput, higher is better; per round in the same process, then median.
import json, sys, collections, statistics as st
S = {'4096x4096x4096':'4k','8192x8192x8192':'8k','32768x1024x2048':'32k','4096x17408x5120':'17k','16384x16384x16384':'16k'}
for fn in sys.argv[1:]:
    print('##', fn.split('/')[-1])
    T = collections.defaultdict(dict); cs = collections.defaultdict(set); vs = {}; p = 0; addr = []
    for l in open(fn):
        if not l.startswith('{'):
            if l.strip() != 'DONE': print('LINE', l.strip()[:200])
            continue
        d = json.loads(l)
        if 'process' in d: p = d['process']; continue
        if 'A' in d: addr.append((p, d)); continue
        if d.get('check') == 'cs': cs[(d['shape'], d['label'])].add(d['cs']); continue
        if d.get('check') == 'vs_hipblaslt': vs.setdefault(d['shape'], []).append(d); continue
        T[(d['shape'], d['label'] if d['label'] == 'hipblaslt_direct' else d['label'].rsplit('_', 1)[0])][(p, d['round'])] = (d['TF'], d['pos'])
    shapes = []
    for k in T:
        if k[0] not in shapes: shapes.append(k[0])
    for s in shapes:
        d = T[(s, 'hipblaslt_direct')]
        all_cs = {c for (ss, _), v in cs.items() if ss == s for c in v}
        n_multi = sum(1 for (ss, _), v in cs.items() if ss == s and len(v) > 1)
        print(f"\n{S[s]} direct median {st.median(v[0] for v in d.values()):.2f} range {min(v[0] for v in d.values()):.2f}-{max(v[0] for v in d.values()):.2f} | distinct outputs {len(all_cs)}, arms with >1 output {n_multi} | " + '; '.join(f"{x['first_label']} rel {x['rel']} differ {x['elements_differ']}" for x in vs.get(s, []) if x['elements_differ']))
        print(f"{'arm':22s} {'TFmed':>7s} {'ratio':>6s} {'r0 p1':>7s} {'r0 p2':>7s} {'med p1':>7s} {'med p2':>7s} n")
        rows = []
        for (ss, a), v in T.items():
            if ss != s or a == 'hipblaslt_direct': continue
            rs = [v[k][0] / d[k][0] for k in v if k in d]
            P = lambda q: [x[0] for k, x in v.items() if k[0] == q]
            rows.append((st.median(rs), a, st.median(x[0] for x in v.values()), v.get((1, 0), (0,))[0], v.get((2, 0), (0,))[0], st.median(P(1)) if P(1) else 0, st.median(P(2)) if P(2) else 0, len(v)))
        for r in sorted(rows, reverse=True):
            print(f"{r[1]:22s} {r[2]:7.2f} {r[0]:6.3f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7.2f} {r[6]:7.2f} {r[7]}")
