# Experiments B and C table. Ratio = ours over hipBLASLt via torch, throughput, higher is better; per round, then median. Round 0 shown separately.
import json, sys, collections, statistics as st
for fn in sys.argv[1:]:
    print('##', fn.split('/')[-1]); T = collections.defaultdict(dict); E = collections.defaultdict(list); M = {}
    for l in open(fn):
        if not l.startswith('{'):
            if l.strip() != 'DONE': print('LINE', l.strip()[:200])
            continue
        d = json.loads(l)
        if 'TF' in d:
            T[d['shape']].setdefault(d['arm'], {})[d['round']] = d['TF']
            if 'fp32_rel' in d: E[(d['shape'], d['arm'])].append((d['fp32_rel'], d['nan']))
        elif 'arm' in d: M[(d['shape'], d['arm'])] = d
    for s, A in T.items():
        tr = A['hipblaslt_torch']
        print(f"\n{s} torch median {st.median(tr.values()):.2f} range {min(tr.values()):.2f}-{max(tr.values()):.2f} r0 {tr[0]:.2f}; fp32_rel torch {max(x[0] for x in E[(s,'hipblaslt_torch')]):.3g}")
        rows = []
        for a, v in A.items():
            if a == 'hipblaslt_torch': continue
            m = M[(s, a)]; R = m.get('resource')
            rows.append((st.median(v[r] / tr[r] for r in v), a, st.median(v.values()), v[0], max(x[0] for x in E[(s, a)]), sum(x[1] for x in E[(s, a)]), m['bounds'], m['compile_s']))
        for r in sorted(rows, reverse=True):
            print(f"  {r[1]:14s} TF {r[2]:7.2f} ratio {r[0]:.3f} r0 {r[3]:7.2f} fp32_rel {r[4]:.3g} nan {r[5]} {r[6]} compile {r[7]}s")
