# Experiment D: per-process rows at 16384^3. Ratio = ours over direct hipBLASLt, higher is better.
import json, sys, collections, statistics as st
p = None; R = collections.defaultdict(lambda: collections.defaultdict(list)); A = {}
for l in open(sys.argv[1]):
    if not l.startswith('{'): continue
    d = json.loads(l)
    if 'process' in d: p = d['process']; continue
    if 'A' in d: A[p] = (d['A'], d['B'], d['C']); continue
    if 'TF' in d: R[p][d['label'].rsplit('_', 1)[0] if d['label'] != 'hipblaslt_direct' else 'direct'].append(d['TF'])
arms = sorted({a for v in R.values() for a in v})
print('proc', *arms, 'A B C')
for k, v in R.items():
    print(k, *[','.join(f'{x:.1f}' for x in v.get(a, [])) for a in arms], *A.get(k, ()))
for a in arms:
    m = [st.median(v[a]) for v in R.values() if a in v]; print(a, 'per-process median range', min(m), max(m), f'swing {max(m)/min(m)-1:.3f}')
