# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Recompute every diagnostic phase summary from the retained raw timestamps."""
import gzip
import json
from pathlib import Path
import statistics
import sys
from bench_mlp_l2_05_10_helper import summarize_trace

directory=Path(sys.argv[1])
records=[json.loads(x) for x in (directory/'bench.jsonl').read_text().splitlines()]
assert records[-1]['test']=='complete'
correct=[x for x in records if x['test']=='profile_correctness']
assert len(correct)==6 and all(x['exact'] and x['replays']==8 for x in correct)
cfg=lambda x:tuple(x[k] for k in ('rows','width','stages','grid'))
assert set(map(cfg,correct))=={(r,w,s,188) for r in (4,32) for w,s in ((128,6),(256,3),(256,4))}
timings=[x for x in records if x['test']=='task_profile']
assert len(timings)==72
with gzip.open(directory/'bench.timestamps.jsonl.gz','rt') as f:raw=[json.loads(x) for x in f]
assert len(raw)==72
assert {(cfg(x),x['trial'],x['layer']) for x in raw}=={
    (cfg(c),trial,layer) for c in correct for trial in range(3) for layer in range(4)}
for a,b in zip(timings,raw):
    assert all(a[k]==b[k] for k in ('rows','width','stages','grid','trial','layer'))
    assert a['summary']==summarize_trace(b['trace'],17408,b['width'],5120,128,
                                      8 if b['rows']==4 else 4,188)
summary=[]
for c in correct:
    v=[x['summary'] for x in timings if cfg(x)==cfg(c)]
    median=lambda fn:statistics.median([fn(x) for x in v])
    s={k:c[k] for k in ('rows','width','stages','grid')}
    s['task_span_us']=median(lambda x:x['task_span_us'])
    for name,op,phase in (('up_gemm',1,'gemm'),('up_wait',1,'dependency_wait'),
                          ('activation',1,'activation'),('down_gemm',2,'gemm'),
                          ('down_wait',2,'dependency_wait')):
        s[name+'_median_us']=median(lambda x:x['ops'][str(op)]['phases'][phase]['median_us'])
    summary.append(s)
report=dict(decision='DIAGNOSTIC_ONLY',exact_configurations=6,profiles=72,
            raw_recomputed=True,summary=summary,
            limitation='Instrumentation changes execution; CTA phase medians overlap and cannot be added into wall time. No speed/deployment claim.')
(directory/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
