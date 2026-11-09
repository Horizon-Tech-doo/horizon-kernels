# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate raw isolated-operation measurements; no MLP eligibility decision."""
import json,math,statistics,sys
from pathlib import Path
from isolated_gemm_configs_05_10_helper import configs,label
source,destination=map(Path,sys.argv[1:3])
r=[json.loads(s) for s in source.read_text().splitlines()]
assert r[-1]['test']=='complete' and r[-1]['decision']=='DIAGNOSTIC_ONLY'
weights={x['n']:x for x in r if x['test']=='working_set'}
assert set(weights)=={34816,5120}
assert weights[34816]['weights']==6 and weights[5120]['weights']==12
assert all(x['weight_bytes']>512*1024**2 for x in weights.values())
expected={(n,m,label(c)) for n in (34816,5120) for m in (4,32) for c in configs(m,n)}
quality=[x for x in r if x['test']=='correctness']
assert len(quality)==20 and {(x['n'],x['rows'],x['name']) for x in quality}==expected
assert all(x['exact'] and x['replays']==64 and x['scales']==[1,256] and x['tickets_reset']
           and x['weights']==weights[x['n']]['weights'] for x in quality)
summaries=[x for x in r if x['test']=='timing_summary']
assert len(summaries)==4 and {(x['n'],x['rows']) for x in summaries}=={(n,m) for n in (34816,5120) for m in (4,32)}
rounds=[x for x in r if x['test']=='timing_round'];assert len(rounds)==12
for x in summaries:
    trials=sorted((t for t in rounds if (t['n'],t['rows'])==(x['n'],x['rows'])),key=lambda t:t['round'])
    assert [t['round'] for t in trials]==[0,1,2]
    names={'baseline'}|{label(c) for c in configs(x['rows'],x['n'])}
    assert set(x['median_us_per_gemm'])==names and set(x['production_over_probe'])==names-{'baseline'}
    for t in trials:
        assert set(t['samples_us_per_gemm'])==names
        assert all(len(v)==20 and all(math.isfinite(a) and a>0 for a in v) for v in t['samples_us_per_gemm'].values())
    for name,medians in x['median_us_per_gemm'].items():
        assert medians==[statistics.median(t['samples_us_per_gemm'][name]) for t in trials]
        if name!='baseline':assert x['production_over_probe'][name]==[a/b for a,b in zip(x['median_us_per_gemm']['baseline'],medians)]
report=dict(decision='DIAGNOSTIC_ONLY',correctness_configs=20,replays=64,
            summaries=summaries,limitation=r[-1]['limitation'],weights=weights)
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(decision=report['decision'],correctness_configs=20,
                     production_over_probe=[dict(n=x['n'],rows=x['rows'],ratios=x['production_over_probe']) for x in summaries]),indent=2))
