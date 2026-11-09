# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate the M32 diagnostic without treating it as a five-bucket speed gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

source,destination=map(Path,sys.argv[1:3])
records=[json.loads(x) for x in source.read_text().splitlines()]
assert records[-1]['test']=='complete' and records[-1]['primary_rows']==1 and records[-1]['all_eager_exact']
eager=[r for r in records if r['test']=='eager_correctness']
graph=[r for r in records if r['test']=='graph_correctness']
changed=[r for r in records if r['test']=='changed_input_graph_correctness']
identity=lambda r:(r['hidden'],r['rows'],r['name'])
assert eager and len(eager)==len(graph)==len(changed)
assert set(map(identity,eager))==set(map(identity,graph))==set(map(identity,changed))
assert all(r['exact'] and r['rows']==32 for r in eager)
assert all(r['exact'] and r['counters_reset'] and r['replays']==2 for r in graph)
assert all(r['exact'] and r['counters_reset'] and r['replays']==64 and r['layers']==4
           and r['scales']==[1.0,256.0] and r['stages']==['GU','ACT','DOWN'] for r in changed)
negative=[r for r in records if r['test']=='missing_producer_negative_control']
assert len(negative)==1 and negative[0]['rejected'] and negative[0]['device_error']==negative[0]['mirror_error']==1
rounds=sorted((r for r in records if r['test']=='timing_round'),key=lambda r:r['round'])
assert len(rounds)==3 and [r['round'] for r in rounds]==[0,1,2]
summaries=[r for r in records if r['test']=='timing_summary'];assert len(summaries)==1
s=summaries[0];assert s['rows']==32
for name,medians in s['median_us_per_mlp'].items():
    values=[r['samples_us_per_mlp'][name] for r in rounds]
    assert all(len(v)==20 and all(math.isfinite(x) and x>0 for x in v) for v in values)
    assert medians==[statistics.median(v) for v in values]
for name,ratios in s['speed_ratios'].items():
    assert ratios==[a/b for a,b in zip(s['median_us_per_mlp']['baseline'],s['median_us_per_mlp'][name])]
eligible=[n for n,v in s['speed_ratios'].items() if '-rw2-' in n and min(v)>=1.02]
effects=[]
for n in s['speed_ratios']:
    if '-rw1-' not in n:continue
    other=n.replace('-rw1-','-rw2-')
    if other in s['speed_ratios']:
        effects.append(dict(control=n,candidate=other,control_over_candidate=[a/b for a,b in
                            zip(s['median_us_per_mlp'][n],s['median_us_per_mlp'][other])]))
best=max(s['speed_ratios'],key=lambda n:min(s['speed_ratios'][n]))
report=dict(decision='M32_CANDIDATE_REQUIRES_FULL_GATE' if eligible else 'NO_ROW_SPEED_WIN',
            eligible=eligible,rows=[32],eager_candidates=len(eager),changed_input_replays=64,
            best_name=best,best_production_ratios=s['speed_ratios'][best],matched_effects=effects,
            negative_fixture=negative[0],candidate_manifest_sha256=hashlib.sha256(source.with_name('candidate.json').read_bytes()).hexdigest(),
            limitation='M32 diagnostic only. Original five-bucket gate, fresh confirmation and integrated serving validation remain required; no deployment.')
destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
