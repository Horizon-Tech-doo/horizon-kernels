# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Apply the unchanged five-row speed gate after the vector8 correctness screen."""
import json
from pathlib import Path
import subprocess
import sys

source,destination=map(Path,sys.argv[1:3])
records=[json.loads(line) for line in source.read_text().splitlines()]
names={f'grid{g}-vector{v}' for g in (272,376) for v in (4,8)}
expected={(r,n) for r in (1,4,8,16,32) for n in names}
for test in ('eager_correctness','graph_correctness','changed_input_graph_correctness'):
    found=[r for r in records if r['test']==test]
    assert len(found)==20 and {(r['rows'],r['name']) for r in found}==expected
    assert all(r['hidden']==5120 and r['exact'] for r in found)
    if test=='graph_correctness':
        assert all(r['replays']==2 and r['counters_reset'] for r in found)
    if test=='changed_input_graph_correctness':
        assert all(r['replays']==64 and r['counters_reset'] and r['layers']==4
                   and r['scales']==[1.0,256.0] and r['stages']==['GU','ACT','DOWN'] for r in found)
negative=[r for r in records if r['test']=='missing_producer_negative_control']
assert len(negative)==2 and {r['vector'] for r in negative}=={4,8}
assert all(r['rejected'] and r['device_error']==r['mirror_error']==1 for r in negative)
working=[r for r in records if r['test']=='working_set']
assert len(working)==1 and working[0]['hidden']==5120 and working[0]['layers']==4
assert working[0]['weight_bytes']>512*1024**2 and working[0]['separator_bytes']==256*1024**2
summaries=[r for r in records if r['test']=='timing_summary']
assert len(summaries)==5 and {r['rows'] for r in summaries}=={1,4,8,16,32}
for r in summaries:
    assert set(r['speed_ratios'])==names and set(r['median_us_per_mlp'])==names|{'baseline'}
    assert r['rounds']==3 and r['iterations']==20
for r in records:
    if r['test']=='timing_round':
        assert set(r['samples_us_per_mlp'])==names|{'baseline'}
        assert r['hidden']==5120
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_04-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
report=json.loads(destination.read_text())
report.update(gate_unchanged=True,frozen_configuration_count=20,
              changed_input_replays_per_candidate=64,negative_fixtures=negative,
              matched_vector4_over_vector8=[dict(rows=r['rows'],grid=g,ratios=[a/b for a,b in zip(
                  r['median_us_per_mlp'][f'grid{g}-vector4'],r['median_us_per_mlp'][f'grid{g}-vector8'])])
                  for r in summaries for g in (272,376)])
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('decision','eligible','eager_candidates')},indent=2))
