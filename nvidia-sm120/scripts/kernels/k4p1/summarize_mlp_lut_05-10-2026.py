# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate raw M4/M32 lookup diagnostic evidence; never authorize promotion."""
import json
from pathlib import Path
import statistics
import sys

directory=Path(sys.argv[1])
records=[json.loads(line) for line in (directory/'bench.jsonl').read_text().splitlines()]
complete=[r for r in records if r['test']=='complete']
assert len(complete)==1 and complete[0]['primary_rows']==2 and complete[0]['all_eager_exact']
unit=[r for r in records if r['test']=='lut_table_check_complete']
assert len(unit)==1 and unit[0]['tested_products']==2083350 and unit[0]['unchanged']
negative=[r for r in records if r['test']=='missing_producer_negative_control']
assert len(negative)==1 and negative[0]['rejected'] and negative[0]['device_error']==negative[0]['mirror_error']==1
verified={}
effects=[]
for rows in (4,32):
    summaries=[r for r in records if r['test']=='timing_summary' and r['rows']==rows]
    assert len(summaries)==1
    summary=summaries[0]
    raw=[r for r in records if r['test']=='timing_round' and r['rows']==rows]
    assert len(raw)==3 and {r['round'] for r in raw}=={0,1,2}
    raw.sort(key=lambda r:r['round'])
    medians={name:[statistics.median(r['samples_us_per_mlp'][name]) for r in raw]
             for name in summary['median_us_per_mlp']}
    assert medians==summary['median_us_per_mlp']
    assert all(len(values)==20 and all(x>0 for x in values)
               for r in raw for values in r['samples_us_per_mlp'].values())
    assert len(medians)==5
    for name in medians:
        if name=='baseline':continue
        for kind in ('eager_correctness','graph_correctness','changed_input_graph_correctness'):
            proof=[r for r in records if r['test']==kind and r['rows']==rows and r['name']==name]
            assert len(proof)==1 and proof[0]['exact']
            if kind=='graph_correctness':assert proof[0]['replays']==2 and proof[0]['counters_reset']
            if kind=='changed_input_graph_correctness':
                assert proof[0]['replays']==64 and proof[0]['counters_reset']
                assert proof[0]['layers']==4 and proof[0]['stages']==['GU','ACT','DOWN']
        ratios=[p/x for p,x in zip(medians['baseline'],medians[name])]
        assert ratios==summary['speed_ratios'][name]
        verified[rows,name]=ratios
    for stages in (3,4):
        prefix=f'gu128-down128-k64-sg{stages}-sd4-up1-static1-hw1-stages{stages}-grid376-lut'
        control=prefix+'0-operation';lookup=prefix+'1-operation'
        effects.append(dict(rows=rows,stages=stages,
            control_over_lookup=[a/b for a,b in zip(medians[control],medians[lookup])],
            production_over_lookup=verified[rows,lookup]))
eligible=[stage for stage in (3,4) if all(min(e['production_over_lookup'])>=1.02
          for e in effects if e['stages']==stage)]
result=dict(decision='DIAGNOSTIC_SPEED_CANDIDATE' if eligible else 'NO_DIAGNOSTIC_SPEED_WIN',
    eligible_stages=eligible,matched_effects=effects,verified_configurations=8,
    full_five_row_gate_pending=True,serving_gate_pending=True)
(directory/'decision.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
