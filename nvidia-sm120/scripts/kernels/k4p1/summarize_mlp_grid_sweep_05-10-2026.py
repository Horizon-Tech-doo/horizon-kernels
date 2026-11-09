# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Frozen full-five-row gate and matched grid effects; no bucket selection."""
import json
from pathlib import Path
import subprocess
import sys
source,destination=map(Path,sys.argv[1:3])
records=[json.loads(line) for line in source.read_text().splitlines()]
expected={(rows,f'gu128-down128-k64-sg4-sd4-up1-static1-hw1-stages4-grid{grid}-init00-operation')
          for rows in (1,4,8,16,32) for grid in (188,236,272,320,376)}
for test in ('eager_correctness','graph_correctness','changed_input_graph_correctness'):
    found=[r for r in records if r['test']==test]
    assert len(found)==25 and {(r['rows'],r['name']) for r in found}==expected
    assert all(r['exact'] for r in found)
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_l2_05-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
report=json.loads(destination.read_text())
effects=[]
for r in (r for r in records if r['test']=='timing_summary'):
    medians=r['median_us_per_mlp']
    def name(grid):return f'gu128-down128-k64-sg4-sd4-up1-static1-hw1-stages4-grid{grid}-init00-operation'
    for grid in (236,272,320,376):
        effects.append(dict(rows=r['rows'],grid=grid,
            grid188_over_candidate=[a/b for a,b in zip(medians[name(188)],medians[name(grid)])],
            grid376_over_candidate=[a/b for a,b in zip(medians[name(376)],medians[name(grid)])]))
assert len(effects)==20
report['matched_grid_effects']=effects
report['frozen_configuration_count']=25
report['gate_unchanged']=True
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('decision','eligible','eager_candidates','best_by_bucket')},indent=2))
