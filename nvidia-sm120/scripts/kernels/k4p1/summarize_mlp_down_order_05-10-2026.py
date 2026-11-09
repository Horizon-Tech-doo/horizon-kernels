# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Original full-five-row gate plus matched DOWN task-order effects."""
import json,subprocess,sys
from pathlib import Path
source,destination=map(Path,sys.argv[1:3])
records=[json.loads(s) for s in source.read_text().splitlines()]
def name(grid,major):return f'gu128-down128-k64-sg4-sd4-up1-static1-hw1-stages4-grid{grid}-downorder{int(major)}-operation'
expected={(rows,name(grid,major)) for rows in (1,4,8,16,32) for grid in (272,376) for major in (False,True)}
for test in ('eager_correctness','graph_correctness','changed_input_graph_correctness'):
    found=[r for r in records if r['test']==test]
    assert len(found)==20 and {(r['rows'],r['name']) for r in found}==expected
    assert all(r['exact'] for r in found)
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_l2_05-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
report=json.loads(destination.read_text());effects=[]
for summary in (r for r in records if r['test']=='timing_summary'):
    medians=summary['median_us_per_mlp']
    for grid in (272,376):
        effects.append(dict(rows=summary['rows'],grid=grid,
            tile_major_over_split_major=[a/b for a,b in zip(medians[name(grid,False)],medians[name(grid,True)])]))
assert len(effects)==10
report['matched_down_order_effects']=effects
report['gate_unchanged']=True
report['frozen_configuration_count']=20
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('decision','eligible','eager_candidates','best_by_bucket')},indent=2))
