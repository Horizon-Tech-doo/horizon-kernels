# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Original five-row gate plus descriptive matched initial-copy effects."""
import json
from pathlib import Path
import subprocess
import sys

source,destination=map(Path,sys.argv[1:3])
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_l2_05-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
records=[json.loads(line) for line in source.read_text().splitlines()]
effects=[]
for summary in (r for r in records if r['test']=='timing_summary'):
    medians=summary['median_us_per_mlp']
    for stage in (3,4):
        prefix=f'gu128-down128-k64-sg{stage}-sd4-up1-static1-hw1-stages{stage}-grid376-init'
        control=prefix+'00-operation'
        for flags in ('01','10','11'):
            name=prefix+flags+'-operation'
            assert name in medians and control in medians
            effects.append(dict(rows=summary['rows'],stages=stage,flags=flags,
                control_over_init=[p/x for p,x in zip(medians[control],medians[name])]))
assert len(effects)==30
report=json.loads(destination.read_text())
report['matched_staged_init_effects']=effects
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('decision','eligible','eager_candidates','best_by_bucket')},indent=2))
