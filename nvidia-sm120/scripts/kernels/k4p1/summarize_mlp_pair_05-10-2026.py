# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Retain the frozen production gate and describe matched scalar/pair effects."""
import json
from pathlib import Path
import subprocess
import sys

source,destination=map(Path,sys.argv[1:3])
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_l2_05-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
records=[json.loads(x) for x in source.read_text().splitlines()]
summaries=[r for r in records if r['test']=='timing_summary']
effect=[]
for row in summaries:
    medians=row['median_us_per_mlp']
    for scalar in medians:
        if '-pair0-' not in scalar:continue
        paired=scalar.replace('-pair0-','-pair1-')
        if paired not in medians:continue
        effect.append(dict(rows=row['rows'],scalar=scalar,paired=paired,
                           scalar_over_pair=[a/b for a,b in zip(medians[scalar],medians[paired])]))
assert effect and {r['rows'] for r in effect}=={1,4,8,16,32}
report=json.loads(destination.read_text())
report['matched_pair_effects']=effect
report['matched_pair_limitation']='Descriptive within-K4 comparison; the original fixed-configuration production speed gate remains unchanged.'
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(decision=report['decision'],eligible=report['eligible'],
                     correctness_candidates=report['eager_candidates'],matched_pairs=len(effect)),indent=2))
