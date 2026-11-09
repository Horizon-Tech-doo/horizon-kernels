# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Frozen family validation before the unchanged full-five-row speed gate."""
import json,subprocess,sys
from pathlib import Path
from mlp_packed_native_configs_05_10_helper import configs
source,destination=map(Path,sys.argv[1:3])
records=[json.loads(s) for s in source.read_text().splitlines()]
expected={(rows,c['name']) for rows in (1,4,8,16,32) for c in configs(rows)}
for test in ('candidate','eager_correctness','graph_correctness','changed_input_graph_correctness'):
    found=[r for r in records if r['test']==test]
    assert len(found)==30 and {(r['rows'],r['name']) for r in found}==expected
    if test=='candidate':
        for r in found:assert r['configuration']==next(c for c in configs(r['rows']) if c['name']==r['name'])
    else:assert all(r['exact'] for r in found)
subprocess.run([sys.executable,str(Path(__file__).with_name('summarize_mlp_l2_05-10-2026.py')),
                str(source),str(destination)],check=True,stdout=subprocess.DEVNULL)
r=json.loads(destination.read_text());r['frozen_family_count']=6;r['gate_unchanged']=True
r['native_gu_policy']='N256; stages6 for rows<=16,4 for rows32, declared before GPU measurements; same whole family must pass every bucket.'
destination.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ('decision','eligible','eager_candidates','best_by_bucket')},indent=2))
