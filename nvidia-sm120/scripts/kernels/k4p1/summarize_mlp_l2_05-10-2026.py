# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Apply the unchanged speed gate only after the stronger L2 replay checks."""
import json
from pathlib import Path
import subprocess
import sys

source, destination = map(Path, sys.argv[1:3])
records = [json.loads(line) for line in source.read_text().splitlines()]
eager = [r for r in records if r['test'] == 'eager_correctness']
changed = [r for r in records if r['test'] == 'changed_input_graph_correctness']
identity = lambda r: (r['hidden'], r['rows'], r['name'])
assert eager and len(changed) == len(eager)
assert set(map(identity, changed)) == set(map(identity, eager))
assert all(r['exact'] and r['counters_reset'] and r['replays'] == 64
           and r['scales'] == [1.0, 256.0] and r['layers'] == 4
           and r['stages'] == ['GU', 'ACT', 'DOWN'] for r in changed)
negative = [r for r in records if r['test'] == 'missing_producer_negative_control']
assert len(negative) == 1
assert negative[0]['rejected'] and negative[0]['device_error'] == negative[0]['mirror_error'] == 1
subprocess.run([sys.executable, str(Path(__file__).with_name('summarize_mlp_04-10-2026.py')),
                str(source), str(destination)], check=True, stdout=subprocess.DEVNULL)
report = json.loads(destination.read_text())
report['changed_input_graph_candidates'] = len(changed)
report['changed_input_replays_per_candidate'] = 64
report['changed_input_stages_per_layer'] = ['GU', 'ACT', 'DOWN']
report['negative_fixture'] = negative[0]
destination.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ('decision', 'eligible', 'eager_candidates',
                  'changed_input_graph_candidates', 'changed_input_replays_per_candidate')}, indent=2))
