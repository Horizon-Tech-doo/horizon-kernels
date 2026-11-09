# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Validate the frozen exploratory race and apply its predeclared selection rule."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

source, destination = map(Path, sys.argv[1:3])
records = [json.loads(line) for line in source.read_text().splitlines()]
assert records[-1]['test'] == 'complete'
assert records[-1]['primary_rows'] == 5 and records[-1]['all_eager_exact']
eager = [r for r in records if r['test'] == 'eager_correctness']
graph = [r for r in records if r['test'] == 'graph_correctness']
assert eager and all(r['exact'] for r in eager)
assert len(eager) == len(graph)
assert {(r['hidden'], r['rows'], r['name']) for r in eager} == {(r['hidden'], r['rows'], r['name']) for r in graph}
assert all(r['exact'] and r['counters_reset'] and r['replays'] == 2 for r in graph)
summary = {r['rows']: r for r in records if r['test'] == 'timing_summary'}
assert set(summary) == {1, 4, 8, 16, 32}
rounds = [r for r in records if r['test'] == 'timing_round']
assert len(rounds) == 15
for r in rounds:
    assert r['round'] in (0, 1, 2)
    for values in r['samples_us_per_mlp'].values():
        assert len(values) == 20 and all(math.isfinite(x) and x > 0 for x in values)
for rows, s in summary.items():
    rr = sorted((r for r in rounds if r['rows'] == rows), key=lambda r: r['round'])
    assert [r['round'] for r in rr] == [0, 1, 2]
    for name, medians in s['median_us_per_mlp'].items():
        assert medians == [statistics.median(r['samples_us_per_mlp'][name]) for r in rr]
    for name, ratios in s['speed_ratios'].items():
        assert ratios == [p/c for p,c in zip(s['median_us_per_mlp']['baseline'], s['median_us_per_mlp'][name])]
names = set.intersection(*(set(r['speed_ratios']) for r in summary.values()))
eligible = [name for name in sorted(names) if all(
    min(summary[rows]['speed_ratios'][name]) >= (1.02 if rows in (4, 32) else .98)
    for rows in summary)]
report = dict(eligible=eligible, decision='FRESH_CONFIRMATION_REQUIRED' if eligible else 'NO_SPEED_WIN',
              eager_candidates=len(eager), graph_candidates=len(graph), numerical_mismatches=0,
              primary_rows=sorted(summary), kernel_branch='horizon/k4-mlp-0410', candidate_manifest_sha256=hashlib.sha256(source.with_name('candidate.json').read_bytes()).hexdigest(),
              kernel_source_hashes={k:v for k,v in json.loads(source.with_name('candidate.json').read_text())['sha256'].items() if k.startswith('contrib/')},
              best_by_bucket=[dict(rows=rows, name=name, baseline_us=s['median_us_per_mlp']['baseline'],
                                  candidate_us=s['median_us_per_mlp'][name], ratios=s['speed_ratios'][name],
                                  conservative_latency_increase_pct=100*(1/min(s['speed_ratios'][name])-1))
                              for rows,s in sorted(summary.items())
                              for name in [max(s['speed_ratios'], key=lambda n: min(s['speed_ratios'][n]))]],
              limitation='Exploratory synthetic MLP only; per-bucket best is descriptive, not a fixed deployable policy.')
destination.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
