# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only temporal overlap of labelled task intervals, not hardware utilization."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys
from bench_mlp_l2_05_10_helper import summarize_trace

directory=Path(sys.argv[1])
def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if end<=start:continue
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result

def duration(intervals):return sum(end-start for start,end in union(intervals))
def intersection(left,right):
    left,right=union(left),union(right);i=j=total=0
    while i<len(left) and j<len(right):
        total+=max(0,min(left[i][1],right[j][1])-max(left[i][0],right[j][0]))
        if left[i][1]<=right[j][1]:i+=1
        else:j+=1
    return total

profiles=[]
with gzip.open(directory/'bench.timestamps.jsonl.gz','rt') as f:
    for line in f:
        record=json.loads(line)
        summarize_trace(record['trace'],17408,record['width'],5120,128,
                        8 if record['rows']==4 else 4,188)
        tasks=[r for cta in record['trace'] for r in cta if r[0]]
        producer=[r for r in tasks if r[7] in (0,1)]
        start=min(r[1] for r in producer);end=max(r[5] for r in producer)
        span=end-start
        gu=defaultdict(list);waiting=defaultdict(list);down=defaultdict(list)
        for r in tasks:
            sm=r[6];op=r[7]
            def clipped(a,b):return [max(start,r[a])-start,min(end,r[b])-start]
            if op in (0,1):gu[sm].append(clipped(1,2))
            else:
                waiting[sm].append(clipped(1,3));down[sm].append(clipped(3,2))
        denominator=188*span
        values=dict(producer_window_us=span/1000,
            gu_task_sm_time_fraction=sum(duration(v) for v in gu.values())/denominator,
            down_wait_sm_time_fraction=sum(duration(v) for v in waiting.values())/denominator,
            down_gemm_sm_time_fraction=sum(duration(v) for v in down.values())/denominator,
            gu_and_down_wait_same_sm_fraction=sum(intersection(gu[sm],waiting[sm]) for sm in range(188))/denominator)
        assert all(0<=value<=1 for key,value in values.items() if key.endswith('_fraction'))
        profiles.append(dict({k:record[k] for k in ('rows','width','stages','grid','trial','layer')},**values))
assert len(profiles)==72
summary=[]
for cfg in sorted({tuple(r[k] for k in ('rows','width','stages','grid')) for r in profiles}):
    selected=[r for r in profiles if tuple(r[k] for k in ('rows','width','stages','grid'))==cfg]
    assert len(selected)==12
    summary.append(dict(zip(('rows','width','stages','grid'),cfg),
        **{key:statistics.median(r[key] for r in selected) for key in selected[0] if key.endswith('_fraction') or key=='producer_window_us'}))
report=dict(decision='DIAGNOSTIC_ONLY',profiles=profiles,summary=summary,
    limitation='Instrumented task labels include copy/memory/math waits. SM-time fractions are interval unions, not utilization counters or speed evidence. All traces use grid188; current grid376/stage4 control is not represented.')
(directory/'overlap-decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
