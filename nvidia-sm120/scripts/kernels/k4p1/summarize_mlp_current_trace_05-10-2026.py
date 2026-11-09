# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only temporal overlap of labelled task intervals, not hardware utilization."""
from collections import defaultdict,Counter
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

records=[json.loads(line) for line in (directory/'bench.jsonl').read_text().splitlines()]
assert records[-1]['test']=='complete'
correct=[r for r in records if r['test']=='profile_correctness']
assert len(correct)==4 and {(r['rows'],r['width'],r['stages'],r['grid']) for r in correct}=={(r,128,4,g) for r in (4,32) for g in (188,376)}
assert all(r['exact'] and r['replays']==64 and r['activation_pair'] and r['scales']==[1,256]
           and r['layers']==4 and r['stages_checked']==['GU','ACT','DOWN'] and r['counters_reset'] for r in correct)
negative=[r for r in records if r['test']=='missing_producer_negative_control']
assert len(negative)==1 and negative[0]['rejected'] and negative[0]['device_error']==negative[0]['mirror_error']==1
logged=[r for r in records if r['test']=='task_profile']
assert len(logged)==96 and all(r['exact'] for r in logged)
raw=[]
with gzip.open(directory/'bench.timestamps.jsonl.gz','rt') as f:raw=[json.loads(line) for line in f]
assert len(raw)==96
key=lambda r:tuple(r[k] for k in ('rows','width','stages','grid','input_scale','trial','layer'))
assert len({key(r) for r in raw})==96
assert {key(r) for r in raw}=={(rows,128,4,grid,scale,trial,layer) for rows in (4,32)
                           for grid in (188,376) for scale in (1,256) for trial in range(3) for layer in range(4)}
for a,b in zip(logged,raw):
    assert key(a)==key(b) and a['seed']==b['seed']==1042026+b['rows']*100+b['input_scale']*1000+b['trial']
    assert a['summary']==summarize_trace(b['trace'],17408,128,5120,128,8 if b['rows']==4 else 4,188)
profiles=[]
for record in raw:
    summarize_trace(record['trace'],17408,record['width'],5120,128,
                    8 if record['rows']==4 else 4,188)
    tasks=[r for cta in record['trace'] for r in cta if r[0]]
    producer=[r for r in tasks if r[7] in (0,1)]
    start=min(r[1] for r in producer);end=max(r[5] for r in producer)
    span=end-start
    gu=defaultdict(list);waiting=defaultdict(list);down=defaultdict(list);up_wait=defaultdict(list);activation=defaultdict(list)
    for r in tasks:
        sm=r[6];op=r[7]
        def clipped(a,b):return [max(start,r[a])-start,min(end,r[b])-start]
        if op in (0,1):
            gu[sm].append(clipped(1,2))
            if op==1:
                up_wait[sm].append(clipped(2,3));activation[sm].append(clipped(3,4))
        else:
            waiting[sm].append(clipped(1,3));down[sm].append(clipped(3,2))
    cta_sms=[]
    for cta in record['trace']:
        mapped={r[6] for r in cta if r[0]}
        assert len(mapped)==1
        cta_sms.append(next(iter(mapped)))
    placement=Counter(Counter(cta_sms).values())
    denominator=188*span
    phase=summarize_trace(record["trace"],17408,128,5120,128,8 if record["rows"]==4 else 4,188)
    values=dict(producer_window_us=span/1000,
        gu_task_sm_time_fraction=sum(duration(v) for v in gu.values())/denominator,
        down_wait_sm_time_fraction=sum(duration(v) for v in waiting.values())/denominator,
        down_gemm_sm_time_fraction=sum(duration(v) for v in down.values())/denominator,
        gu_and_down_wait_same_sm_fraction=sum(intersection(gu[sm],waiting[sm]) for sm in range(188))/denominator)
    values['task_span_us']=phase['task_span_us']
    values['up_wait_sm_time_fraction']=sum(duration(v) for v in up_wait.values())/denominator
    values['activation_sm_time_fraction']=sum(duration(v) for v in activation.values())/denominator
    values['gu_and_up_wait_same_sm_fraction']=sum(intersection(gu[sm],up_wait[sm]) for sm in range(188))/denominator
    values['gu_and_activation_same_sm_fraction']=sum(intersection(gu[sm],activation[sm]) for sm in range(188))/denominator
    values['cta_per_sm_histogram']=dict(sorted(placement.items()))
    values['sms_used']=len(set(cta_sms))
    values['operation_placement']={}
    for op in (0,1,2):
        counts=Counter(r[6] for r in tasks if r[7]==op)
        assert counts
        values['operation_placement'][str(op)]=dict(
            sms_used=len(counts),tasks=sum(counts.values()),
            max_tasks_per_sm=max(counts.values()))
    for label,op,name in [('up_gemm',1,'gemm'),('up_wait',1,'dependency_wait'),('activation',1,'activation'),('down_wait',2,'dependency_wait'),('down_gemm',2,'gemm')]:
        values[label+'_median_us']=phase['ops'][str(op)]['phases'][name]['median_us']
    assert all(0<=value<=1 for key,value in values.items() if key.endswith('_fraction'))
    profiles.append(dict({k:record[k] for k in ('rows','width','stages','grid','input_scale','seed','trial','layer')},**values))
assert len(profiles)==96
summary=[]
for cfg in sorted({tuple(r[k] for k in ('rows','width','stages','grid','input_scale')) for r in profiles}):
    selected=[r for r in profiles if tuple(r[k] for k in ('rows','width','stages','grid','input_scale'))==cfg]
    assert len(selected)==12
    placement_variants=Counter(json.dumps(r['cta_per_sm_histogram'],sort_keys=True) for r in selected)
    summary.append(dict(zip(('rows','width','stages','grid','input_scale'),cfg),
        placement_variants=[dict(histogram=json.loads(hist),profiles=count) for hist,count in sorted(placement_variants.items())],
        sms_used_median=statistics.median(r['sms_used'] for r in selected),
        operation_placement_medians={op:{metric:statistics.median(
            r['operation_placement'][op][metric] for r in selected)
            for metric in ('sms_used','tasks','max_tasks_per_sm')}
            for op in ('0','1','2')},
        **{key:statistics.median(r[key] for r in selected) for key in selected[0] if key.endswith('_fraction') or key.endswith('_us')}))
report=dict(decision='DIAGNOSTIC_ONLY',profiles=profiles,summary=summary,
    limitation='Instrumented task labels include copy/memory/math waits. SM-time fractions are interval unions, not utilization counters or speed evidence. Matched paired-read width128/stage4 traces compare grid188/376 at normal/stress inputs; instrumentation can change placement. No speed-gate claim.')
(directory/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
