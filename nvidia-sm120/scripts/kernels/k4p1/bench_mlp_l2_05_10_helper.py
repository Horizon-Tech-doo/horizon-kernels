# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only validation and phase summaries for mixed-MLP task timestamps."""
import statistics

def summarize_trace(trace,intermediate,gu_width,hidden,down_width,sd,sms):
    tasks=[r for cta in trace for r in cta if r[0]!=0]
    groups={op:[r for r in tasks if r[7]==op] for op in (0,1,2)}
    tiles=intermediate//gu_width
    assert {op:len(v) for op,v in groups.items()}=={0:tiles,1:tiles,2:hidden//down_width*sd}
    assert len({(r[7],r[8],r[9]) for r in tasks})==len(tasks)
    for r in tasks:
        assert len(r)==10 and 0<=r[6]<sms and r[7] in (0,1,2)
        order=(0,1,2,5) if r[7]==0 else ((0,1,2,3,4,5) if r[7]==1 else (0,1,3,2,5))
        assert all(0<r[a]<=r[b] for a,b in zip(order,order[1:])),'nonmonotonic task stamps'
        if r[7]<2:assert 0<=r[8]<2*tiles and r[9]==0 and (r[8]<tiles)==(r[7]==0)
        else:assert 0<=r[8]<hidden//down_width and 0<=r[9]<sd
    def stats(values):
        v=sorted(x/1000 for x in values)
        return dict(sum_us=sum(v),median_us=statistics.median(v),p95_us=v[min(len(v)-1,int(.95*len(v)))],max_us=max(v))
    result={}
    for op,records in groups.items():
        phases={'prefetch_issue':(0,1),'gemm':(3 if op==2 else 1,2)}
        if op>0:phases['dependency_wait']=(1 if op==2 else 2,3)
        if op==1:phases['activation']=(3,4)
        phases['publication']=(4 if op==1 else 2,5)
        result[str(op)]=dict(tasks=len(records),phases={name:stats([r[b]-r[a] for r in records])
                                                     for name,(a,b) in phases.items()})
    return dict(tasks=len(tasks),task_span_us=(max(r[5] for r in tasks)-min(r[0] for r in tasks))/1000,
                sms_used=len({r[6] for r in tasks}),ops=result,
                limitation='Instrumented task span excludes final grid reset and is not a speed measurement.')
