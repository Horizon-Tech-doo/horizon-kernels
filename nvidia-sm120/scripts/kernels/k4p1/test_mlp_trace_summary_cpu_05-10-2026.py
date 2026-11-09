# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete, duplicated and nonmonotonic diagnostic task evidence."""
import copy
from bench_mlp_l2_05_10_helper import summarize_trace

records=[]
for bx in range(4):
    op=0 if bx<2 else 1
    records.append([100,110,150,160 if op else 0,180 if op else 0,190,bx%2,op,bx,0])
for bx in range(2):
    for bk in range(2):records.append([100,110,180,130,0,190,(bx+bk)%2,2,bx,bk])
trace=[[r] for r in records]
result=summarize_trace(trace,256,128,256,128,2,2)
assert result['tasks']==8 and result['sms_used']==2
assert result['ops']['2']['phases']['dependency_wait']['median_us']==.02
assert result['ops']['2']['phases']['gemm']['median_us']==.05
def rejects(t):
    try:summarize_trace(t,256,128,256,128,2,2)
    except AssertionError:return
    raise AssertionError('invalid evidence accepted')
rejects(trace[:-1])
bad=copy.deepcopy(trace);bad[-1]=copy.deepcopy(bad[-2]);rejects(bad)
bad=copy.deepcopy(trace);bad[-1][0][3]=200;rejects(bad)
bad=copy.deepcopy(trace);bad[-1][0][6]=2;rejects(bad)
bad=copy.deepcopy(trace);bad.append([[100,110,180,130,0,190,0,3,0,0]]);rejects(bad)
print('TRACE_SUMMARY_CPU_PASS phases and five invalid-evidence controls')
