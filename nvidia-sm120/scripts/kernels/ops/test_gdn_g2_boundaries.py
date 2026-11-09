# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""G2 boundary/final-slot writes and invalid-request no-ops vs G1.

Run under the exclusive GPU lock: python test_gdn_g2_boundaries.py OUT.jsonl.
Requires the existing ops dev environment. This does not test conv compaction.
"""
import json
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__import__("os").environ.get("HK_WORK", ".")) / "ops"))
import test_gdn_g2 as base

P = base.tg.real_params(0)
gen = torch.Generator(device='cuda').manual_seed(31)
nr, nl, width = 7, 2, 4
acc_host = [1, 2, 3, 4, 0, 4, 4]
acc = torch.tensor(acc_host, dtype=torch.int32, device='cuda')
ones = torch.ones(nr, dtype=torch.int32, device='cuda')
sidx1 = torch.arange(1, nr*width+1, dtype=torch.int32, device='cuda').view(nr, width)
sidx2 = torch.arange(1, nr+1, dtype=torch.int32, device='cuda').view(nr, 1)
s1 = [base.tg.make_state(nr*width+1, gen)[0] for _ in range(nl)]
s2 = [base.tg.make_state(3*nr+1, gen)[0] for _ in range(nl)]
for a, b in zip(s1, s2):
    for r in range(nr):
        b[r+1].copy_(a[r*width+1])
initial = [s.clone() for s in s2]
corr, kn, dbuf = base.buffers(nl, nr)
mixed, a, b, z, cu = base.tg.make_inputs([width]*nr, gen)
for li in range(nl):
    o1 = torch.empty(width*nr, base.HV, base.DV, device='cuda', dtype=torch.bfloat16)
    o2 = torch.empty_like(o1)
    base.gd.fused_gdn_decode_post_conv_mtp(mixed,a,b,P['A_log'],P['dt_bias'],sidx1,cu,ones,s1[li],z,P['norm_w'],o1,base.DK**-.5,1e-6,'silu')
    base.gr.verify(mixed,a,b,P['A_log'],P['dt_bias'],sidx2,cu,ones,s2[li],z,P['norm_w'],o2,corr[li],kn[li],dbuf[li],base.DK**-.5,1e-6,'silu')
    assert base.tg.bits_equal(o1,o2)==0
addrs = torch.tensor([s.data_ptr() for s in s2], dtype=torch.int64, device='cuda')
src = sidx2[:,0].contiguous(); src[5] = 0
fin = torch.arange(nr+1, 2*nr+1, dtype=torch.int32, device='cuda'); fin[6] = 0
bnd = torch.arange(2*nr+1, 3*nr+1, dtype=torch.int32, device='cuda')
with open(sys.argv[1], 'w') as out:
    for boundary in range(5):
        lens = torch.full((nr,), boundary, dtype=torch.int32, device='cuda')
        expected = [s.clone() for s in initial]
        for li in range(nl):
            s2[li].copy_(initial[li])
            for r in range(4):
                expected[li][nr+r+1].copy_(s1[li][r*width+acc_host[r]])
                if 0 < boundary <= acc_host[r]:
                    expected[li][2*nr+r+1].copy_(s1[li][r*width+boundary])
        for vr in (32,128):
            def commit():
                base.gr.commit(addrs,src,fin,bnd,lens,acc,corr,kn,dbuf,base.HV,base.H,s2[0].stride(0),vr=vr)
            commit(); torch.cuda.synchronize()
            assert all(base.tg.bits_equal(x,y)==0 for x,y in zip(s2,expected))
            for attempt in range(3):
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):
                        commit()
                    break
                except RuntimeError:
                    if attempt==2: raise
            for _ in range(2):
                for x,y in zip(s2,initial): x.copy_(y)
                graph.replay(); torch.cuda.synchronize()
                assert all(base.tg.bits_equal(x,y)==0 for x,y in zip(s2,expected))
            row=dict(boundary=boundary,vr=vr,layers=nl,requests=nr,graph_replays=2,ok=True)
            out.write(json.dumps(row)+'\n');out.flush();print(json.dumps(row),flush=True)
    out.write(json.dumps(dict(complete=True,cases=10))+'\n')
