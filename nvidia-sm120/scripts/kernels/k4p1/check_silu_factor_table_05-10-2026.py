# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU bit checks: every finite BF16 gate, 32 up patterns and changed-input graphs."""
import json
import torch
from tilelang.contrib.horizon_k4.silu_factor_table import factor_table,product_kernel,reference_factor_kernel

def check_table(emit):
    table=factor_table()
    assert table.dtype==torch.float32 and table.numel()==65536
    original=table.view(torch.int32).clone()
    codes=torch.arange(65536,dtype=torch.int32,device='cuda')
    gates=codes.to(torch.int16).view(torch.bfloat16)
    finite_gate=torch.isfinite(gates)
    assert int(finite_gate.sum())==65280
    x=torch.empty((65536,2),dtype=torch.bfloat16,device='cuda')
    x[:,0].copy_(gates);x[:,1].fill_(1)
    factors=torch.empty(65536,dtype=torch.float32,device='cuda')
    reference_factor_kernel()(x,factors);torch.cuda.synchronize()
    mismatches=int(torch.count_nonzero(table.view(torch.int32)[finite_gate]!=factors.view(torch.int32)[finite_gate]))
    emit(dict(test='lut_factor_bits',finite_gates=65280,mismatches=mismatches))
    assert mismatches==0,'table factors differ from the original helper with loaded up=1'
    baseline=torch.compile(lambda z:torch.nn.functional.silu(z[:,0])*z[:,1],fullgraph=True,dynamic=False)
    baseline(x);torch.cuda.synchronize()
    output=torch.empty(65536,dtype=torch.bfloat16,device='cuda')
    product_kernel()(x,table,output);torch.cuda.synchronize()
    graphs=[]
    for fn in (lambda:baseline(x),lambda:product_kernel()(x,table,output)):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,capture_error_mode='thread_local'):result=fn()
        graphs.append((graph,result))
    constants=[0,0x8000,0x3f80,0xbf80,0x0001,0x8001,0x0080,0x8080,0x7f7f,0xff7f]
    total=0
    for trial in range(32):
        if trial<len(constants):up_codes=torch.full_like(codes,constants[trial])
        else:up_codes=(codes*131+7919*trial)&65535
        up=up_codes.to(torch.int16).view(torch.bfloat16)
        x[:,1].copy_(up)
        graphs[0][0].replay();graphs[1][0].replay();torch.cuda.synchronize()
        expected=graphs[0][1]
        valid=finite_gate&torch.isfinite(up)
        tested=int(valid.sum());total+=tested
        mismatches=int(torch.count_nonzero(output.view(torch.int16)[valid]!=expected.view(torch.int16)[valid]))
        emit(dict(test='lut_product_bits',trial=trial,tested=tested,mismatches=mismatches,
                  source='independent Torch/Inductor',graph=True,changed_input=True))
        assert mismatches==0,('SiLU table product differs',trial,mismatches)
    assert total>=2000000
    assert torch.equal(original,table.view(torch.int32))
    emit(dict(test='lut_table_check_complete',finite_gate_patterns=65280,up_patterns=32,
              tested_products=total,table_bytes=table.numel()*table.element_size(),unchanged=True))
