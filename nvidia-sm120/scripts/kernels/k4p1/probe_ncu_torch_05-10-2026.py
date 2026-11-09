# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import sys
print('PYTHON_START',flush=True)
import torch
print('TORCH_IMPORTED',flush=True)
if len(sys.argv)>1 and sys.argv[1]=='cpu':
 print('CPU_IMPORT_PASS',flush=True);sys.exit(0)
x=torch.ones(1<<20,device='cuda');torch.cuda.synchronize()
print('TORCH_CUDA_ALLOCATED',flush=True)
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph,capture_error_mode='thread_local'):y=x+2
print('TORCH_GRAPH_CAPTURED',flush=True)
graph.replay();torch.cuda.synchronize();assert torch.all(y==3).item()
torch.cuda.nvtx.range_push('hz_profile')
graph.replay();torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
assert torch.all(y==3).item()
print('TORCH_CUDA_GRAPH_PASS',flush=True)
