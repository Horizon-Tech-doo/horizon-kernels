# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile without a CUDA context; this is not execution validation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK']='1'
import torch
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
import json,sys
sys.path.insert(0,sys.argv[1])
from stable_align import stable_align_kernel
from tvm.target import Target
with Target({'kind':'cuda','arch':'sm_120a'}):
    for pairs,experts,bs in [(9,257,8),(288,257,8),(1152,257,16),(2304,257,16),(33,32,32)]:
        stable_align_kernel(pairs,experts,bs)
        print(json.dumps({'pairs':pairs,'experts':experts,'block_size':bs,'compiled':True}),flush=True)
