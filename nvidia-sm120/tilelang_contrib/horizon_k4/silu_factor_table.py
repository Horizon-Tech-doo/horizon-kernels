# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Immutable float32 SiLU factors for every BF16 bit pattern, per CUDA device.

Build before graph capture, using the same CUDA helper as the persistent MLP.
No lower-precision factor storage or approximate exponential is introduced.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4.mlp import SILU_SRC
from tilelang.contrib.horizon_k4.tasks import cache_src

LUT_INDEX_SRC=r"""
__device__ __forceinline__ unsigned hz_k4_bf16_index(unsigned bits,int high) {
  return (bits >> (16*high)) & 0xffffu;
}
"""

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def factor_kernel():
    @T.prim_func
    def main(table:T.Tensor((65536,),T.float32)):
        with T.Kernel(512,threads=128,prelude=SILU_SRC+cache_src()) as bx:
            tid=T.get_thread_binding()
            index=bx*128+tid
            gate=T.call_extern(T.float32,'hz_k4_bf16_pair_cvt',T.cast(index,T.uint32),0)
            table[index]=T.call_extern(T.float32,'hz_silu_mul_ind',gate,T.float32(1.0))
    return main

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def product_kernel():
    @T.prim_func
    def main(x:T.Tensor((65536,2),T.bfloat16),table:T.Tensor((65536,),T.float32),
             output:T.Tensor((65536,),T.bfloat16)):
        with T.Kernel(512,threads=128,prelude=cache_src()+LUT_INDEX_SRC) as bx:
            tid=T.get_thread_binding()
            index=bx*128+tid
            bits=T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(x[index,0],"r"))
            gate_index=T.call_extern(T.uint32,'hz_k4_bf16_index',bits,0)
            up=T.call_extern(T.float32,'hz_k4_bf16_pair_cvt',bits,1)
            output[index]=T.cast(table[gate_index]*up,T.bfloat16)
    return main

@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED:True})
def reference_factor_kernel():
    @T.prim_func
    def main(x:T.Tensor((65536,2),T.bfloat16),output:T.Tensor((65536,),T.float32)):
        with T.Kernel(512,threads=128,prelude=SILU_SRC+cache_src()) as bx:
            tid=T.get_thread_binding()
            index=bx*128+tid
            bits=T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(x[index,0],"r"))
            gate=T.call_extern(T.float32,'hz_k4_bf16_pair_cvt',bits,0)
            up=T.call_extern(T.float32,'hz_k4_bf16_pair_cvt',bits,1)
            output[index]=T.call_extern(T.float32,'hz_silu_mul_ind',gate,up)
    return main

_TABLES={}
def factor_table(device='cuda'):
    import torch
    dev=torch.device(device)
    if dev.type!='cuda':raise ValueError('SiLU factor table requires CUDA')
    if dev.index is None:dev=torch.device('cuda',torch.cuda.current_device())
    if dev.index not in _TABLES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('SiLU factor table must be initialized before capture')
        with torch.cuda.device(dev):
            table=torch.empty(65536,dtype=torch.float32,device=dev)
            factor_kernel()(table)
            # Consumers on another stream must not race table initialization.
            torch.cuda.synchronize(dev)
        _TABLES[dev.index]=table
    return _TABLES[dev.index]
