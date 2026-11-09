# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic isolation of the existing K4 task body; no serving integration.

No producer waits or global completion counter are needed: independent split-K
jobs finish at the kernel boundary. Ticket reduction/reset is the existing body.
Isolating an operation changes compiler liveness and removes scheduling overhead;
its timing cannot be substituted for the combined MLP speed gate.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import tasks as tk


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def isolated_gemm_kernel(n,k,rows,split,grid,width=128,stages=4,threads=128):
    if rows not in (4,32) or width not in (128,256) or threads not in (128,256) or threads<width:
        raise ValueError('unsupported diagnostic geometry')
    wm=16 if rows<=16 else 32
    if grid not in (188,376) or stages not in (4,6):
        raise ValueError('unsupported diagnostic grid/pipeline')
    src,pre,body=tk.gemm_task_hw('isolated',n,k,wm,split,stages,64,width,threads,True)
    ntasks=n//width*split
    bf16,f32,i32=T.bfloat16,T.float32,T.int32
    @T.prim_func
    def main(x:T.Tensor((rows,k),bf16),w:T.Tensor((n//32,k//32,128),i32),
             s:T.Tensor((n//32,k//32,16),i32),alpha:T.Tensor((n,),f32),
             y:T.Tensor((rows,n),bf16),partial:T.Tensor((split,rows,n),f32),
             tickets:T.Tensor((n//width,),i32)):
        with T.Kernel(grid,threads=threads,prelude=tk.hw_common_src()+src) as cta:
            tid=T.get_thread_binding()
            completed=T.alloc_shared((1,),i32)
            a=T.alloc_shared((stages,wm,64),bf16)
            wq=T.alloc_shared((stages*width//32,2,128),i32)
            scales=T.alloc_shared((stages*width//32,2,16),i32)
            ticket=T.alloc_shared((1,),i32)
            sb=T.alloc_local((3,),T.uint32)
            T.annotate_layout({a:tk.make_mma_swizzle_layout(a)})
            tk.gemm_hw_addr(a,wq,scales,sb)
            for wave in T.unroll((ntasks+grid-1)//grid):
                task=cta+wave*grid
                if task<ntasks:
                    bx,bk=task//split,task%split
                    pre(w,s,bx,bk,sb,tid)
                    body(x,y,w,s,alpha,partial,tickets,0,rows,bx,bk,tid,completed,a,wq,scales,ticket,sb)
    return main


class IsolatedGemm:
    def __init__(self,n,k,rows,split,grid,width=128,stages=4,threads=128,device='cuda'):
        import torch
        self.n,self.k,self.rows=n,k,rows
        self.kernel=isolated_gemm_kernel(n,k,rows,split,grid,width,stages,threads)
        self.y=torch.empty((rows,n),device=device,dtype=torch.bfloat16)
        self.partial=torch.empty((split,rows,n),device=device,dtype=torch.float32)
        self.tickets=torch.zeros(n//width,device=device,dtype=torch.int32)
    def __call__(self,x,weight):
        import torch
        if x.shape!=(self.rows,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():
            raise ValueError('declared contiguous BF16 input required')
        self.kernel(x,weight.Wg,weight.Sg,weight.alpha,self.y,self.partial,self.tickets)
        return self.y
    def check(self):
        import torch
        if torch.count_nonzero(self.tickets).item():
            raise RuntimeError('split tickets did not reset')
