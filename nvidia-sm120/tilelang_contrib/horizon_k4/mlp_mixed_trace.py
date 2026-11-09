# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Instrumented asymmetric MLP for phase diagnosis; never a speed-gate candidate.

Copy of the mixed arithmetic path with per-task globaltimer records. Records
separate weight prefetch, GEMM body and dependency waits; retain raw stamps.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import runtime as rt, tasks as tk
from tilelang.contrib.horizon_k4.mlp import Mlp, SILU_SRC
from tilelang.contrib.horizon_k4.mlp_schedule import validate, build_up_fused


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mlp_mixed_trace_kernel(hidden, intermediate, wm, sg, sd, ncta, rows, stages=4,
                      timeout_ns=2_000_000_000, block_k=64, block_n=256, stages_down=4, block_n_down=128, activation_pair=False):
    validate(rows, hidden, intermediate, (sg, sd), ncta, 'operation')
    if block_k not in (64,128) or hidden % (sg*block_k) or intermediate % (sd*block_k):
        raise ValueError('unsupported GEMM K tile/split geometry')
    if block_n not in (128,256) or hidden % block_n or intermediate % block_n:
        raise ValueError("unsupported wide task geometry")
    if block_n_down not in (128,256) or hidden % block_n_down:
        raise ValueError("unsupported DOWN geometry")
    threads=max(block_n,block_n_down)
    max_stages=max(stages,stages_down)
    max_blocks=max(stages*block_n//32,stages_down*block_n_down//32)
    kch=block_k//32
    if sg != 1:
        raise ValueError('up-owned activation requires single-split GU')
    srcg, preg, gemmg = tk.gemm_task_hw('mlpg', 2 * intermediate, hidden, wm, sg, stages, block_k, block_n, threads, True)
    srcd, pred, gemmd = tk.gemm_task_hw('mlpd', hidden, intermediate, wm, sd, stages_down, block_k, block_n_down, threads, True)
    src = rt.rt_src() + SILU_SRC + tk.hw_common_src() + srcg + srcd
    activate = tk.silu_task_body(intermediate, intermediate // sd, 1)
    wait, signal, end = rt.macros(timeout_ns)
    m, maxt, nc = rows, T.dynamic('MAXT'), T.dynamic('NC')
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32
    ntick = (2 * intermediate + hidden) // 128
    tiles, per, cols = intermediate // block_n, intermediate // (sd * 128), intermediate // sd
    ngu, ndown = 2 * tiles, hidden // block_n_down * sd
    gu_waves = (ngu + ncta - 1) // ncta
    down_waves = (ndown + ncta - 1) // ncta
    slots = gu_waves + down_waves

    @T.prim_func
    def main(desc: T.Tensor((ncta, maxt, rt.DESC), i32), sigtab: T.Tensor((ncta, maxt), i32),
             ntask: T.Tensor((ncta,), i32), counters: T.Tensor((nc,), i32),
             done: T.Tensor((1,), i32), err: T.Tensor((1,), i32), errh: T.Tensor((1,), i32),
             x: T.Tensor((m, hidden), bf16),
             wg: T.Tensor((2 * intermediate // 32, hidden // 32, 128), i32),
             scaleg: T.Tensor((2 * intermediate // 32, hidden // 32, 16), i32),
             alphag: T.Tensor((2 * intermediate,), f32),
             wd: T.Tensor((hidden // 32, intermediate // 32, 128), i32),
             scaled: T.Tensor((hidden // 32, intermediate // 32, 16), i32),
             alphad: T.Tensor((hidden,), f32),
             gu: T.Tensor((m, 2 * intermediate), bf16), a: T.Tensor((m, intermediate), bf16),
             y: T.Tensor((m, hidden), bf16), pg: T.Tensor((sg, m, 2 * intermediate), f32),
             pd: T.Tensor((sd, m, hidden), f32), tickets: T.Tensor((ntick,), i32), trace: T.Tensor((ncta, slots, 10), T.int64)):
        with T.Kernel(ncta, threads=threads, prelude=src) as cta:
            tid = T.get_thread_binding()
            ok = T.alloc_shared((1,), i32)
            completed = T.alloc_shared((1,), i32)
            finish = T.alloc_shared((1,), i32)
            dl = T.alloc_local((rt.DESC,), i32)
            A_s = T.alloc_shared((max_stages,wm,block_k),bf16)
            W_s = T.alloc_shared((max_blocks,kch,128),i32)
            S_s = T.alloc_shared((max_blocks,kch,16),i32)
            ticket = T.alloc_shared((1,),i32)
            sb = T.alloc_local((3,),T.uint32)
            T.annotate_layout({A_s: tk.make_mma_swizzle_layout(A_s)})
            # Both phases index one packed block arena with their own nblk stride.
            gb = (A_s,W_s,S_s,ticket,sb)
            db = gb
            tk.gemm_hw_addr(A_s,W_s,S_s,sb)
            # Exact round-robin operation schedule, expressed as arithmetic.
            # No descriptors, op dispatch or dynamic per-CTA task loop.
            for wave in T.unroll((ngu + ncta - 1) // ncta):
                task = cta + wave * ncta
                if task < ngu:
                    bx, bk = task // sg, task % sg
                    if tid == 0:
                        trace[cta,wave,0] = T.call_extern(T.int64,'hz_gtime')
                        trace[cta,wave,6] = T.call_extern(T.int64,'hz_smid')
                        trace[cta,wave,7] = T.cast(T.if_then_else(bx < tiles,0,1),T.int64)
                        trace[cta,wave,8] = T.cast(bx,T.int64)
                        trace[cta,wave,9] = T.cast(bk,T.int64)
                    preg(wg, scaleg, bx, bk, gb[4], tid)
                    if tid == 0:
                        trace[cta,wave,1] = T.call_extern(T.int64,'hz_gtime')
                    gemmg(x, gu, wg, scaleg, alphag, pg, tickets, 0, m,
                          bx, bk, tid, completed, *gb)
                    if tid == 0:
                        trace[cta,wave,2] = T.call_extern(T.int64,'hz_gtime')
                    if bx < tiles:
                        if tid == 0:
                            T.call_extern('handle', 'hz_signal', T.access_ptr(counters[bx], 'rw'))
                    else:
                        tile = bx - tiles
                        dl[rt.D_W0], dl[rt.D_T0] = tile, 1
                        dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                        wait(dl, counters, err, ok, tid, errh)
                        if tid == 0:
                            trace[cta,wave,3] = T.call_extern(T.int64,'hz_gtime')
                        if ok[0] == 1:
                            if activation_pair:
                                # Widths and row strides are even; each load is 4-byte aligned.
                                for row, pair in T.Parallel(m, block_n // 2):
                                    c = tile * block_n + 2 * pair
                                    gate_bits = T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(gu[row,c],"r"))
                                    up_bits = T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(gu[row,intermediate+c],"r"))
                                    for v in T.unroll(2):
                                        a[row,c+v] = T.cast(T.call_extern(f32,'hz_silu_mul_ind',
                                            T.call_extern(f32,'hz_k4_bf16_pair_cvt',gate_bits,v),
                                            T.call_extern(f32,'hz_k4_bf16_pair_cvt',up_bits,v)),bf16)
                            else:
                                for row, col in T.Parallel(m, block_n):
                                    c = tile * block_n + col
                                    a[row, c] = T.cast(T.call_extern(f32, 'hz_silu_mul_ind',
                                        T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,c],"r")), T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,intermediate+c],"r"))), bf16)
                        T.sync_threads()
                        if tid == 0:
                            trace[cta,wave,4] = T.call_extern(T.int64,'hz_gtime')
                        for half in T.unroll(block_n // 128):
                            signal(tiles + (tile * (block_n // 128) + half) // per, counters, tid, ok[0] == 1)
                    if tid == 0:
                        trace[cta,wave,5] = T.call_extern(T.int64,'hz_gtime')
            for wave in T.unroll((ndown + ncta - 1) // ncta):
                task = (cta - ngu % ncta + ncta) % ncta + wave * ncta
                if task < ndown:
                    bx, bk = task // sd, task % sd
                    if tid == 0:
                        trace[cta,gu_waves+wave,0] = T.call_extern(T.int64,'hz_gtime')
                        trace[cta,gu_waves+wave,6] = T.call_extern(T.int64,'hz_smid')
                        trace[cta,gu_waves+wave,7] = 2
                        trace[cta,gu_waves+wave,8] = T.cast(bx,T.int64)
                        trace[cta,gu_waves+wave,9] = T.cast(bk,T.int64)
                    pred(wd, scaled, bx, bk, gb[4], tid)
                    if tid == 0:
                        trace[cta,gu_waves+wave,1] = T.call_extern(T.int64,'hz_gtime')
                    dl[rt.D_W0], dl[rt.D_T0] = tiles + bk, per
                    dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                    wait(dl, counters, err, ok, tid, errh)
                    if tid == 0:
                        trace[cta,gu_waves+wave,3] = T.call_extern(T.int64,'hz_gtime')
                    if ok[0] == 1:
                        gemmd(a, y, wd, scaled, alphad, pd, tickets, 2 * tiles, m,
                              bx, bk, tid, completed, *db)
                    if tid == 0:
                        trace[cta,gu_waves+wave,2] = T.call_extern(T.int64,'hz_gtime')
                        trace[cta,gu_waves+wave,5] = T.call_extern(T.int64,'hz_gtime')
            end(counters, nc, done, ncta, tid, finish, threads)
    return main


from tilelang.contrib.horizon_k4.mlp_mixed import MixedMlp

class TraceMixedMlp(MixedMlp):
    def __init__(self, rows, hidden, intermediate, splits, ncta, device='cuda',
                 stages=3, block_k=64, block_n=256, stages_down=4, block_n_down=128, activation_pair=False):
        import torch
        super().__init__(rows,hidden,intermediate,splits,ncta,device,stages,block_k,block_n,stages_down,block_n_down,activation_pair=activation_pair)
        ngu = 2 * intermediate // block_n
        ndown = hidden // block_n_down * splits[1]
        slots = (ngu+ncta-1)//ncta + (ndown+ncta-1)//ncta
        self.trace = torch.zeros((ncta,slots,10),dtype=torch.int64,device=device)
        kernel = mlp_mixed_trace_kernel(hidden,intermediate,16 if rows<=16 else 32,
                 *splits,ncta,rows,stages,block_k=block_k,block_n=block_n,
                 stages_down=stages_down,block_n_down=block_n_down,activation_pair=activation_pair)
        self.kernel = lambda *args: kernel(*args,self.trace)
