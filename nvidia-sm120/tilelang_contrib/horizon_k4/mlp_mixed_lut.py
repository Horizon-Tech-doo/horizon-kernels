# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Isolated experimental SiLU factor lookup; no serving integration.

The float32 factor table is built by the original CUDA SiLU helper, once before
graph capture. Decode retains its final float32 multiplication/BF16 rounding.
GPU exhaustive factor/product checks and full MLP gates remain mandatory.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import runtime as rt, tasks as tk
from tilelang.contrib.horizon_k4.silu_factor_table import LUT_INDEX_SRC
from tilelang.contrib.horizon_k4.mlp import Mlp, SILU_SRC
from tilelang.contrib.horizon_k4.mlp_schedule import validate, build_up_fused


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mlp_mixed_lut_kernel(hidden, intermediate, wm, sg, sd, ncta, rows, stages=4,
                      timeout_ns=2_000_000_000, block_k=64, block_n=256, stages_down=4, block_n_down=128, activation_pair=True, gu_row_warps=1):
    validate(rows, hidden, intermediate, (sg, sd), ncta, 'operation')
    if block_k not in (64,128) or hidden % (sg*block_k) or intermediate % (sd*block_k):
        raise ValueError('unsupported GEMM K tile/split geometry')
    if block_n not in (128,256) or hidden % block_n or intermediate % block_n:
        raise ValueError("unsupported wide task geometry")
    if block_n_down not in (128,256) or hidden % block_n_down:
        raise ValueError("unsupported DOWN geometry")
    threads=max(block_n*gu_row_warps,block_n_down)
    max_stages=max(stages,stages_down)
    max_blocks=max(stages*block_n//32,stages_down*block_n_down//32)
    kch=block_k//32
    if sg != 1:
        raise ValueError('up-owned activation requires single-split GU')
    srcg, preg, gemmg = tk.gemm_task_hw('mlpg', 2 * intermediate, hidden, wm, sg, stages, block_k, block_n, threads, True,row_warps=gu_row_warps)
    srcd, pred, gemmd = tk.gemm_task_hw('mlpd', hidden, intermediate, wm, sd, stages_down, block_k, block_n_down, threads, True)
    src = rt.rt_src() + SILU_SRC + tk.hw_common_src() + srcg + srcd + LUT_INDEX_SRC
    activate = tk.silu_task_body(intermediate, intermediate // sd, 1)
    wait, signal, end = rt.macros(timeout_ns)
    m, maxt, nc = rows, T.dynamic('MAXT'), T.dynamic('NC')
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32
    ntick = (2 * intermediate + hidden) // 128
    tiles, per, cols = intermediate // block_n, intermediate // (sd * 128), intermediate // sd
    ngu, ndown = 2 * tiles, hidden // block_n_down * sd

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
             pd: T.Tensor((sd, m, hidden), f32), tickets: T.Tensor((ntick,), i32), lut: T.Tensor((65536,),f32)):
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
                    preg(wg, scaleg, bx, bk, gb[4], tid)
                    gemmg(x, gu, wg, scaleg, alphag, pg, tickets, 0, m,
                          bx, bk, tid, completed, *gb)
                    if bx < tiles:
                        if tid == 0:
                            T.call_extern('handle', 'hz_signal', T.access_ptr(counters[bx], 'rw'))
                    else:
                        tile = bx - tiles
                        dl[rt.D_W0], dl[rt.D_T0] = tile, 1
                        dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                        wait(dl, counters, err, ok, tid, errh)
                        if ok[0] == 1:
                            if activation_pair:
                                # Widths and row strides are even; each load is 4-byte aligned.
                                for row, pair in T.Parallel(m, block_n // 2):
                                    c = tile * block_n + 2 * pair
                                    gate_bits = T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(gu[row,c],"r"))
                                    up_bits = T.call_extern(T.uint32,'hz_k4_u32_cg',T.access_ptr(gu[row,intermediate+c],"r"))
                                    for v in T.unroll(2):
                                        a[row,c+v] = T.cast(
                                            lut[T.call_extern(T.uint32,'hz_k4_bf16_index',gate_bits,v)] *
                                            T.call_extern(f32,'hz_k4_bf16_pair_cvt',up_bits,v),bf16)
                            else:
                                for row, col in T.Parallel(m, block_n):
                                    c = tile * block_n + col
                                    a[row, c] = T.cast(T.call_extern(f32, 'hz_silu_mul_ind',
                                        T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,c],"r")), T.call_extern(f32,"hz_k4_bf16_cg",T.access_ptr(gu[row,intermediate+c],"r"))), bf16)
                        for half in T.unroll(block_n // 128):
                            signal(tiles + (tile * (block_n // 128) + half) // per, counters, tid, ok[0] == 1)
            for wave in T.unroll((ndown + ncta - 1) // ncta):
                task = (cta - ngu % ncta + ncta) % ncta + wave * ncta
                if task < ndown:
                    bx, bk = task // sd, task % sd
                    pred(wd, scaled, bx, bk, gb[4], tid)
                    dl[rt.D_W0], dl[rt.D_T0] = tiles + bk, per
                    dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                    wait(dl, counters, err, ok, tid, errh)
                    if ok[0] == 1:
                        gemmd(a, y, wd, scaled, alphad, pd, tickets, 2 * tiles, m,
                              bx, bk, tid, completed, *db)
            end(counters, nc, done, ncta, tid, finish, threads)
    return main


from tilelang.contrib.horizon_k4.mlp_mixed import MixedMlp
from tilelang.contrib.horizon_k4.silu_factor_table import factor_table

class LutMixedMlp(MixedMlp):
    def __init__(self,rows,hidden,intermediate,splits,ncta,device='cuda',
                 stages=3,block_k=64,block_n=256,stages_down=4,block_n_down=128):
        super().__init__(rows,hidden,intermediate,splits,ncta,device,stages,block_k,
                         block_n,stages_down,block_n_down,activation_pair=True)
        self.table=factor_table(device)
        kernel=mlp_mixed_lut_kernel(hidden,intermediate,16 if rows<=16 else 32,
               *splits,ncta,rows,stages,block_k=block_k,block_n=block_n,
               stages_down=stages_down,block_n_down=block_n_down,activation_pair=True)
        self.kernel=lambda *args:kernel(*args,self.table)
