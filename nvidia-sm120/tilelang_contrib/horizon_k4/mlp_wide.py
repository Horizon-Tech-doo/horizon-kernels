# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Experimental up-owned activation fusion; unchanged per-output GEMM arithmetic.

Descriptors remain in the wrapper contract for evidence and CPU proof, but this
kernel derives tasks from compile-time geometry. No serving integration.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import runtime as rt, tasks as tk
from tilelang.contrib.horizon_k4.mlp import Mlp, SILU_SRC
from tilelang.contrib.horizon_k4.mlp_schedule import validate, build_up_fused


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mlp_wide_kernel(hidden, intermediate, wm, sg, sd, ncta, rows, stages=4,
                      timeout_ns=2_000_000_000, block_k=64, block_n=256):
    validate(rows, hidden, intermediate, (sg, sd), ncta, 'operation')
    if block_k not in (64,128) or hidden % (sg*block_k) or intermediate % (sd*block_k):
        raise ValueError('unsupported GEMM K tile/split geometry')
    if block_n not in (128,256) or hidden % block_n or intermediate % block_n:
        raise ValueError("unsupported wide task geometry")
    if sg != 1:
        raise ValueError('up-owned activation requires single-split GU')
    srcg, preg, gemmg = tk.gemm_task_hw('mlpg', 2 * intermediate, hidden, wm, sg, stages, block_k, block_n)
    srcd, pred, gemmd = tk.gemm_task_hw('mlpd', hidden, intermediate, wm, sd, stages, block_k, block_n)
    src = rt.rt_src() + SILU_SRC + tk.hw_common_src() + srcg + srcd
    activate = tk.silu_task_body(intermediate, intermediate // sd, 1)
    wait, signal, end = rt.macros(timeout_ns)
    m, maxt, nc = rows, T.dynamic('MAXT'), T.dynamic('NC')
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32
    ntick = (2 * intermediate + hidden) // 128
    tiles, per, cols = intermediate // block_n, intermediate // (sd * 128), intermediate // sd
    ngu, ndown = 2 * tiles, hidden // block_n * sd

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
             pd: T.Tensor((sd, m, hidden), f32), tickets: T.Tensor((ntick,), i32)):
        with T.Kernel(ncta, threads=block_n, prelude=src) as cta:
            tid = T.get_thread_binding()
            ok = T.alloc_shared((1,), i32)
            completed = T.alloc_shared((1,), i32)
            finish = T.alloc_shared((1,), i32)
            dl = T.alloc_local((rt.DESC,), i32)
            gb = tk.gemm_hw_alloc(wm, stages, block_k, block_n)
            tk.gemm_hw_addr(gb[0], gb[1], gb[2], gb[4])
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
                              bx, bk, tid, completed, *gb)
            end(counters, nc, done, ncta, tid, finish, block_n)
    return main


class WideMlp(Mlp):
    def __init__(self, rows, hidden, intermediate, splits, ncta, device='cuda',
                 stages=3, block_k=64, block_n=256):
        from tilelang.contrib.horizon_k4.mlp_wide_plan import WidePlan
        super().__init__(rows, hidden, intermediate, splits, ncta, device,
                         'operation', 1, stages, False, False, True)
        self.schedule = WidePlan(hidden, intermediate, splits, ncta, block_n)
        self.ts = self.schedule.tensors(device)
        self.kernel = mlp_wide_kernel(hidden, intermediate, 16 if rows<=16 else 32,
                                    *splits, ncta, rows, stages, block_k=block_k, block_n=block_n)
