# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Experimental arithmetic task plan; same round-robin schedule and GEMM arithmetic.

Descriptors remain in the wrapper contract for evidence and CPU proof, but this
kernel derives tasks from compile-time geometry. No serving integration.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import runtime as rt, tasks as tk
from tilelang.contrib.horizon_k4.mlp import Mlp, SILU_SRC
from tilelang.contrib.horizon_k4.mlp_schedule import validate


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mlp_static_kernel(hidden, intermediate, wm, sg, sd, ncta, rows, stages=4,
                      timeout_ns=2_000_000_000, block_k=64):
    validate(rows, hidden, intermediate, (sg, sd), ncta, 'operation')
    if block_k not in (64,128) or hidden % (sg*block_k) or intermediate % (sd*block_k):
        raise ValueError('unsupported GEMM K tile/split geometry')
    srcg, preg, gemmg = tk.gemm_task_hw('mlpg', 2 * intermediate, hidden, wm, sg, stages, block_k)
    srcd, pred, gemmd = tk.gemm_task_hw('mlpd', hidden, intermediate, wm, sd, stages, block_k)
    src = rt.rt_src() + SILU_SRC + tk.hw_common_src() + srcg + srcd
    activate = tk.silu_task_body(intermediate, intermediate // sd, 1)
    wait, signal, end = rt.macros(timeout_ns)
    m, maxt, nc = rows, T.dynamic('MAXT'), T.dynamic('NC')
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32
    ntick = (2 * intermediate + hidden) // 128
    tiles, per, cols = intermediate // 128, intermediate // (sd * 128), intermediate // sd
    ngu, nact, ndown = 2 * tiles * sg, rows * sd, hidden // 128 * sd

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
        with T.Kernel(ncta, threads=tk.THREADS, prelude=src) as cta:
            tid = T.get_thread_binding()
            ok = T.alloc_shared((1,), i32)
            completed = T.alloc_shared((1,), i32)
            finish = T.alloc_shared((1,), i32)
            dl = T.alloc_local((rt.DESC,), i32)
            gb = tk.gemm_hw_alloc(wm, stages, block_k)
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
                    signal((bx % tiles) // per, counters, tid, completed[0] == 1)
            for wave in T.unroll((nact + ncta - 1) // ncta):
                task = (cta - ngu % ncta + ncta) % ncta + wave * ncta
                if task < nact:
                    slice_index, row = task // m, task % m
                    dl[rt.D_W0], dl[rt.D_T0] = slice_index, 2 * per
                    dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                    wait(dl, counters, err, ok, tid, errh)
                    if ok[0] == 1:
                        activate(gu, a, row, slice_index * cols, tid)
                    signal(sd + slice_index, counters, tid, ok[0] == 1)
            for wave in T.unroll((ndown + ncta - 1) // ncta):
                task = (cta - (ngu + nact) % ncta + ncta) % ncta + wave * ncta
                if task < ndown:
                    bx, bk = task // sd, task % sd
                    pred(wd, scaled, bx, bk, gb[4], tid)
                    dl[rt.D_W0], dl[rt.D_T0] = sd + bk, m
                    dl[rt.D_W1], dl[rt.D_T1] = -1, 0
                    wait(dl, counters, err, ok, tid, errh)
                    if ok[0] == 1:
                        gemmd(a, y, wd, scaled, alphad, pd, tickets, 2 * tiles, m,
                              bx, bk, tid, completed, *gb)
            end(counters, nc, done, ncta, tid, finish, tk.THREADS)
    return main


class StaticMlp(Mlp):
    def __init__(self, rows, hidden, intermediate, splits, ncta, device='cuda',
                 policy='operation', hw=1, stages=4, fused=False, selfreset=False, specialize=True, block_k=64):
        if policy != 'operation' or hw != 1 or fused or selfreset or not specialize:
            raise ValueError('arithmetic plan supports specialized operation/HW1 only')
        super().__init__(rows, hidden, intermediate, splits, ncta, device,
                         policy, hw, stages, fused, selfreset, specialize)
        self.kernel = mlp_static_kernel(hidden, intermediate, 16 if rows <= 16 else 32,
                                        *splits, ncta, rows, stages, block_k=block_k)
