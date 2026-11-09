# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Experimental persistent gate/up -> SiLU -> down, separate from model hooks.

One bounded kernel, canonical existing GEMM split arithmetic and SiLU body.
No normalization, attention, state cache or model-specific dimensions are embedded.
The caller must synchronize and reject a failed launch; no serving integration yet.
"""
import tilelang
import tilelang.language as T
from tilelang.contrib.horizon_k4 import runtime as rt, tasks as tk
from tilelang.contrib.horizon_k4.mlp_native_tasks import gemm_native_task_body
from tilelang.contrib.horizon_k4.mlp_pair_tasks import gemm_pair_task_body, gemm_pair_alloc
from tilelang.contrib.horizon_k4.mlp_schedule import GU, SILU, DOWN, build, build_fused, validate

SILU_SRC = r"""
#ifndef HZ_SILU_SRC_DEFINED
#define HZ_SILU_SRC_DEFINED
__device__ __forceinline__ float hz_silu_mul_ind(float a, float u) {
  float e = expf(-a);
  float d = 1.0f + e;
  float s;
  asm("div.full.f32 %0, %1, %2;" : "=f"(s) : "f"(a), "f"(d));
  return s * u;
}
#endif
"""


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mlp_kernel(hidden, intermediate, wm, sg, sd, ncta, stages=4, hw=0,
               timeout_ns=2_000_000_000, fused=False, selfreset=False, rows_hint=0):
    validate(1, hidden, intermediate, (sg, sd), ncta, 'operation')
    if rows_hint and (type(rows_hint) is not int or not 1 <= rows_hint <= 32):
        raise ValueError('invalid specialized batch size')
    if fused not in (0, 1, 2, 3, 4, 5):
        raise ValueError('unknown MLP fusion path')
    if selfreset and (sg != 1 or 2 * sd > 2 * intermediate // 128):
        raise ValueError('slice reset requires single-split GU; ticket arena must be free')
    if fused and sg != 1:
        raise ValueError('paired activation currently requires unsplit gate/up')
    if fused >= 2 and hw:
        raise ValueError('paired-wide GU currently uses the TileLang pipeline only')
    if hw not in (0, 1) or wm not in (16, 32) or stages < 3:
        raise ValueError('unsupported MLP pipeline configuration')
    if hw:
        srcg, preg, gemmg = tk.gemm_task_hw('mlpg', 2 * intermediate, hidden, wm, sg, stages)
        srcd, pred, gemmd = tk.gemm_task_hw('mlpd', hidden, intermediate, wm, sd, stages)
        src = tk.hw_common_src() + srcg + srcd
    else:
        gemmg = tk.gemm_task_body(2 * intermediate, hidden, wm, sg, stages)
        gemmd = tk.gemm_task_body(hidden, intermediate, wm, sd, stages)
        src = tk.gemm_src()
        if fused in (4, 5):
            gemmg = gemm_native_task_body(2 * intermediate, hidden, wm, sg, 6 if wm == 16 else 4, 8, paired=True, prepacked=fused == 5)
            gemmd = gemm_native_task_body(hidden, intermediate, wm, sd, 4, 4)
        elif fused >= 2:
            gemmg = gemm_pair_task_body(intermediate, hidden, wm, stages, direct=fused == 3)
    src = rt.rt_src() + SILU_SRC + src
    if selfreset:
        src += r"""
#ifndef HZ_MLP_SLICE_RESET
#define HZ_MLP_SLICE_RESET
__device__ __forceinline__ void hz_mlp_consume(int *count, int expected, int *ready) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(old) : "l"(count) : "memory");
  if (old == expected - 1) {
    asm volatile("st.release.gpu.global.b32 [%0], 0;" :: "l"(ready) : "memory");
    asm volatile("st.release.gpu.global.b32 [%0], 0;" :: "l"(count) : "memory");
  }
}
#endif
"""
    activate = tk.silu_task_body(intermediate, 128 if fused else intermediate // sd, 1)
    wait, signal, end = rt.macros(timeout_ns)
    m, maxt, nc = (rows_hint if rows_hint else T.dynamic('M')), T.dynamic('MAXT'), T.dynamic('NC')
    f32, bf16, i32 = T.float32, T.bfloat16, T.int32
    ntick = (2 * intermediate + hidden) // 128

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
        with T.Kernel(ncta, threads=256 if fused in (4, 5) else tk.THREADS, prelude=src) as cta:
            tid = T.get_thread_binding()
            ok = T.alloc_shared((1,), i32)
            completed = T.alloc_shared((1,), i32)
            finish = T.alloc_shared((1,), i32)
            dl = T.alloc_local((rt.DESC,), i32)
            if hw:
                gb = tk.gemm_hw_alloc(wm, stages)
                tk.gemm_hw_addr(gb[0], gb[1], gb[2], gb[4])
            else:
                if fused >= 2:
                    gb = gemm_pair_alloc(wm)
                else:
                    gb = tk.gemm_alloc(wm)
            if fused in (4, 5):
                # Different stage depths need distinct shared arenas in TileLang.
                db = tk.gemm_alloc(wm)
            else:
                db = gb
            for t in T.serial(ntask[cta]):
                rt.load_desc(desc, cta, t, dl)
                if hw:
                    if dl[rt.D_OP] == GU:
                        preg(wg, scaleg, dl[rt.D_A0], dl[rt.D_A1], gb[4], tid)
                    elif dl[rt.D_OP] == DOWN:
                        pred(wd, scaled, dl[rt.D_A0], dl[rt.D_A1], gb[4], tid)
                # GU has no predecessors. Its body initializes completion.
                if not selfreset or dl[rt.D_OP] != GU:
                    wait(dl, counters, err, ok, tid, errh)
                    if tid == 0:
                        completed[0] = 1
                if (selfreset and dl[rt.D_OP] == GU) or ok[0] == 1:
                    if dl[rt.D_OP] == GU:
                        if fused in (2, 3):
                            gemmg(x, gu, wg, scaleg, alphag, pg, tickets, 0, m,
                                  dl[rt.D_A0], dl[rt.D_A1], tid, completed, *gb, a)
                        else:
                            gemmg(x, gu, wg, scaleg, alphag, pg, tickets, 0, m,
                                  dl[rt.D_A0], dl[rt.D_A1], tid, completed, *gb)
                        if fused:
                            # One CTA owns both producer tiles; the second GEMM
                            # completes before its activation tile is published.
                            if fused == 1:
                                if hw:
                                    preg(wg, scaleg, dl[rt.D_A0] + intermediate // 128, 0, gb[4], tid)
                                gemmg(x, gu, wg, scaleg, alphag, pg, tickets, 0, m,
                                      dl[rt.D_A0] + intermediate // 128, 0, tid, completed, *gb)
                            if fused in (2, 4, 5):
                                # Elementwise rows need no intermediate CTA barriers.
                                for row, col in T.Parallel(m, 128):
                                    c = dl[rt.D_A0] * 128 + col
                                    a[row, c] = T.cast(T.call_extern(f32, "hz_silu_mul_ind",
                                        T.cast(gu[row, c], f32), T.cast(gu[row, intermediate + c], f32)), bf16)
                            elif fused == 1:
                                for row in T.serial(m):
                                    activate(gu, a, row, dl[rt.D_A0] * 128, tid)
                    elif dl[rt.D_OP] == SILU:
                        activate(gu, a, dl[rt.D_A0], dl[rt.D_A1], tid)
                    elif dl[rt.D_OP] == DOWN:
                        gemmd(a, y, wd, scaled, alphad, pd, tickets, 2 * intermediate // 128, m,
                              dl[rt.D_A0], dl[rt.D_A1], tid, completed, *db)
                if selfreset:
                    if dl[rt.D_OP] == GU:
                        # Fused activation writes follow the GEMM's own barrier.
                        if fused:
                            T.sync_threads()
                        if tid == 0 and sigtab[cta, t] >= 0:
                            T.call_extern("handle", "hz_signal", T.access_ptr(counters[sigtab[cta, t]], "rw"))
                    elif dl[rt.D_OP] == SILU:
                        # silu_task_body already ends with a CTA barrier.
                        if tid == 0 and ok[0] == 1:
                            if sigtab[cta, t] >= 0:
                                T.call_extern("handle", "hz_signal", T.access_ptr(counters[sigtab[cta, t]], "rw"))
                            T.call_extern("handle", "hz_mlp_consume", T.access_ptr(tickets[sd + dl[rt.D_W0]], "rw"),
                                          m, T.access_ptr(counters[dl[rt.D_W0]], "rw"))
                    elif dl[rt.D_OP] == DOWN:
                        # Every split consumes one A slice, even when it did not
                        # finalize its output tile. The final consumer resets it.
                        if tid == 0 and ok[0] == 1:
                            T.call_extern("handle", "hz_mlp_consume", T.access_ptr(tickets[dl[rt.D_A1]], "rw"),
                                          hidden // 128, T.access_ptr(counters[dl[rt.D_W0]], "rw"))
                else:
                    signal(sigtab[cta, t], counters, tid, ok[0] == 1 and completed[0] == 1)
            if not selfreset:
                end(counters, nc, done, ncta, tid, finish, 256 if fused in (4, 5) else tk.THREADS)
    return main


class Mlp:
    def __init__(self, rows, hidden, intermediate, splits, ncta, device='cuda',
                 policy='operation', hw=0, stages=4, fused=False, selfreset=False, specialize=False):
        import torch
        validate(rows, hidden, intermediate, splits, ncta, policy)
        self.rows, self.hidden, self.intermediate = rows, hidden, intermediate
        self.fused = fused
        self.splits = tuple(splits)
        self.kernel = mlp_kernel(hidden, intermediate, 16 if rows <= 16 else 32,
                                 *splits, ncta, stages, hw, fused=fused, selfreset=selfreset, rows_hint=rows if specialize else 0)
        self.schedule = (build_fused if fused else build)(rt.Schedule, rows, hidden, intermediate, splits, ncta, policy)
        self.ts = self.schedule.tensors(device)
        # Device flag is sufficient for this standalone synchronized benchmark;
        # production integration would additionally require the mapped host guard.
        self.errh = torch.zeros(1, dtype=torch.int32, device=device)
        make = lambda *shape, dtype=torch.bfloat16: torch.empty(*shape, dtype=dtype, device=device)
        sg, sd = splits
        self.gu, self.a, self.y = make(rows, 2 * intermediate), make(rows, intermediate), make(rows, hidden)
        self.pg, self.pd = make(sg, rows, 2 * intermediate, dtype=torch.float32), make(sd, rows, hidden, dtype=torch.float32)
        self.tickets = torch.zeros((2 * intermediate + hidden) // 128, dtype=torch.int32, device=device)

    def __call__(self, x, gate_up, down):
        import torch
        if tuple(x.shape) != (self.rows, self.hidden) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError('MLP requires the declared contiguous BF16 input')
        layout = getattr(gate_up, '_k4_layout', None)
        if (self.fused == 5) != (layout == 'gate128-up128'):
            raise ValueError('GU packed-layout capability mismatch')
        t = self.ts
        self.kernel(t['desc'], t['sigtab'], t['ntask'], t['counters'], t['done'], t['err'], self.errh, x,
                    gate_up.Wg, gate_up.Sg, gate_up.alpha, down.Wg, down.Sg, down.alpha,
                    self.gu, self.a, self.y, self.pg, self.pd, self.tickets)
        return self.y

    def check(self):
        """After synchronization only: failed launches and dirty replay state are errors."""
        import torch
        if any(torch.count_nonzero(t).item() for t in
               (self.ts['err'], self.errh, self.ts['done'], self.ts['counters'], self.tickets)):
            raise RuntimeError('MLP timeout or replay state did not reset; outputs invalid')
