# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K4 runtime self-test kernel (phase 2a): a persistent 188-CTA kernel running a random task DAG through the runtime's
counters, checking every value it reads.

Task = one slot of B (NT, W) int32. Task s writes B[s, j] = seq * 7919 + s * W + j after waiting for its (up to two)
source slots, whose values it verifies element by element (mismatches are counted in bad[0]). Slots are small
(W = 8 -> 4 slots per 128-byte line), so a consumer that reads through L1 caches a line holding neighbouring slots that
other SMs write later: with plain loads (rd = 0) stale data is expected, with L2 loads (rd = 1: ld.global.cg) it must
never happen. `seq` changes every launch, so data left over from a previous launch is caught too.
"""

import tilelang
import tilelang.language as T

from tilelang.contrib.horizon_k4 import runtime as rt


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def dag_kernel(ncta: int, W: int, rd: int = 1, threads: int = 128, timeout_ns: int = 2_000_000_000,
               sleep_ns: int = 0, fence: int = 0):
    NT = T.dynamic("NT")
    MAXT = T.dynamic("MAXT")
    NC = T.dynamic("NC")
    i32 = T.int32
    task_wait, task_signal, grid_end = rt.macros(timeout_ns)

    @T.prim_func
    def main(desc: T.Tensor((ncta, MAXT, rt.DESC), i32), sigtab: T.Tensor((ncta, MAXT), i32),
             ntask: T.Tensor((ncta,), i32), counters: T.Tensor((NC,), i32), done: T.Tensor((1,), i32),
             err: T.Tensor((1,), i32), B: T.Tensor((NT, W), i32), bad: T.Tensor((1,), i32), seq: i32):
        with T.Kernel(ncta, threads=threads, prelude=rt.rt_src(sleep_ns, bool(fence))) as cta:
            tid = T.get_thread_binding()
            ok_s = T.alloc_shared((1,), i32)
            tk_s = T.alloc_shared((1,), i32)
            nbad = T.alloc_local((1,), i32)
            dl = T.alloc_local((rt.DESC,), i32)
            nbad[0] = 0
            for t in T.serial(ntask[cta]):
                rt.load_desc(desc, cta, t, dl)
                task_wait(dl, counters, err, ok_s, tid, err)
                if ok_s[0] == 1:
                    s = dl[rt.D_A0]
                    for src_i in T.unroll(2):
                        src = dl[rt.D_A1 + src_i]
                        if src >= 0:
                            for jj in T.serial(T.ceildiv(W, threads)):
                                j = jj * threads + tid
                                if j < W:
                                    if rd == 1:
                                        v = T.call_extern(i32, "hz_ld_cg", T.access_ptr(B[src, j], "r"))
                                        if v != seq * 7919 + src * W + j:
                                            nbad[0] = nbad[0] + 1
                                    else:
                                        if B[src, j] != seq * 7919 + src * W + j:
                                            nbad[0] = nbad[0] + 1
                    for jj in T.serial(T.ceildiv(W, threads)):
                        j = jj * threads + tid
                        if j < W:
                            B[s, j] = seq * 7919 + s * W + j
                task_signal(sigtab[cta, t], counters, tid, True)
            if nbad[0] > 0:
                T.atomic_add(bad[0], nbad[0])
            grid_end(counters, NC, done, ncta, tid, tk_s, threads)

    return main


def build_dag(nt: int, ncta: int = rt.NCTA_DEFAULT, window: int = 2048, chain: bool = False, seed: int = 0,
              unsatisfiable: bool = False):
    """Schedule for nt slot tasks; slot i waits for up to two random earlier slots within `window` (chain: i - 1)."""
    import random
    r = random.Random(seed)
    sc = rt.Schedule(ncta)
    c0 = sc.counters(nt)
    tasks = []
    for i in range(nt):
        if chain:
            srcs = [i - 1] if i > 0 else []
        else:
            k = 0 if i == 0 else r.choice((0, 1, 2, 2))
            srcs = sorted({r.randrange(max(0, i - window), i) for _ in range(k)}) if k else []
        waits = [(c0 + s, 1) for s in srcs]
        a = srcs + [-1] * (2 - len(srcs))
        tasks.append((i, a[0], a[1], waits, c0 + i))
    if unsatisfiable:            # one task waits for 2 signals on a counter that gets only one: must time out
        i, a1, a2, waits, sig = tasks[-1]
        tasks[-1] = (i, 0, -1, [(c0 + 0, 2)], sig)
        sc.targets[c0] = 2       # bypass check(): the test wants this
    sc.add(0, tasks)
    return sc
