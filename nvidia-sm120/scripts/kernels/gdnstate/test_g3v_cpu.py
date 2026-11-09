# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the G3v protocol on vLLM 0.30's layout (30-09-2026), torch reference only (g3v_ref.py). No GPU.

World A = vLLM 0.30 as is (window kernel writes a state per token into the window slots; align pre/post copies;
mixed batches through the FLA path). World B = the same machinery with G3v + the plugin hooks (expand before a
crossing / non-pure batch / post-copy, meta reset). Identical inputs and events; required bit for bit:
  every step's outputs; every state vLLM reads through a copy (pre-copy / post-copy sources); every completed
  block (prefix-cache content); each request's committed state slot[n_acc - 1] at the end (after a final expand).
Covered: QL 2 <-> 4 switching (the 3/1 draft schedule), random acceptance, small blocks (many boundary crossings and
post-copies), random mixed (non-pure) batches, a request whose window runs past its block table (null ring slot ->
window mode), request-slot reuse with a stale ring (meta reset at add_request), production geometry.
usage: python test_g3v_cpu.py   (exit 0 = all pass)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g3v_ref as V  # noqa: E402
import gdn_state_ref as R  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def bits(x, y):
    xi = x.contiguous().view(torch.int32 if x.dtype == torch.float32 else torch.int16)
    yi = y.contiguous().view(torch.int32 if y.dtype == torch.float32 else torch.int16)
    return int(((xi != yi) & ~((x == 0) & (y == 0))).sum())


def inputs(ntoks, HV, H, gen):
    L = sum(ntoks)
    CQ = 2 * H * 128 + HV * 128
    qkv = torch.nn.functional.silu(torch.randn(L, CQ, generator=gen)).to(torch.bfloat16)
    a = (torch.randn(L, HV, generator=gen) * 2).to(torch.bfloat16)
    b = (torch.randn(L, HV, generator=gen) * 2).to(torch.bfloat16)
    z = torch.randn(L, HV, 128, generator=gen).to(torch.bfloat16)
    cu = torch.zeros(len(ntoks) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(ntoks), 0)
    return qkv, a, b, z, cu


def run(nr=4, HV=6, H=2, B=7, steps=120, seed=1, mixed_p=0.15, short_table=None, RF=None, reuse_at=None):
    gen = torch.Generator().manual_seed(seed)
    rnd = torch.Generator().manual_seed(seed + 99)
    RF = RF or 3 * V.rec_floats(HV, H) + 5                    # PPS 3, HALF 4: the production geometry
    A_log = torch.randn(HV, generator=gen) - 1.0
    dt_bias = torch.randn(HV, generator=gen)
    norm_w = (1 + 0.1 * torch.randn(128, generator=gen)).to(torch.bfloat16)
    init = torch.randn(nr, HV, 128, 128, generator=gen) * 0.3
    npr = (steps * 4) // B + 16
    WA = V.World(nr, HV, H, B, RF, False, seed, nblocks_per_req=npr, init=init)
    WB = V.World(nr, HV, H, B, RF, True, seed, nblocks_per_req=npr, init=init)
    if short_table is not None:                               # this request's block table ends early
        for W in (WA, WB):
            W.bt[short_table] = W.bt[short_table][:W.col[short_table] + 3]
    out_bad, obs_bad, steps_run = 0, 0, 0
    for s in range(steps):
        T = 4 if torch.rand(1, generator=rnd).item() < 0.5 else 2
        Ts = [T] * nr
        mixed = torch.rand(1, generator=rnd).item() < mixed_p
        if short_table is not None and WA.computed[short_table] + T + B >= len(WA.bt[short_table]) * B:
            break
        if reuse_at is not None and s == reuse_at:
            # request 0 finishes; a new request takes its request-state index (stale ring in world B) and a fresh
            # block table; vLLM: add_request -> num_accepted = 1; plugin: meta reset
            for W in (WA, WB):
                W.bt[0] = list(range(1 + nr * npr, 1 + (nr + 1) * npr)) if W.st.S.shape[0] > 1 + nr * npr else W.bt[0]
                W.computed[0] = 9
                W.col[0] = (9 - 1) // B
                W.acc[0] = 1
                W.st.S[W.bt[0][W.col[0]]] = init[1]
            WB.meta[WB.req_idx[0]] = 0
        na, nb = len(WA.observed), len(WB.observed)
        WA.preprocess(Ts, mixed)
        WB.preprocess(Ts, mixed)
        inp = inputs(Ts, HV, H, gen)
        oa = WA.forward(Ts, inp, A_log, dt_bias, norm_w, mixed)
        ob = WB.forward(Ts, inp, A_log, dt_bias, norm_w, mixed)
        out_bad += bits(oa, ob)
        acc = [int(torch.randint(1, T + 1, (1,), generator=rnd)) for _ in range(nr)]
        WA.postprocess(acc)
        WB.postprocess(acc)
        for (ea, eb) in zip(WA.observed[na:], WB.observed[nb:]):
            assert ea[:3] == eb[:3]
            obs_bad += bits(ea[3], eb[3])          # the state vLLM copied, as it was at copy time
        if len(WA.observed) != len(WB.observed):
            obs_bad += 1
        steps_run += 1
    # final: expand everything in B, then committed states and every completed block
    WB._expand(list(range(nr)), WB.col)
    fin_bad = sum(bits(WA.committed(i), WB.committed(i)) for i in range(nr))
    blk_bad = sum(bits(WA.st.S[WA.bt[i][j]], WB.st.S[WB.bt[i][j]]) for i in range(nr) for j in range(WA.col[i]))
    return out_bad, obs_bad, fin_bad, blk_bad, WB.stats, steps_run, len(WA.observed)


def main():
    for kw in (dict(seed=1), dict(seed=2, B=5, mixed_p=0.3), dict(seed=3, B=11, mixed_p=0.0, steps=150),
               dict(seed=4, nr=3, B=3, mixed_p=0.1)):
        o, ob, f, bk, st, n, nobs = run(**kw)
        check(f"G3v == vLLM 0.30 align machinery {kw}", o == 0 and ob == 0 and f == 0 and bk == 0,
              f"out {o}, copy sources {ob} ({nobs} copies), committed {f}, completed blocks {bk}; {n} steps, "
              f"flushes {st['flushes']}/{st['steps']}, replayed {st['replayed']}, expanded {st['expanded']}, "
              f"window-mode rows {st['window_mode']}")
    o, ob, f, bk, st, n, nobs = run(seed=5, short_table=2, steps=60, B=6)
    check("request whose window runs past its block table -> window mode, still exact",
          o == 0 and ob == 0 and f == 0 and bk == 0 and st["window_mode"] > 0,
          f"window-mode rows {st['window_mode']}, out {o}, committed {f}")
    o, ob, f, bk, st, n, nobs = run(seed=6, reuse_at=40, steps=90)
    check("request-state index reused with a stale ring (meta reset at add_request)",
          o == 0 and ob == 0 and f == 0 and bk == 0, f"out {o}, committed {f}")
    o, ob, f, bk, st, n, nobs = run(nr=2, HV=48, H=16, B=5, steps=10, seed=7, RF=30720, mixed_p=0.2)
    check("production geometry HV48/H16, conv region 30,720 floats (PPS 3, HALF 4)",
          o == 0 and ob == 0 and f == 0 and bk == 0, f"out {o}, committed {f}, flushes {st['flushes']}")
    pps, half = V.ring_geometry(48, 16, 30720, 4)
    check("production ring geometry", (pps, half) == (3, 4), f"PPS {pps}, HALF {half}")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
