# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the GDN state work (29-09-2026). No GPU, no CUDA: runs on a laptop (torch CPU) or the server with
CUDA_VISIBLE_DEVICES="" (nothing here calls a backward pass, which would initialise CUDA through gpushare).

  1. rounding: bit-level RNE fp32 -> bf16 equals torch's conversion; fp16 store saturates instead of producing inf
  2. G3 replay protocol == vLLM window protocol, bit for bit with a float32 checkpoint: chained steps with random
     window lengths (1..4 and 1..2, i.e. 3 and 1 drafts), random acceptance, padding requests, invalid slots, forced
     flushes, periodic materialize (checkpoint == vLLM's slot state_idx[r, n_acc - 1]), two ring sizes
  3. ring invariants (asserted inside the reference): per launch the replayed and written ring positions never
     overlap, meta entries agree; flush frequency matches the paper model
  4. 16-bit checkpoint: replay rounds only at flushes -> error vs fp32 smaller than the per-step 16-bit window
  5. plugin: horizon_kernels.gdn_state dispatch/coverage logic with vLLM mocked (no vLLM needed)
  6. GPU-result verifier (verify_gdn_state_results.py) accepts a complete synthetic record set, rejects broken ones
usage: python test_gdn_state_cpu.py   (exit 0 = all pass)
"""
import importlib
import json
import os
import sys
import tempfile
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gdn_state_ref as R  # noqa: E402

torch.manual_seed(0)
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def bits_diff(x, y):
    if x.dtype in (torch.bfloat16, torch.float16):
        xi, yi = x.view(torch.int16), y.view(torch.int16)
    else:
        xi, yi = x.view(torch.int32), y.view(torch.int32)
    both_zero = (x == 0) & (y == 0)
    return int(((xi != yi) & ~both_zero).sum())


# ------------------------------------------------------------------------------------------------------ 1 rounding
def test_rounding():
    x = torch.randn(200000) * torch.exp(torch.randn(200000) * 6)
    x = torch.cat([x, torch.tensor([0.0, -0.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -9, 3.0e38, -3.0e38, 1e-40, -1e-40])])
    a = R.bf16_rne_bits(x)
    b = x.to(torch.bfloat16)
    check("bf16 RNE bit trick == torch", bits_diff(a, b) == 0, f"{bits_diff(a, b)} diffs")
    y = torch.tensor([70000.0, -70000.0, 65504.0, 65519.0, 65520.0, 1e-8, 3.0])
    s = R.store_round(y, torch.float16)
    check("fp16 store saturates (no inf)", bool(torch.isfinite(s).all()) and float(s[0]) == 65504.0
          and float(s[1]) == -65504.0 and float(s[6]) == 3.0, str(s.tolist()))


# --------------------------------------------------------------------------------------------- 2/3 replay == window
def _inputs(ntoks, HV, H, gen, amp=1.0):
    L = sum(ntoks)
    CQ = 2 * H * R.DK + HV * R.DV
    qkv = torch.nn.functional.silu(torch.randn(L, CQ, generator=gen) * amp).to(torch.bfloat16)
    a = (torch.randn(L, HV, generator=gen) * 2).to(torch.bfloat16)
    b = (torch.randn(L, HV, generator=gen) * 2).to(torch.bfloat16)
    z = torch.randn(L, HV, R.DV, generator=gen).to(torch.bfloat16)
    cu = torch.zeros(len(ntoks) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(ntoks), 0)
    return qkv, a, b, z, cu


def chain(HV=6, H=2, nr=5, steps=160, T=4, half=16, sdt=torch.float32, seed=1, flush_p=0.05, mat_every=23,
          invalid_req=None):
    """run vLLM's window protocol and G3 side by side; returns (max out diff, max ckpt diff, ring stats, n checks)"""
    gen = torch.Generator().manual_seed(seed)
    rnd = torch.Generator().manual_seed(seed + 7)
    W = 4
    A_log = torch.randn(HV, generator=gen) - 1.0
    dt_bias = torch.randn(HV, generator=gen)
    norm_w = (1 + 0.1 * torch.randn(R.DV, generator=gen)).to(torch.bfloat16)
    # window protocol: W slots per request (slot 0 unused = vLLM's null block)
    NSw = nr * W + 1
    st_w = torch.zeros(NSw, HV, R.DV, R.DK, dtype=sdt)
    sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W)
    init = torch.randn(nr, HV, R.DV, R.DK, generator=gen) * 0.3
    for r in range(nr):
        st_w[sidx[r, 0]] = R.store_round(init[r], sdt)
    # G3: one checkpoint slot per request + ring
    st_g = torch.zeros(nr + 1, HV, R.DV, R.DK, dtype=sdt)
    cidx = torch.arange(1, nr + 1, dtype=torch.int32)
    ridx = torch.arange(nr, dtype=torch.int32)
    for r in range(nr):
        st_g[cidx[r]] = st_w[sidx[r, 0]]
    ring = R.Ring(nr, HV, H, half)
    if invalid_req is not None:
        cidx[invalid_req] = 0
        sidx[invalid_req, :] = 0
    nacc = torch.ones(nr, dtype=torch.int32)           # first step: state in slot 0 (vLLM), empty ring (G3)
    prev_T = [1] * nr
    worst_out, worst_ck, checks = 0, 0, 0
    for s in range(steps):
        ntoks = [int(torch.randint(1, T + 1, (1,), generator=rnd)) for _ in range(nr)]
        if s % 17 == 5:
            ntoks[s % nr] = 0                              # a padding request in the batch
        qkv, a, b, z, cu = _inputs(ntoks, HV, H, gen)
        ow = torch.full((qkv.shape[0], HV, R.DV), 7.0, dtype=torch.bfloat16)
        og = ow.clone()
        fflush = (torch.rand(nr, generator=rnd) < flush_p).to(torch.int32)
        R.window_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, st_w, z, norm_w, ow)
        R.replay_step(qkv, a, b, A_log, dt_bias, cidx, ridx, cu, nacc, fflush, st_g, ring, z, norm_w, og)
        worst_out = max(worst_out, bits_diff(ow, og))
        checks += 1
        # next acceptance: 1..nt of this window (a request with 0 tokens keeps its acceptance and pending records)
        new = nacc.clone()
        for r in range(nr):
            if ntoks[r] > 0:
                new[r] = int(torch.randint(1, ntoks[r] + 1, (1,), generator=rnd))
                prev_T[r] = ntoks[r]
        nacc = new
        if mat_every and s % mat_every == mat_every - 1 and sdt == torch.float32:
            R.materialize(cidx, ridx, nacc, st_g, ring)
            for r in range(nr):
                if r == invalid_req:
                    continue
                vslot = int(sidx[r, int(nacc[r]) - 1])
                worst_ck = max(worst_ck, bits_diff(st_w[vslot], st_g[cidx[r]]))
            # after a materialize the ring is empty: the next step must not re-apply the pending records.
            # vLLM's next source is slot n_acc - 1; G3's checkpoint already holds it -> tell G3 "nothing pending"
            # (meta npend = 0 makes n_acc irrelevant), exactly what the plugin does after a materialize.
    return worst_out, worst_ck, dict(ring.stats), checks


def test_replay_equals_window():
    for (T, half, nr, steps, seed) in ((4, 16, 5, 160, 1), (2, 16, 5, 160, 2), (4, 32, 3, 120, 3),
                                      (4, 8, 4, 120, 4)):
        wo, wc, stats, n = chain(T=T, half=half, nr=nr, steps=steps, seed=seed)
        check(f"G3 fp32 == vLLM window protocol (T<={T}, HALF={half}, {nr} req, {steps} steps)", wo == 0 and wc == 0,
              f"out diffs {wo}, ckpt diffs {wc}, flushes {stats['flushes']}/{stats['steps']} req-steps, "
              f"replayed {stats['replayed']}")
    wo, wc, stats, n = chain(T=4, half=16, nr=4, steps=80, seed=9, invalid_req=2)
    check("G3 fp32 with an invalid request (slot 0) == vLLM", wo == 0 and wc == 0, f"{wo}/{wc}")
    # production geometry, a few steps (48 value heads, 16 key heads)
    wo, wc, stats, n = chain(HV=48, H=16, nr=2, steps=12, T=4, half=16, seed=5, mat_every=5)
    check("G3 fp32 == vLLM window, production geometry HV48/H16", wo == 0 and wc == 0, f"{wo}/{wc}")


def test_flush_model():
    wo, wc, stats, n = chain(T=4, half=16, nr=6, steps=300, seed=11, flush_p=0.0, mat_every=0)
    per = stats["flushes"] / max(1, stats["steps"])
    # acceptance uniform in 1..nt, nt uniform in 1..4 -> mean acc 1.75: flush about every (16 - 4..) / 1.75 steps
    check("flush rate in the expected range (HALF 16, mean acc ~1.75)", 0.08 < per < 0.25, f"{per:.3f} per req-step")
    tb = R.traffic_bytes(32, 4, acc_mean=3.4)
    tb2 = R.traffic_bytes(32, 2, acc_mean=1.6)
    print("  traffic model 32 req, T=4: window %.0f MB, G3 %.0f MB; T=2: window %.0f MB, G3 %.0f MB"
          % (tb["window"] / 1e6, tb["g3"] / 1e6, tb2["window"] / 1e6, tb2["g3"] / 1e6))
    check("traffic model: G3 < 45% of window at T=4, < 60% at T=2",
          tb["g3"] < 0.45 * tb["window"] and tb2["g3"] < 0.6 * tb2["window"])


# ----------------------------------------------------------------------------------------------- 4 16-bit checkpoint
def err_vs_fp32(sdt, protocol, steps=200, seed=21):
    gen = torch.Generator().manual_seed(seed)
    rnd = torch.Generator().manual_seed(seed + 1)
    HV, H, nr = 6, 2, 3
    A_log = torch.randn(HV, generator=gen) - 3.0          # slow decay heads (worst case for rounding)
    dt_bias = torch.randn(HV, generator=gen) - 2.0
    norm_w = torch.ones(R.DV).to(torch.bfloat16)
    init = torch.randn(nr, HV, R.DV, R.DK, generator=gen) * 0.3
    res = {}
    for dt in (torch.float32, sdt):
        g2 = torch.Generator().manual_seed(seed + 5)
        r2 = torch.Generator().manual_seed(seed + 6)
        if protocol == "window":
            st = torch.zeros(nr * 4 + 1, HV, R.DV, R.DK, dtype=dt)
            sidx = (torch.arange(nr * 4, dtype=torch.int32) + 1).view(nr, 4)
            for r in range(nr):
                st[sidx[r, 0]] = R.store_round(init[r], dt)
        else:
            st = torch.zeros(nr + 1, HV, R.DV, R.DK, dtype=dt)
            cidx = torch.arange(1, nr + 1, dtype=torch.int32)
            ridx = torch.arange(nr, dtype=torch.int32)
            for r in range(nr):
                st[cidx[r]] = R.store_round(init[r], dt)
            ring = R.Ring(nr, HV, H, 16)
        nacc = torch.ones(nr, dtype=torch.int32)
        outs = []
        for s in range(steps):
            ntoks = [4] * nr
            qkv, a, b, z, cu = _inputs(ntoks, HV, H, g2)
            o = torch.empty(qkv.shape[0], HV, R.DV, dtype=torch.bfloat16)
            if protocol == "window":
                R.window_step(qkv, a, b, A_log, dt_bias, sidx, cu, nacc, st, z, norm_w, o)
            else:
                R.replay_step(qkv, a, b, A_log, dt_bias, cidx, ridx, cu, nacc, torch.zeros(nr, dtype=torch.int32),
                              st, ring, z, norm_w, o)
            outs.append(o.float())
            nacc = torch.randint(1, 5, (nr,), generator=r2, dtype=torch.int32)
        res[dt] = torch.stack(outs)
    ref, x = res[torch.float32], res[sdt]
    return float(((x - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).median())


def test_16bit_checkpoint():
    for sdt in (torch.bfloat16, torch.float16):
        ew = err_vs_fp32(sdt, "window")
        er = err_vs_fp32(sdt, "replay")
        check(f"{str(sdt)[6:]}: replay checkpoint error < per-step window error", er < ew,
              f"median rel err window {ew:.2e}, replay {er:.2e}")


# ---------------------------------------------------------------------------------------------------- 5 plugin
def _mock_vllm(tmp):
    """a minimal fake vllm package: _custom_ops.fused_gdn_decode_post_conv_mtp, logger, the qwen GDN module globals"""
    root = os.path.join(tmp, "vllm")
    os.makedirs(os.path.join(root, "model_executor/layers/mamba/gdn"), exist_ok=True)
    for d in ("", "model_executor", "model_executor/layers", "model_executor/layers/mamba",
              "model_executor/layers/mamba/gdn"):
        open(os.path.join(root, d, "__init__.py"), "a").close()
    open(os.path.join(root, "logger.py"), "w").write(
        "import logging\n"
        "class _L(logging.Logger):\n"
        "    def warning_once(self, *a, **k): self.warning(*a)\n"
        "    def info_once(self, *a, **k): self.info(*a)\n"
        "logging.setLoggerClass(_L)\n"
        "def init_logger(n): return logging.getLogger(n)\n")
    open(os.path.join(root, "_custom_ops.py"), "w").write(
        "CALLS = []\n"
        "def fused_gdn_decode_post_conv_mtp(*a, **k):\n"
        "    CALLS.append(('vllm', a, k))\n"
        "    return k.get('out')\n")
    open(os.path.join(root, "model_executor/layers/mamba/mamba_utils.py"), "w").write(
        "import torch\n"
        "class MambaStateDtypeCalculator:\n"
        "    @classmethod\n"
        "    def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype='auto'):\n"
        "        return (torch.bfloat16, torch.float32)\n")
    open(os.path.join(root, "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"), "w").write(
        "import torch\nFUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)\n")
    return tmp


def test_plugin():
    repo = os.path.abspath(os.path.join(HERE, "..", "..", ".."))       # server/
    with tempfile.TemporaryDirectory() as tmp:
        _mock_vllm(tmp)
        sys.path.insert(0, tmp)
        sys.path.insert(0, repo)
        # fake tilelang kernel module: records calls
        tl = types.ModuleType("tilelang")
        contrib = types.ModuleType("tilelang.contrib")
        hops = types.ModuleType("tilelang.contrib.horizon_ops")
        gs = types.ModuleType("tilelang.contrib.horizon_ops.gdn_state")
        gs.CALLS = []
        gs.T_MAX = 8
        gs.supported = lambda mixed, state, gate, a, b: state.dtype in (torch.float32, torch.bfloat16, torch.float16)
        gs.fused_gdn_decode_post_conv_mtp = lambda *a, **k: gs.CALLS.append(("ours", a, k)) or k.get("out")
        gs.warm = None
        hops.gdn_state = gs
        sys.modules.update({"tilelang": tl, "tilelang.contrib": contrib, "tilelang.contrib.horizon_ops": hops,
                            "tilelang.contrib.horizon_ops.gdn_state": gs})
        for m in [m for m in list(sys.modules) if m.startswith("horizon_kernels") or m.startswith("vllm")]:
            del sys.modules[m]
        try:
            os.environ["HORIZON_GDN_STATE"] = "fp16"
            os.environ["HORIZON_GDN2"] = "1"
            P = importlib.import_module("horizon_kernels.gdn_state")
            P._WARM_ALWAYS_OK = True            # the fake kernel needs no compile / warm-up
            P.install()
            from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C
            from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
            from vllm import _custom_ops as ops
            check("plugin: GDN ssm state dtype -> float16",
                  C.gated_delta_net_state_dtype(torch.bfloat16, "auto", "auto") == (torch.bfloat16, torch.float16))
            check("plugin: float16 added to vLLM's fused-decode state dtypes", torch.float16 in Q.FUSED_GDN_STATE_DTYPES)
            check("plugin: op wrapped and marked", getattr(ops.fused_gdn_decode_post_conv_mtp, "_horizon_gdn2", False))
            HV, H = 48, 16
            L = 8
            mixed = torch.zeros(L, 2 * H * 128 + HV * 128, dtype=torch.bfloat16)
            ba = torch.zeros(L, 2 * HV, dtype=torch.bfloat16)
            b, a = ba[:, :HV], ba[:, HV:]
            gate = torch.zeros(L, HV * 128 * 2, dtype=torch.bfloat16)[:, HV * 128:].view(L, HV, 128)
            out = torch.zeros(L, HV, 128, dtype=torch.bfloat16)
            kw = dict(mixed_qkv=mixed, a=a, b=b, A_log=torch.zeros(HV), dt_bias=torch.zeros(HV),
                      state_indices=torch.zeros(2, 4, dtype=torch.int32), cu_seqlens=torch.tensor([0, 4, 8]),
                      num_accepted_tokens=torch.ones(2, dtype=torch.int32), output_gate=gate,
                      norm_weight=torch.ones(128), out=out)
            for dt in (torch.float32, torch.bfloat16, torch.float16):
                st = torch.zeros(3, HV, 128, 128, dtype=dt)
                gs.CALLS.clear(); sys.modules["vllm._custom_ops"].CALLS.clear()
                ops.fused_gdn_decode_post_conv_mtp(state=st, **kw)
                check(f"plugin: {str(dt)[6:]} state -> our kernel", len(gs.CALLS) == 1
                      and not sys.modules["vllm._custom_ops"].CALLS)
            # uncovered call (9 draft slots) with an fp32 state -> vLLM's op; with fp16 -> hard error (vLLM has none)
            kw2 = dict(kw, state_indices=torch.zeros(2, 9, dtype=torch.int32))
            gs.CALLS.clear(); sys.modules["vllm._custom_ops"].CALLS.clear()
            ops.fused_gdn_decode_post_conv_mtp(state=torch.zeros(3, HV, 128, 128), **kw2)
            check("plugin: uncovered fp32 call falls back to vLLM", len(sys.modules["vllm._custom_ops"].CALLS) == 1)
            raised = False
            try:
                ops.fused_gdn_decode_post_conv_mtp(state=torch.zeros(3, HV, 128, 128, dtype=torch.float16), **kw2)
            except RuntimeError:
                raised = True
            check("plugin: uncovered fp16 call raises (no silent wrong kernel)", raised)
            # defaults: no switch -> nothing changes
            for m in [m for m in list(sys.modules) if m.startswith("horizon_kernels") or m.startswith("vllm")]:
                del sys.modules[m]
            os.environ.pop("HORIZON_GDN_STATE"); os.environ.pop("HORIZON_GDN2")
            P = importlib.import_module("horizon_kernels.gdn_state")
            P.install()
            from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator as C2
            from vllm import _custom_ops as ops2
            check("plugin: switches unset -> vLLM untouched",
                  C2.gated_delta_net_state_dtype(torch.bfloat16, "auto", "auto") == (torch.bfloat16, torch.float32)
                  and not hasattr(ops2.fused_gdn_decode_post_conv_mtp, "_horizon_gdn2"))
            os.environ["HORIZON_GDN_STATE"] = "int8"
            for m in [m for m in list(sys.modules) if m.startswith("horizon_kernels") or m.startswith("vllm")]:
                del sys.modules[m]
            P = importlib.import_module("horizon_kernels.gdn_state")
            bad = False
            try:
                P.install()
            except ValueError:
                bad = True
            check("plugin: unknown HORIZON_GDN_STATE value refused", bad)
            os.environ.pop("HORIZON_GDN_STATE")
        finally:
            for m in [m for m in list(sys.modules) if m.startswith("horizon_kernels") or m.startswith("vllm")
                      or m.startswith("tilelang")]:
                del sys.modules[m]
            sys.path.remove(tmp)
            sys.path.remove(repo)


# ------------------------------------------------------------------------------------------- 6 GPU result verifier
def test_verifier():
    import verify_gdn_state_results as V
    good = V.synthetic_complete()
    ok, why = V.verify(good)
    check("verifier accepts a complete synthetic record set", ok, why)
    bad_cases = {
        "window mismatch": lambda rs: [dict(r, out_diff=3) if r.get("test") == "window" else r for r in rs],
        "missing speed row": lambda rs: [r for r in rs if not (r.get("test") == "bench" and r.get("nr") == 32
                                                                and r.get("ql") == 4)],
        "replay ckpt diff": lambda rs: [dict(r, ckpt_diff=1) if r.get("test") == "replay_chain" else r for r in rs],
        "graph fail": lambda rs: [dict(r, graph_replay_equal=False) if r.get("test") == "graph" else r for r in rs],
        "nan timing": lambda rs: [dict(r, us={k: float("nan") for k in r["us"]}) if r.get("test") == "bench" else r
                                  for r in rs],
        "no terminal": lambda rs: [r for r in rs if r.get("test") != "end"],
    }
    for name, f in bad_cases.items():
        ok, why = V.verify(f(json.loads(json.dumps(good))))
        check(f"verifier rejects: {name}", not ok, why)


if __name__ == "__main__":
    test_rounding()
    test_replay_equals_window()
    test_flush_model()
    test_16bit_checkpoint()
    test_plugin()
    test_verifier()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
