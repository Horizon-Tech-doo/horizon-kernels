# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for G3v-sel (30-09-2026 evening): the one-launch selecting expand (gdn_g3v.expand_sel) through the REAL
plugin hooks (horizon_kernels.gdn_g3 installed on vLLM 0.30's MambaHybridModelState; vLLM's own pre/post/prepare
methods replaced by no-ops, so only the G3v work runs). MEASURE ONLY.

usage: python test_g3v_sel_gpu.py OUT.jsonl --pkg-dir PKG --hk-dir HK      (vLLM 0.30 venv; vLLM stopped; GPU lock)
  PKG/hzgs = horizon_ops/{__init__,gdn_decode,gdn_state,gdn_g3v}.py (fork horizon/g3v-sel-0115);
  HK = a directory with horizon_kernels/{__init__,gdn_g3}.py (the candidate plugin module).
Sections:
  equiv   production page layout (819,200-float pages, conv region 30,720 floats), 3 GDN KV-cache groups (gids 1..3
          behind an attention group 0), rings in mode B made by two real G3v steps per layer; then H1 (pure and
          mixed batches) + H4, and H2, each run twice from the same snapshot: the per-group path (ctx.sel False,
          proven bit-exact against vLLM's kernel by jobs 68 / 70) and the one-launch path (ctx.sel True). Gate:
          every state page (all layers), the meta table and the request-index buffer bit-identical; each hook kind
          must have selected and rewritten something. n = 1 / 2 / 5 / 8 / 32 / 64 (B 8 and 1600) and 300 (2nd
          selection pass), PP-filtered rows (-1), missing state indices (-1), windows past the block-table end.
  hooks   the per-step hook cost the 30-09 gate attributed the 1-user loss to: H1 + H4 + H2 of a pure decode step
          with nothing to expand (the common case), 48 layers in 3 groups, B 1600, n 1 / 8 / 64, old vs new:
          host time per step (no sync), stream time per step (events, synced), and a torch.profiler count of the
          GPU kernels launched per step with their summed device time.
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--pkg-dir", required=True)
ap.add_argument("--hk-dir", required=True)
A = ap.parse_args()
os.environ["HORIZON_GDN_G3"] = "1"
sys.path.insert(0, A.hk_dir)
sys.path.insert(0, A.pkg_dir)
from hzgs import gdn_g3v as G3  # noqa: E402
import horizon_kernels  # noqa: E402

_ops = types.ModuleType("horizon_kernels.g3v_ops")
_ops.gdn_g3v = G3
sys.modules["horizon_kernels.g3v_ops"] = _ops
sys.modules["horizon_kernels.g3v_ops.gdn_g3v"] = G3
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState as M  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q  # noqa: E402

M.add_request = lambda self, i, d: None
M.preprocess_state = lambda self, ib, bt, kc, nc: None
M.prepare_attn = lambda self, ib, cm, bt, sm, ag, kc, for_capture=False, ubatch_idx=0: None
M.postprocess_state = lambda self, im, ns, nc=None: None
Q.QwenGatedDeltaNetAttention._forward_core_decode_spec_post_conv_fused_norm = lambda self, *a, **k: None
from horizon_kernels import gdn_g3 as P  # noqa: E402

P.install()
dev = torch.device("cuda", 0)          # == the KV cache / query_start_loc device in vLLM (P._RIDX key)
HV, H, DK, DV = 48, 16, 128, 128
CQ = 2 * H * DK + HV * DV
QKVZ = CQ + HV * DV
PAGE, RF, SSOFF = 819200, 30720, 30720
NRQ = 512
T0 = time.time()


def log(rec):
    rec["t"] = round(time.time() - T0, 1)
    print(json.dumps(rec), flush=True)
    with open(A.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def bits(x, y):
    if not x.is_floating_point():
        return int((x != y).sum())
    return int(((x.view(torch.int32) != y.view(torch.int32)) & ~((x == 0) & (y == 0))).sum())


def pages(nb):
    flat = torch.zeros(nb * PAGE, device=dev)
    flat.normal_(0, 0.05)
    st = torch.as_strided(flat, (nb, HV, DV, DK), (PAGE, DV * DK, DK, 1), SSOFF)
    ring = torch.as_strided(flat, (nb, RF), (PAGE, 1), 0)
    return flat, st, ring


def params():
    return {"A_log": (torch.rand(HV, device=dev) * 2 - 1), "dt_bias": torch.randn(HV, device=dev).bfloat16(),
            "norm_w": (1 + 0.1 * torch.randn(DV, device=dev)).bfloat16()}


class Setup:
    """NG = 3 GDN groups (gids 1..3) x lpg layers, each layer its own pages; batch-row indexed block tables"""

    def __init__(self, n, B, lpg, WB, seed, shared_pages=False):
        self.n, self.B, self.WB, self.lpg = n, B, WB, lpg
        rng = np.random.default_rng(seed)
        self.rng = rng
        nb = 1 + n * WB
        NL = 3 * lpg
        if shared_pages:
            one = pages(nb)
            self.pg = [one] * NL
        else:
            self.pg = [pages(nb) for _ in range(NL)]
        self.bts = [torch.zeros(n + 2, WB, dtype=torch.int32, device=dev)]          # gid 0: attention group
        for g in range(3):
            perm = rng.permutation(n * WB) + 1
            self.bts.append(torch.tensor(np.concatenate([perm.reshape(n, WB), np.zeros((2, WB), int)]),
                                         dtype=torch.int32, device=dev))
        self.bts = tuple(self.bts)
        im = torch.tensor(rng.permutation(NRQ)[:n], dtype=torch.int64)
        if n > 2:
            im[1::5] = -1
        self.im = im.to(dev)
        sti = rng.integers(0, WB, NRQ)
        sti[rng.random(NRQ) < 0.1] = -1
        self.sti = torch.tensor(sti, dtype=torch.int32, device=dev)
        self.meta = torch.zeros(NL, NRQ, HV, 4, dtype=torch.int32, device=dev)
        groups = []
        for g in range(3):
            ls = list(range(g * lpg, (g + 1) * lpg))
            groups.append((g + 1, torch.tensor([self.pg[i][1].data_ptr() for i in ls], dtype=torch.int64, device=dev),
                           torch.tensor([self.pg[i][2].data_ptr() for i in ls], dtype=torch.int64, device=dev),
                           torch.tensor(ls, dtype=torch.int32, device=dev)))
        self.ctx = types.SimpleNamespace(
            ok=True, groups=groups, B=B, SW=4, HV=HV, H=H, SS=PAGE, RF=RF, meta=self.meta,
            rows=torch.zeros(P._RIDX_LEN + 1, dtype=torch.int32, device=dev),
            cnt=torch.zeros(1, dtype=torch.int32, device=dev), device=dev, layers=[("x", None)], ptr0=0, idx={},
            sel=True, sel_warm=True, gids=[1, 2, 3],
            all_saddr=torch.cat([g[1] for g in groups]), all_raddr=torch.cat([g[2] for g in groups]),
            all_lmeta=torch.cat([g[3] for g in groups]),
            all_lgrp=torch.tensor([g for g in range(3) for _ in range(lpg)], dtype=torch.int32, device=dev))
        P._CTX.clear()
        P._CTX[dev] = self.ctx
        P._model_ctx = lambda ms, kc: self.ctx
        self.ms = object.__new__(M)
        self.ms._align_mode = True
        self.ms._mamba_state_idx_gpu = self.sti

    def windows(self, g):
        """(n, 4) window slots of group g (1..3) at the rows' state index, 0 past the end / invalid rows"""
        im, sti, bt = self.im.cpu(), self.sti.cpu(), self.bts[g].cpu()
        w = torch.zeros(self.n, 4, dtype=torch.int32)
        for r in range(self.n):
            req = int(im[r])
            col = int(sti[req]) if req >= 0 else -1
            if col < 0:
                continue
            for k in range(4):
                if col + k < self.WB:
                    w[r, k] = bt[r, col + k]
        return w.to(dev)

    def make_mode_b(self, Pm, steps=2):
        """two real G3v steps per layer: rings with pending records (the plugin's decode path)"""
        gen = torch.Generator(device=dev)
        gen.manual_seed(5)
        ridx = torch.where(self.im >= 0, self.im, -1).to(torch.int32)
        cu = torch.arange(self.n + 1, dtype=torch.int32, device=dev) * 4
        for s in range(steps):
            qkvz = torch.nn.functional.silu(torch.randn(4 * self.n, QKVZ, device=dev, generator=gen)).bfloat16()
            ba = (torch.randn(4 * self.n, 2 * HV, device=dev, generator=gen) * 2).bfloat16()
            nacc = torch.tensor(self.rng.integers(1, 5, self.n) if s else np.ones(self.n, int), dtype=torch.int32,
                                device=dev)
            out = torch.empty(4 * self.n, HV, DV, dtype=torch.bfloat16, device=dev)
            for g in range(3):
                w = self.windows(g + 1)
                for li in range(g * self.lpg, (g + 1) * self.lpg):
                    _, st, ring = self.pg[li]
                    G3.step(qkvz[:, :CQ], ba[:, HV:], ba[:, :HV], Pm["A_log"], Pm["dt_bias"], w, cu, nacc, ridx, st,
                            ring, self.meta[li], qkvz[:, CQ:].view(-1, HV, DV), Pm["norm_w"], out, DK ** -0.5, 1e-6,
                            "silu")
        torch.cuda.synchronize()

    def batch(self, pure):
        n = self.n
        ql = np.full(n, 4)
        d = ql - 1
        if not pure:
            d[0] = 0
            ql[0] = 37
        qsl = torch.zeros(n + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(np.cumsum(ql), dtype=torch.int32)
        return types.SimpleNamespace(num_reqs=n, idx_mapping=self.im, query_start_loc=qsl.to(dev),
                                     num_scheduled_tokens=ql, num_draft_tokens_per_req=d,
                                     is_prefilling_np=np.zeros(n, bool))

    def nc_pre(self, cross_p):
        """num_computed_tokens by request-state index: o = offset in the window block; a 4-token step crosses iff
        o + 4 > B"""
        sti = self.sti.cpu().numpy().clip(0)
        o = np.where(self.rng.random(NRQ) < cross_p, self.rng.integers(max(1, self.B - 3), self.B + 1, NRQ),
                     self.rng.integers(1, max(2, self.B - 4), NRQ))
        return torch.tensor(sti * self.B + o, dtype=torch.int32, device=dev)

    def nc_post(self, hit_p):
        """post: aligned >= running iff nc % B < acc"""
        sti = self.sti.cpu().numpy().clip(0)
        o = np.where(self.rng.random(NRQ) < hit_p, self.rng.integers(0, 2, NRQ),
                     self.rng.integers(min(5, self.B - 1), self.B, NRQ))
        return torch.tensor(sti * self.B + self.B + o, dtype=torch.int32, device=dev)

    def snap(self):
        return [p[0].clone() for p in self.pg], self.meta.clone()

    def restore(self, s):
        for p, f in zip(self.pg, s[0]):
            p[0].copy_(f)
        self.meta.copy_(s[1])
        P._ridx(dev).fill_(77)

    def state(self):
        torch.cuda.synchronize()
        return [p[0].clone() for p in self.pg], self.meta.clone(), P._ridx(dev).clone()


def run_hook(S, kind, nc, ns=None, ib=None):
    if kind == "post":
        M.postprocess_state(S.ms, S.im, ns, nc)
    else:
        M.preprocess_state(S.ms, ib, S.bts, None, nc)
        M.prepare_attn(S.ms, ib, None, S.bts, None, None, None, for_capture=False)


def equiv(Pm, n, B, lpg, WB, seed):
    S = Setup(n, B, lpg, WB, seed)
    S.make_mode_b(Pm)
    base = S.snap()
    pend = int(((S.meta[:, :, :, 0] // 256) % 256 > 0).sum())
    bad_all = 0
    for kind in ("pre_pure", "pre_mixed", "post"):
        for rep in range(2):
            ib = S.batch(kind != "pre_mixed") if kind != "post" else None
            nc = S.nc_pre(0.5) if kind != "post" else S.nc_post(0.5)
            ns = torch.tensor(S.rng.integers(0, 5, n), dtype=torch.int32, device=dev) if kind == "post" else None
            res = {}
            for sel in (False, True):
                S.restore(base)
                S.ctx.sel = sel
                run_hook(S, kind, nc, ns, ib)
                res[sel] = S.state()
            f0, m0, r0 = res[False]
            f1, m1, r1 = res[True]
            dstate = sum(bits(a, b) for a, b in zip(f0, f1))
            dmeta, dridx = bits(m0, m1), bits(r0, r1)
            changed = sum(int((a.view(torch.int32) != b.view(torch.int32)).sum()) for a, b in zip(f0, base[0]))
            cleared = int(((base[1][:, :, :, 0] // 256) % 256 > 0).sum() - ((m0[:, :, :, 0] // 256) % 256 > 0).sum())
            bad = dstate + dmeta + dridx
            bad_all += bad
            log({"test": "equiv", "n": n, "B": B, "layers": 3 * lpg, "kind": kind, "rep": rep, "pending_before": pend,
                 "rings_expanded": cleared, "floats_rewritten": changed, "state_diff": dstate, "meta_diff": dmeta,
                 "ridx_diff": dridx})
            del res, f0, f1, m0, m1
    del S, base
    torch.cuda.empty_cache()
    return bad_all


def hooks_cost(n, iters=200):
    S = Setup(n, 1600, 16, 6, 11, shared_pages=True)
    ib = S.batch(True)
    sti = S.sti.cpu().numpy().clip(0)
    nc_pre = torch.tensor(sti * 1600 + 100, dtype=torch.int32, device=dev)     # no crossing (o + 4 <= B)
    nc_post = torch.tensor(sti * 1600 + 1600 + 700, dtype=torch.int32, device=dev)  # nc % B = 700 >= acc
    ns = torch.full((n,), 3, dtype=torch.int32, device=dev)

    def step():
        M.preprocess_state(S.ms, ib, S.bts, None, nc_pre)
        M.prepare_attn(S.ms, ib, None, S.bts, None, None, None, for_capture=False)
        M.postprocess_state(S.ms, S.im, ns, nc_post)

    rec = {"test": "hooks", "n": n, "layers": 48, "groups": 3, "B": 1600}
    for sel in (False, True):
        S.ctx.sel = sel
        for _ in range(20):
            step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            step()
        host = (time.perf_counter() - t) / iters * 1e6
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            step()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1) * 1000)
        ts.sort()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
        kern = [e for e in prof.events() if e.device_type.name == "CUDA"]
        names = {}
        for e in kern:
            names[e.name[:60]] = names.get(e.name[:60], 0) + 1
        tag = "new" if sel else "old"
        rec[tag] = {"host_us_per_step": round(host, 1), "stream_us_per_step_median": round(ts[len(ts) // 2], 1),
                    "stream_us_p10": round(ts[len(ts) // 10], 1), "gpu_ops_per_step": round(len(kern) / 10, 1),
                    "gpu_us_per_step": round(sum(e.device_time_total for e in kern) / 10, 1),
                    "top_ops": sorted(names.items(), key=lambda x: -x[1])[:6]}
    rec["stream_ratio_new_old"] = round(rec["new"]["stream_us_per_step_median"] /
                                        rec["old"]["stream_us_per_step_median"], 3)
    log(rec)
    del S
    torch.cuda.empty_cache()


if __name__ == "__main__":
    log({"test": "start", "torch": torch.__version__, "device": torch.cuda.get_device_name(),
         "sel_kernel": G3.__file__})
    torch.manual_seed(0)
    Pm = params()
    bad = 0
    for i, (n, B) in enumerate([(1, 8), (2, 8), (5, 8), (8, 8), (32, 8), (64, 8), (8, 1600), (64, 1600)]):
        bad += equiv(Pm, n, B, 2, 6, 100 + i)
    bad += equiv(Pm, 300, 8, 1, 4, 200)
    for n in (1, 8, 64):
        hooks_cost(n)
    log({"test": "end", "complete": True, "equiv_bad": bad, "secs": round(time.time() - T0, 1)})
