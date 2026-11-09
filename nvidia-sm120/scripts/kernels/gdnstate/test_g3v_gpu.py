# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for G3v (horizon_ops.gdn_g3v: the G3 replay ring on vLLM 0.30's state layout), 30-09-2026. MEASURE ONLY.

usage: python test_g3v_gpu.py OUT.jsonl --pkg-dir PKG [quick]      (vLLM 0.30 venv; vLLM stopped; under the GPU lock)
  PKG contains hzgs/ = horizon_ops/{__init__,gdn_decode,gdn_state,gdn_g3v}.py (branch horizon/g3v-0115).
Pages are laid out like vLLM's hybrid pages: per block one 819,200-float page = conv region (30,720 floats) + ssm
state (48 x 128 x 128). Sections:
  chain   World A = vLLM 0.30's own CUDA kernel (torch.ops._C.fused_gdn_decode_post_conv_mtp) + the align-mode copies
          (pre-copy on a window move, post-copy per postprocess_mamba_fused_kernel's rule) + non-pure steps;
          World B = G3v step + expand (the plugin's H1/H2 rules) + the same copies; small blocks (many moves and
          post-copies), QL 2 <-> 4 per step, random acceptance, random non-pure steps. Gate: outputs every step, the
          state of every copy source at copy time, every completed block and every committed state bit-identical.
  graph   determinism (3 runs) + CUDA-graph replay (thread_local) of the G3v step incl. ring and meta updates
  bench   us per layer, pure decode steps: vLLM kernel vs G3v at 1/8/32/64 requests x QL 2/4, CUDA-graph replay of
          one step over >512 MB of rotated layers (ring + meta advance every replay: steady state incl. flushes);
          plus the expand cost (16 layers, all rows pending)
"""
import argparse
import json
import math
import sys
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("mode", nargs="?", default="full")
ap.add_argument("--pkg-dir", required=True)
A = ap.parse_args()
sys.path.insert(0, A.pkg_dir)
from vllm import _custom_ops as ops  # noqa: E402
from hzgs import gdn_g3v as G3  # noqa: E402
from hzgs import gdn_decode as GD  # noqa: E402

QUICK = A.mode == "quick"
dev = torch.device("cuda")
HV, H, DK, DV = 48, 16, 128, 128
CQ = 2 * H * DK + HV * DV
QKVZ = CQ + HV * DV
PAGE, RF, SSOFF = 819200, 30720, 30720
MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"
T0 = time.time()


def log(rec):
    rec["t"] = round(time.time() - T0, 1)
    print(json.dumps(rec), flush=True)
    with open(A.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def real_params(layer=0):
    from safetensors import safe_open
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    p = f"model.language_model.layers.{layer}.linear_attn."
    with safe_open(f"{MODEL}/{idx[p + 'A_log']}", "pt") as f:
        return {"A_log": f.get_tensor(p + "A_log").float().to(dev).contiguous(),
                "dt_bias": f.get_tensor(p + "dt_bias").to(torch.bfloat16).to(dev).contiguous(),
                "norm_w": f.get_tensor(p + "norm.weight").to(torch.bfloat16).to(dev).contiguous()}


def pages(nb, gen=None):
    flat = torch.zeros(nb * PAGE, device=dev)
    if gen is not None:
        flat.normal_(0, 0.05, generator=gen)
    st = torch.as_strided(flat, (nb, HV, DV, DK), (PAGE, DV * DK, DK, 1), SSOFF)
    ring = torch.as_strided(flat, (nb, RF), (PAGE, 1), 0)
    return flat, st, ring


def bits(x, y):
    if not x.is_floating_point():
        return int((x != y).sum())
    xi = x.contiguous().view(torch.int32 if x.dtype == torch.float32 else torch.int16)
    yi = y.contiguous().view(torch.int32 if y.dtype == torch.float32 else torch.int16)
    return int(((xi != yi) & ~((x == 0) & (y == 0))).sum())


def inputs(ntoks, gen):
    L = max(1, sum(ntoks))
    qkvz = torch.nn.functional.silu(torch.randn(L, QKVZ, device=dev, generator=gen)).to(torch.bfloat16)
    ba = (torch.randn(L, 2 * HV, device=dev, generator=gen) * 2).to(torch.bfloat16)
    cu = torch.zeros(len(ntoks) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(ntoks), 0)
    return qkvz[:, :CQ], ba[:, HV:], ba[:, :HV], qkvz[:, CQ:].view(L, HV, DV), cu.to(dev)


def vllm_step(P, mixed, a, b, z, sidx, cu, nacc, st, out):
    ops.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed, a=a, b=b, A_log=P["A_log"], dt_bias=P["dt_bias"],
                                       state_indices=sidx, cu_seqlens=cu, num_accepted_tokens=nacc, state=st,
                                       output_gate=z, norm_weight=P["norm_w"], out=out, scale=DK ** -0.5,
                                       norm_eps=1e-6, output_gate_activation="silu")


class World:
    def __init__(self, nr, B, g3, init, npr):
        self.nr, self.B, self.g3 = nr, B, g3
        self.nb = 1 + nr * npr
        self.flat, self.st, self.ring = pages(self.nb)
        self.bt = [list(range(1 + i * npr, 1 + (i + 1) * npr)) for i in range(nr)]
        self.meta = torch.zeros(1, 64, HV, 4, dtype=torch.int32, device=dev)
        self.computed = [5 + 3 * i for i in range(nr)]
        self.col = [(c - 1) // B for c in self.computed]
        self.acc = [1] * nr
        for i in range(nr):
            self.st[self.bt[i][self.col[i]]] = init[i]
        self.saddr = torch.tensor([self.st.data_ptr()], dtype=torch.int64, device=dev)
        self.raddr = torch.tensor([self.ring.data_ptr()], dtype=torch.int64, device=dev)
        self.lmeta = torch.zeros(1, dtype=torch.int32, device=dev)
        self.ridx = torch.arange(nr, dtype=torch.int32, device=dev)
        self.obs, self.flushes = [], 0

    def win(self, i, col):
        return [self.bt[i][col + k] if col + k < len(self.bt[i]) else 0 for k in range(4)]

    def expand(self, rows, cols):
        if not self.g3 or not rows:
            return
        sidx = torch.tensor([self.win(i, cols[i]) for i in range(self.nr)], dtype=torch.int32, device=dev)
        rl = torch.tensor(rows + [0] * (self.nr - len(rows)), dtype=torch.int32, device=dev)
        G3.expand(self.saddr, self.raddr, self.lmeta, rl, torch.tensor([len(rows)], dtype=torch.int32, device=dev),
                  sidx, self.ridx, self.meta, HV, H, PAGE, RF)

    def pre(self, T, mixed):
        new = [(self.computed[i] + T - 1) // self.B for i in range(self.nr)]
        cross = [i for i in range(self.nr) if new[i] != self.col[i]]
        self.expand(list(range(self.nr)) if mixed else cross, self.col)
        for i in cross:
            src = self.bt[i][self.col[i] + self.acc[i] - 1]
            self.obs.append(self.st[src].clone())
            self.st[self.bt[i][new[i]]] = self.st[src]
            self.acc[i] = 1
        self.col = new

    def fwd(self, P, inp, mixed):
        mixed_qkv, a, b, z, cu = inp
        sidx = torch.tensor([self.win(i, self.col[i]) for i in range(self.nr)], dtype=torch.int32, device=dev)
        nacc = torch.tensor(self.acc, dtype=torch.int32, device=dev)
        out = torch.full((mixed_qkv.shape[0], HV, DV), 7.0, dtype=torch.bfloat16, device=dev)
        if self.g3 and not mixed:
            G3.step(mixed_qkv, a, b, P["A_log"], P["dt_bias"], sidx, cu, nacc, self.ridx, self.st, self.ring,
                    self.meta[0], z, P["norm_w"], out, DK ** -0.5, 1e-6, "silu")
        else:
            vllm_step(P, mixed_qkv, a, b, z, sidx, cu, nacc, self.st, out)
        return out

    def post(self, acc):
        rows, cps = [], []
        for i in range(self.nr):
            self.acc[i] = acc[i]
            self.computed[i] += acc[i]
            newc = self.computed[i]
            running, aligned = newc - acc[i] + 1, (newc // self.B) * self.B
            if aligned >= running:
                rows.append(i)
                cps.append((i, aligned - running, aligned // self.B - 1))
        self.expand(rows, self.col)
        for i, bias, dest in cps:
            if self.col[i] == dest:
                self.acc[i] = 1
            if self.col[i] == dest and bias == 0:
                continue
            src = self.bt[i][self.col[i] + bias]
            self.obs.append(self.st[src].clone())
            self.st[self.bt[i][dest]] = self.st[src]


def chain(P, nr, steps, B, seed, mixed_p):
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    rnd = torch.Generator().manual_seed(seed)
    npr = (5 + 3 * nr + steps * 4) // B + 16      # the initial positions (5 + 3 i) plus up to 4 tokens a step
    init = torch.randn(nr, HV, DV, DK, device=dev, generator=gen) * 0.05
    WA, WB = World(nr, B, False, init, npr), World(nr, B, True, init, npr)
    ob = cb = 0
    for s in range(steps):
        T = 4 if torch.rand(1, generator=rnd).item() < 0.5 else 2
        mixed = torch.rand(1, generator=rnd).item() < mixed_p
        na, nb_ = len(WA.obs), len(WB.obs)
        WA.pre(T, mixed); WB.pre(T, mixed)
        inp = inputs([T] * nr, gen)
        ob += bits(WA.fwd(P, inp, mixed), WB.fwd(P, inp, mixed))
        acc = [int(torch.randint(1, T + 1, (1,), generator=rnd)) for _ in range(nr)]
        WA.post(acc); WB.post(acc)
        cb += (len(WA.obs) != len(WB.obs)) + sum(bits(x, y) for x, y in zip(WA.obs[na:], WB.obs[nb_:]))
    WB.expand(list(range(nr)), WB.col)
    torch.cuda.synchronize()
    fin = sum(bits(WA.st[WA.bt[i][WA.col[i] + WA.acc[i] - 1]], WB.st[WB.bt[i][WB.col[i] + WB.acc[i] - 1]])
              for i in range(nr))
    blk = sum(bits(WA.st[WA.bt[i][j]], WB.st[WB.bt[i][j]]) for i in range(nr) for j in range(WA.col[i]))
    rec = {"test": "chain", "nr": nr, "steps": steps, "B": B, "mixed_p": mixed_p, "out_diff": ob, "copy_diff": cb,
           "copies": len(WA.obs), "committed_diff": fin, "blocks_diff": blk}
    log(rec)
    return ob + cb + fin + blk


def graph_test(P, nr=8, ql=4):
    gen = torch.Generator(device=dev); gen.manual_seed(3)
    flat, st, ring = pages(nr * 4 + 1, gen)
    meta = torch.zeros(64, HV, 4, dtype=torch.int32, device=dev)
    sidx = (torch.arange(nr * 4, dtype=torch.int32, device=dev) + 1).view(nr, 4)
    ridx = torch.arange(nr, dtype=torch.int32, device=dev)
    nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
    mixed, a, b, z, cu = inputs([ql] * nr, gen)
    out = torch.empty(mixed.shape[0], HV, DV, dtype=torch.bfloat16, device=dev)

    def step():
        G3.step(mixed, a, b, P["A_log"], P["dt_bias"], sidx, cu, nacc, ridx, st, ring, meta, z, P["norm_w"], out,
                DK ** -0.5, 1e-6, "silu")

    for _ in range(3):
        step()                                     # rings with content, mixed flush / no-flush meta
    snap = (flat.clone(), meta.clone())
    res = []
    for _ in range(3):
        flat.copy_(snap[0]); meta.copy_(snap[1]); step(); torch.cuda.synchronize()
        res.append((out.clone(), flat.clone(), meta.clone()))
    det = all(bits(res[0][k], r[k]) == 0 for r in res[1:] for k in range(3))
    flat.copy_(snap[0]); meta.copy_(snap[1])
    g = torch.cuda.CUDAGraph()
    for attempt in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                step()
            break
        except Exception as e:  # noqa: BLE001
            print("capture retry", attempt, e, flush=True)
            torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
    eq = True
    for _ in range(2):
        flat.copy_(snap[0]); meta.copy_(snap[1]); out.zero_(); g.replay(); torch.cuda.synchronize()
        eq = eq and all(bits(res[0][k], x) == 0 for k, x in enumerate((out, flat, meta)))
    log({"test": "graph", "kernel": "g3v_step", "deterministic": det, "graph_replay_equal": eq})
    return det and eq


def bench(P, rounds=3, iters=20):
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    for nr in (1, 8, 32, 64):
        for ql in (2, 4):
            mixed, a, b, z, cu = inputs([ql] * nr, gen)
            out = torch.empty(mixed.shape[0], HV, DV, dtype=torch.bfloat16, device=dev)
            nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
            sidx = (torch.arange(nr * 4, dtype=torch.int32, device=dev) + 1).view(nr, 4)
            ridx = torch.arange(nr, dtype=torch.int32, device=dev)
            per = (nr * 4 + 1) * PAGE * 4
            nl = min(max(4, math.ceil(600e6 / per)), max(2, int(8e9 // per)))
            layers = [pages(nr * 4 + 1, gen) + (torch.zeros(64, HV, 4, dtype=torch.int32, device=dev),)
                      for _ in range(nl)]

            def run_vllm():
                for flat, st, ring, meta in layers:
                    vllm_step(P, mixed, a, b, z, sidx, cu, nacc, st, out)

            def run_g3():
                for flat, st, ring, meta in layers:
                    G3.step(mixed, a, b, P["A_log"], P["dt_bias"], sidx, cu, nacc, ridx, st, ring, meta, z,
                            P["norm_w"], out, DK ** -0.5, 1e-6, "silu")

            graphs = {}
            for name, fn in (("vllm_f32", run_vllm), ("g3v_f32", run_g3)):
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                for attempt in range(5):
                    try:
                        with torch.cuda.graph(g, capture_error_mode="thread_local"):
                            fn()
                        break
                    except Exception as e:  # noqa: BLE001  (gpushare capture flake)
                        print("capture retry", attempt, e, flush=True)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                graphs[name] = g
            best = {n: float("inf") for n in graphs}
            order = list(graphs)
            for rnd in range(rounds):
                for name in (order if rnd % 2 == 0 else order[::-1]):
                    g = graphs[name]
                    g.replay(); torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ts = []
                    for _ in range(iters):
                        e0.record(); g.replay(); e1.record(); e1.synchronize()
                        ts.append(e0.elapsed_time(e1) * 1000 / nl)
                    ts.sort()
                    best[name] = min(best[name], ts[len(ts) // 2])
            log({"test": "bench", "nr": nr, "ql": ql, "layers": nl, "us": {k: round(v, 2) for k, v in best.items()},
                 "ratio_g3v_vllm": round(best["g3v_f32"] / best["vllm_f32"], 4)})
            # expand cost: 16 layers, every row pending (the ring content of the last replay)
            if ql == 4 and nr in (32, 64):
                n16 = min(16, nl)
                saddr = torch.tensor([layers[i][1].data_ptr() for i in range(n16)], dtype=torch.int64, device=dev)
                raddr = torch.tensor([layers[i][2].data_ptr() for i in range(n16)], dtype=torch.int64, device=dev)
                metas = torch.stack([layers[i][3] for i in range(n16)])
                snap = metas.clone()
                rows = torch.arange(nr, dtype=torch.int32, device=dev)
                cnt = torch.tensor([nr], dtype=torch.int32, device=dev)
                ts = []
                for _ in range(5):
                    metas.copy_(snap)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    G3.expand(saddr, raddr, torch.arange(n16, dtype=torch.int32, device=dev), rows, cnt, sidx, ridx,
                              metas, HV, H, PAGE, RF)
                    e1.record(); e1.synchronize()
                    ts.append(e0.elapsed_time(e1) * 1000)
                zero = torch.zeros(1, dtype=torch.int32, device=dev)
                e0.record()
                G3.expand(saddr, raddr, torch.arange(n16, dtype=torch.int32, device=dev), rows, zero, sidx, ridx, metas,
                          HV, H, PAGE, RF)
                e1.record(); e1.synchronize()
                log({"test": "expand_cost", "nr": nr, "layers": n16, "us_all_pending": round(sorted(ts)[2], 1),
                     "us_empty": round(e0.elapsed_time(e1) * 1000, 1)})
            del layers, graphs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    P = real_params(0)
    log({"test": "start", "mode": A.mode, "torch": torch.__version__, "device": torch.cuda.get_device_name()})
    plan = [(2, 150, 7, 11, 0.15), (8, 80, 5, 12, 0.3), (32, 40, 11, 13, 0.1), (64, 20, 3, 14, 0.2)]
    if QUICK:
        plan = [(2, 60, 7, 11, 0.15), (32, 20, 5, 13, 0.2)]
    bad = sum(chain(P, *p) for p in plan)
    gok = graph_test(P)
    bench(P)
    log({"test": "end", "complete": True, "chain_bad": bad, "graph_ok": gok, "secs": round(time.time() - T0, 1)})
