# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 3: fused MoE block against the stock block, geometry of the public 35B-A3B checkpoint
(256 routed experts + 1 shared, hidden 2048, intermediate 512, top-8).

Arms per cell (one process, same activations, the same rotated layers):
  new     one skinny gate GEMM over the fused 264-row gate (rows <= --skinny-max, F.linear above), the route kernel
          (route_shared), the STABLE alignment kernel (stable_align.align, handed to Marlin-MoE through the
          alignment-reuse hook), Marlin-MoE over 257 experts (the shared expert is expert 256)
  stock   what vLLM runs: F.linear gate, F.linear shared gate, fused_topk, Marlin-MoE over 256 experts (its own
          moe_align_block_size), shared expert on dense Marlin, sigmoid gate, add
  ctl     added by the harness
The expert GEMMs are Marlin's in both arms: this measures the routing glue and the shared-expert fusion only.
Cells: 1, 2, 4, 8, 16, 32, 64, 128 tokens.

Correctness per cell before timing:
  fp64 reference of the block on up to 24 rows (weights dequantised exactly, each arm's own routing); rule of the
  first study: new rel-L2 <= 1.10 x stock rel-L2 + 1e-4; routing ids equal to stock; the stable alignment equal to a
  CPU-built canonical alignment element for element; the hook really reused it; non-finite count; repeat equality
  and graph replay == eager from the harness (the stock arm is known to fail repeat equality: it is recorded, and
  it does not gate).
Weights: synthetic seeded layers of that geometry (global scales 1.0), or with --ckpt-dir real layers of a checkpoint
directory opened read-only. Activations are always synthetic.

usage: race_moe_block.py --out FILE [--layers 2] [--ckpt-dir DIR --ckpt-layers 3,7] [--reverse] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

TARGET = "moe_block"
FIXED_MIB = 800
E, H, I, TOPK, LD = 256, 2048, 512, 8, 264
MS = [1, 2, 4, 8, 16, 32, 64, 128]
ENERGY_M = (1, 4, 32, 128)
STOCK_US = {1: 55, 2: 70, 4: 91, 8: 120, 16: 160, 32: 216, 64: 260, 128: 325}   # first study, for the time estimate
LAYER_BYTES = (E + 1) * (2 * I * H // 2 + H * I // 2) + (E + 1) * (2 * I * H // 16 + H * I // 16)


def build_cells(a):
    cells = []
    by = a.layers * 2 * LAYER_BYTES + 2 * LAYER_BYTES + 256 * 2**20
    for i, M in enumerate(MS):
        energy = M in ENERGY_M
        est = C.est_seconds(a, [(STOCK_US[M], a.layers)] * 3, (25.0 * a.layers if i == 0 else 0) + 6.0, energy, 3)
        cells.append(C.Cell(f"M{M}", {"M": M, "layers": a.layers, "weights": "checkpoint" if a.ckpt_dir else
                                      "synthetic"}, by / 2**20, est, energy))
    return cells


def main():
    p = C.base_parser(__doc__, mem_budget_mib=4096)
    p.add_argument("--layers", type=int, default=2, help="rotated layers (each about 0.9 GB for both arms)")
    p.add_argument("--ckpt-dir", help="checkpoint directory, opened read-only, instead of synthetic layers")
    p.add_argument("--ckpt-layers", default="3,7")
    p.add_argument("--skinny-max", type=int, default=0, help="largest row count on the skinny gate GEMM (0: its own limit)")
    a = p.parse_args()
    if a.ckpt_dir:
        a.layers = len(a.ckpt_layers.split(","))
    if a.layers < 2:
        p.error("at least two rotated layers")
    race = C.Race(a, TARGET, FIXED_MIB, "fused MoE block (skinny gate + route kernel + stable alignment) vs stock "
                  "routing glue, Marlin-MoE experts in both arms")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    torch = race.start({"geometry": {"experts": E, "hidden": H, "intermediate": I, "topk": TOPK},
                        "weights": "checkpoint" if a.ckpt_dir else "synthetic", "layers": a.layers})
    import json
    import types
    import torch.nn.functional as F
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import marlin_moe as mm
    from vllm.model_executor.layers.fused_moe.router import fused_topk_router as ftr
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mu
    from vllm.scalar_type import scalar_types
    from tilelang.contrib.horizon_fused import skinny_bf16 as sk
    from tilelang.contrib.horizon_moe import route
    try:
        from tilelang.contrib.horizon_moe import stable_align as sa
    except ImportError:
        import stable_align as sa
    from horizon_kernels import moe as hmoe
    hmoe._install_align_reuse()
    dev = "cuda"
    skinny_max = a.skinny_max or sk.SUPPORTS_M
    lut64 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev,
                         dtype=torch.float64)

    def deq(w, s, s2):
        w = w.to(dev).view(torch.uint8)
        v = torch.stack([lut64[(w & 15).long()], lut64[(w >> 4).long()]], -1).reshape(w.shape[0], -1)
        return v * s.to(dev).float().double().repeat_interleave(16, 1) * float(s2)

    def synthetic(seed):
        """(w13, w13 scales, g13, w2, w2 scales, g2, gate, shared gate) with the checkpoint's layout."""
        g = torch.Generator(device=dev).manual_seed(seed)
        n = E + 1

        def sc(*shape):
            return (torch.rand(*shape, device=dev, generator=g) * 0.06 + 0.02).to(torch.float8_e4m3fn)
        w13 = torch.randint(0, 256, (n, 2 * I, H // 2), device=dev, generator=g, dtype=torch.uint8)
        w2 = torch.randint(0, 256, (n, H, I // 2), device=dev, generator=g, dtype=torch.uint8)
        gate = (torch.randn(E, H, device=dev, generator=g) * 0.05).to(torch.bfloat16)
        sgate = (torch.randn(1, H, device=dev, generator=g) * 0.05).to(torch.bfloat16)
        return w13, sc(n, 2 * I, H // 16), torch.ones(n, device=dev), w2, sc(n, H, I // 16), torch.ones(n, device=dev), \
            gate, sgate

    def from_checkpoint(layer):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(a.ckpt_dir, "model.safetensors.index.json")))["weight_map"]
        files = {}

        def get(name):
            fn = idx[name]
            if fn not in files:
                files[fn] = safe_open(os.path.join(a.ckpt_dir, fn), "pt", device="cpu")
            return files[fn].get_tensor(name)
        pre = f"model.language_model.layers.{layer}.mlp"
        w13, s13, g13, w2, s2, g2 = [], [], [], [], [], []
        for e in list(range(E)) + ["shared"]:
            b = f"{pre}.shared_expert" if e == "shared" else f"{pre}.experts.{e}"
            gw, gs, gg = (get(b + ".gate_proj" + k) for k in (".weight", ".weight_scale", ".weight_scale_2"))
            uw, us, ug = (get(b + ".up_proj" + k) for k in (".weight", ".weight_scale", ".weight_scale_2"))
            dw, ds, dg = (get(b + ".down_proj" + k) for k in (".weight", ".weight_scale", ".weight_scale_2"))
            assert torch.equal(gg, ug), "gate/up global scales not tied"
            w13.append(torch.cat([gw, uw], 0)); s13.append(torch.cat([gs, us], 0)); g13.append(gg.float().reshape(()))
            w2.append(dw); s2.append(ds); g2.append(dg.float().reshape(()))
        st = lambda v: torch.stack(v).to(dev)  # noqa: E731
        return st(w13), st(s13), st(g13), st(w2), st(s2), st(g2), get(f"{pre}.gate.weight").to(dev, torch.bfloat16), \
            get(f"{pre}.shared_expert_gate.weight").to(dev, torch.bfloat16)

    def marlin_moe(w13, s13, g13, w2, s2, g2):
        layer = types.SimpleNamespace(num_experts=w13.shape[0], hidden_size=H, intermediate_size_per_partition=I,
                                      params_dtype=torch.bfloat16)
        r = mu.prepare_nvfp4_moe_layer_for_marlin(layer, w13.clone(), s13.clone(), g13.clone(), w2.clone(),
                                                  s2.clone(), g2.clone(), is_act_and_mul=True)
        return dict(w1=r[0], w1_scale=r[1], global_scale1=r[2], w2=r[3], w2_scale=r[4], global_scale2=r[5],
                    workspace=layer.workspace)

    def dense(w, s, g, n, k):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        m.weight_scale = torch.nn.Parameter(s.clone(), requires_grad=False)
        m.weight_global_scale = torch.nn.Parameter(g.reshape(1).clone(), requires_grad=False)
        m.output_size_per_partition, m.input_size_per_partition, m.params_dtype = n, k, torch.bfloat16
        mu.prepare_fp4_layer_for_marlin(m)
        return m

    def build_layer(i, keep_raw):
        raw = from_checkpoint(int(a.ckpt_layers.split(",")[i])) if a.ckpt_dir else synthetic(a.seed * 100 + i)
        w13, s13, g13, w2, s2, g2, gate, sgate = raw
        L = types.SimpleNamespace(gate=gate, sgate=sgate)
        L.m256 = marlin_moe(w13[:E], s13[:E], g13[:E], w2[:E], s2[:E], g2[:E])
        L.m257 = marlin_moe(w13, s13, g13, w2, s2, g2)
        L.sgu = dense(w13[E], s13[E], g13[E], 2 * I, H)
        L.sdn = dense(w2[E], s2[E], g2[E], H, I)
        g264 = torch.zeros(LD, H, device=dev, dtype=torch.bfloat16)
        g264[:E] = gate
        g264[E] = sgate[0]
        L.gate264 = g264.contiguous()
        L.raw = raw[:6] if keep_raw else None
        return L

    def moe(c, x, tw, ti):
        return mm.fused_marlin_moe(hidden_states=x, w1=c["w1"], w2=c["w2"], bias1=None, bias2=None,
                                   w1_scale=c["w1_scale"], w2_scale=c["w2_scale"], topk_weights=tw, topk_ids=ti,
                                   quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=-1,
                                   activation=MoEActivation.SILU, global_scale1=c["global_scale1"],
                                   global_scale2=c["global_scale2"], workspace=c["workspace"])

    def dlin(m, x):
        return mu.apply_fp4_marlin_linear(x, m.weight, m.weight_scale, m.weight_global_scale, m.workspace,
                                          m.output_size_per_partition, m.input_size_per_partition)

    def arm_stock(L, x, want=False):
        logits = F.linear(x, L.gate)
        tw, ti, _ = ftr.fused_topk(x, logits, TOPK, True, None, "softmax")
        routed = moe(L.m256, x, tw, ti)
        gu = dlin(L.sgu, x)
        s = dlin(L.sdn, F.silu(gu[:, :I]) * gu[:, I:])
        y = routed + torch.sigmoid(F.linear(x, L.sgate)) * s
        return (y, tw, ti) if want else y

    def arm_new(L, x, want=False):
        logits = sk.linear(x, L.gate264) if x.shape[0] <= skinny_max else F.linear(x, L.gate264)
        tw, ti = route.route_shared(logits, E, TOPK, True)
        if ti.dtype != torch.int32 or not ti.is_contiguous():
            ti = ti.to(torch.int32).contiguous()
        bs = hmoe._marlin_block_size(x.shape[0], TOPK + 1, E + 1)
        al = sa.align(ti, E + 1, bs)
        hmoe._ALIGN[0] = (ti, bs, E + 1, al)
        y = moe(L.m257, x, tw, ti)
        return (y, tw, ti, al, bs) if want else y

    def canonical(ids, bs, experts):
        rows = ids.cpu().reshape(-1).tolist()
        size, blocks = route.align_sizes(len(rows), experts, bs)
        so, eo = [], []
        for ex in range(experts):
            pairs = [i for i, v in enumerate(rows) if v == ex]
            padded = ((len(pairs) + bs - 1) // bs) * bs
            so.extend(pairs + [len(rows)] * (padded - len(pairs)))
            eo.extend([ex] * (padded // bs))
        used = len(so)
        so.extend([len(rows)] * (size - used))
        eo.extend([-1] * (blocks - len(eo)))
        return [torch.tensor(v, device=ids.device, dtype=torch.int32) for v in (so, eo, [used])]

    def reference(L, x, tw, ti, rows):
        """fp64 block output for the given rows: routed slots with the arm's own routing + the gated shared expert."""
        w13, s13, g13, w2, s2, g2 = L.raw
        x64 = x.double()
        y = torch.zeros(len(rows), H, device=dev, dtype=torch.float64)

        def expert(e, xr):
            wgu = deq(w13[e], s13[e], g13[e])
            wd = deq(w2[e], s2[e], g2[e])
            gu = wgu @ xr
            return wd @ (F.silu(gu[:I]) * gu[I:])
        for k, t in enumerate(rows):
            for j in range(TOPK):
                y[k] += float(tw[t, j]) * expert(int(ti[t, j]), x64[t])
            sg = torch.sigmoid(L.sgate.double()[0] @ x64[t])
            y[k] += sg * expert(E, x64[t])
        return y

    with set_current_vllm_config(VllmConfig()):
        layers = [build_layer(i, i == a.layers - 1) for i in range(a.layers)]
        last = layers[-1]
        route.warm(E, LD, TOPK, True, dev, block_sizes=(8, 16))
        for cfg in sk.all_configs(LD, H):
            for m in (1, cfg[0], sk.SUPPORTS_M):
                sk.linear(torch.zeros(m, H, device=dev, dtype=torch.bfloat16), last.gate264, cfg=cfg)
        torch.cuda.synchronize()
        code_new = [route, sk, sa, hmoe, mm]
        code_stock = [mm, ftr, mu, "so:_C.", "so:libcublas"]
        for index, cell in enumerate(cells):
            if race.over_time(cell):
                break
            M = cell.params["M"]
            try:
                g = torch.Generator(device=dev).manual_seed(a.seed + M)
                x = (torch.randn(M, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
                ok = True
                if not a.warm_only:
                    rows = sorted({int(round(i * (M - 1) / max(1, min(M, 24) - 1))) for i in range(min(M, 24))})
                    ys, tws, tis = arm_stock(last, x, True)
                    r_stock = race.rel_l2(ys[rows], reference(last, x, tws, tis, rows))
                    before = hmoe._ALIGN_STATS["reused"]
                    yn, twn, tin, al, bs = arm_new(last, x, True)
                    torch.cuda.synchronize()
                    reused = hmoe._ALIGN_STATS["reused"] == before + 1
                    r_new = race.rel_l2(yn[rows], reference(last, x, twn, tin, rows))
                    can = canonical(tin, bs, E + 1)
                    align_exact = all(torch.equal(u, v) for u, v in zip(al, can))
                    route_eq = float((tin[:, :TOPK].sort(1).values == tis.sort(1).values).all(1).float().mean())
                    acc = r_new <= r_stock * 1.10 + 1e-4
                    ok = bool(acc and align_exact and reused and route_eq == 1.0)
                    race.emit(t="check", cell=cell.id, arm="new", kind="block_fp64", rows_checked=len(rows),
                              rel_l2_fp64=r_new, stock_rel_l2_fp64=r_stock, rule="new <= 1.10 * stock + 1e-4",
                              accuracy_pass=bool(acc), alignment_equals_canonical=align_exact, block_size=bs,
                              alignment_reused_by_marlin=reused, routing_equal_to_stock=route_eq,
                              skinny_gate=M <= skinny_max, **{"pass": ok})
                    del ys, yn

                def run_new():
                    y = None
                    for L in layers:
                        y = None
                        y = arm_new(L, x)
                    return [y]

                def run_stock():
                    y = None
                    for L in layers:
                        y = None
                        y = arm_stock(L, x)
                    return [y]
                arms = [C.Arm("new", "new", run_new, a.layers, code=code_new),
                        C.Arm("stock", "vendor", run_stock, a.layers, code=code_stock,
                              note="repeat equality of this arm is reported, not gated")]
            except Exception as e:  # noqa: BLE001
                race.cell_failed(cell, e)
                continue
            race.run_cell(cell, arms, index, gate=ok)
            del arms, x
        hmoe._ALIGN[0] = None
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
