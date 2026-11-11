# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 1: W4A16 NVFP4 decode GEMM against Marlin and BF16 cuBLAS.

Arms per cell (one process, same activation buffer, each arm reads the same seeded synthetic layers):
  new     tilelang.contrib.sm120_nvfp4.decode_regfrag_xl.linear, default (tuned or generic) configuration
  marlin  vLLM's apply_fp4_marlin_linear on the same layers, prepared by prepare_fp4_layer_for_marlin
  cublas  torch.nn.functional.linear on the BF16 dequantised weights (role vendor_alt)
  ctl     added by the harness: the new arm's kernel objects captured into a second graph
  sep     the separator alone (one tiny add after every call in every arm, as in the first race); the analysis
          prints ratios with and without it
Cells: --set tuned    5 layer shapes of the hybrid 27B frame x 32 row counts (160 cells)
       --set heldout  6 shapes outside the tuned table x 8 row counts (48 cells)
       --set smoke    1 cell
Correctness per cell before timing: fp64 product with the exactly dequantised weights of the last rotated layer,
relative L2 <= 0.0025 for every arm; non-finite count; repeat equality; graph replay after poisoning == eager.
Synthetic seeded layers only (decode_contract.synthetic_layer); no checkpoint is opened.

usage: race_w4a16.py --out FILE [--set tuned|heldout|smoke] [--reverse] [--resume FILE...] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

TARGET = "w4a16"
FIXED_MIB = 700
LIMIT = 0.0025
SHAPES = [("gdn.in_proj_qkvz", 16384, 5120), ("attn.qkv_proj", 14336, 5120), ("o/out_proj", 5120, 6144),
          ("mlp.gate_up", 34816, 5120), ("mlp.down", 5120, 17408)]
HELD = [("held.a", 4096, 4096), ("held.b", 11008, 4096), ("held.c", 4096, 11008), ("held.d", 6144, 4096),
        ("held.e", 10240, 7680), ("held.f", 20480, 2560)]
MS = [1, 4, 8, 16, 17, 24, 32, 33, 40, 48, 49, 56, 64, 72, 96, 104, 128, 144, 192, 208, 232, 256, 272, 304, 384, 416,
      512, 544, 608, 768, 832, 1024]
MS_HELD = [1, 8, 32, 64, 128, 256, 512, 1024]
ENERGY_M = (1, 32)
ENERGY_HELD = ("held.a", "held.e")


def chunk_rows(K):
    """Rows per dequantisation chunk, so the fp64 transient stays near 96 MiB for wide layers."""
    return max(64, min(2048, (96 * 2**20) // (K * 24)))


def geometry(a, N, K):
    per = N * K // 2 + N * K // 16
    copies = C.copies_for(a.ws_mib, per)
    cb = min(copies, C.copies_for(a.ws_mib, N * K * 2))
    return per, copies, cb


def build_cells(a):
    cells = []
    sets = {"tuned": [(s, MS, None) for s in SHAPES], "heldout": [(s, MS_HELD, ENERGY_HELD) for s in HELD],
            "smoke": [(SHAPES[2], [8], None)]}[a.set]
    for (name, N, K), ms, eshapes in sets:
        per, copies, cb = geometry(a, N, K)
        first = True
        for M in ms:
            by = (copies * per * 2 + 2 * per + cb * N * K * 2 + 10 * M * N * 2 + M * K * 2 + M * N * 8
                  + chunk_rows(K) * K * 24)
            us = max(per / 1.63e6, 2.0 * M * N * K / 2.5e8)          # microseconds per call, rough
            us_cb = max(N * K * 2 / 1.63e6, 2.0 * M * N * K / 2.5e8)
            energy = M in ENERGY_M and (eshapes is None or name in eshapes)
            est = C.est_seconds(a, [(us, copies)] * 3 + [(us_cb, cb), (3.0, copies)], 14.0 if first else 3.0,
                                energy, 4)
            cells.append(C.Cell(f"{N}x{K}_M{M}", {"shape": name, "N": N, "K": K, "M": M, "copies": copies,
                                                  "cublas_copies": cb, "set": a.set}, by / 2**20, est, energy))
            first = False
    return cells


def deq_rows(torch, packed, scale, alpha, r0, r1, dtype):
    """Exact dequantisation of rows r0:r1 (decode_contract layout): alpha[n] * fp4 * block scale."""
    u = packed[r0:r1].contiguous().view(torch.uint8)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=u.device, dtype=dtype)
    w = torch.empty(u.shape[0], u.shape[1] * 2, device=u.device, dtype=dtype)
    w[:, 0::2] = lut[(u & 0xF).long()]
    w[:, 1::2] = lut[(u >> 4).long()]
    return w * scale[r0:r1].float().to(dtype).repeat_interleave(16, 1) * alpha[r0:r1].to(dtype)[:, None]


def main():
    p = C.base_parser(__doc__, mem_budget_mib=4096)
    p.add_argument("--set", default="tuned", choices=("tuned", "heldout", "smoke"))
    p.add_argument("--marlin-global", default="auto", choices=("auto", "reciprocal", "checkpoint"),
                   help="what the marlin arm gets as weight_global_scale: alpha (vLLM >= 0.30) or 1 / alpha (0.29)")
    a = p.parse_args()
    race = C.Race(a, TARGET, FIXED_MIB, f"W4A16 NVFP4 decode GEMM vs Marlin and BF16 cuBLAS, set {a.set}")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    torch = race.start({"set": a.set, "limit_rel_l2": LIMIT, "separator": "one add_ after every call"})
    import types
    import torch.nn.functional as F
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rf
    from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
    from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mu

    sep = torch.zeros(1, device="cuda")
    state = {"key": None}
    # Since vLLM 0.30 the quantisation scheme hands the kernels the RECIPROCAL
    # of the checkpoint's global scale (layer.weight_global_scale = 1 / global_scale) before
    # prepare_fp4_layer_for_marlin; 0.29 passed the checkpoint value. With the 0.29 convention on 0.30 every Marlin
    # output was non-finite (rel L2 inf at all cells). The fp64 gate of the marlin arm decides, not this comment.
    from importlib import metadata
    try:
        vv = tuple(int(x) for x in metadata.version("vllm").split(".")[:2])
    except Exception:  # noqa: BLE001
        vv = (0, 30)
    conv = a.marlin_global if a.marlin_global != "auto" else ("reciprocal" if vv >= (0, 30) else "checkpoint")

    def marlin_global(alpha):
        return alpha[:1].clone() if conv == "reciprocal" else (1.0 / alpha[:1]).clone()

    def build_shape(N, K):
        state.clear()
        torch.cuda.empty_cache()
        per, copies, cb = geometry(a, N, K)
        step = chunk_rows(K)
        preps, marl, bfs, last = [], [], [], None
        for i in range(copies):
            packed, scale, alpha = synthetic_layer(N, K, seed=a.seed * 1000 + i)
            if i >= copies - cb:
                wb = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
                for r0 in range(0, N, step):
                    wb[r0:r0 + step] = deq_rows(torch, packed, scale, alpha, r0, r0 + step, torch.float32)
                bfs.append(wb)
            layer = types.SimpleNamespace(output_size_per_partition=N, input_size_per_partition=K,
                                          params_dtype=torch.bfloat16, bias=None,
                                          weight=packed.view(torch.uint8).clone(), weight_scale=scale.clone(),
                                          weight_global_scale=marlin_global(alpha))
            mu.prepare_fp4_layer_for_marlin(layer)
            marl.append(layer)
            preps.append(xl.prepare(packed, scale, alpha))
            last = (packed, scale, alpha)
        state.update(key=(N, K), preps=preps, marl=marl, bfs=bfs, last=last)

    for index, cell in enumerate(cells):
        if race.over_time(cell):
            break
        N, K, M = cell.params["N"], cell.params["K"], cell.params["M"]
        try:
            if state.get("key") != (N, K):
                build_shape(N, K)
            preps, marl, bfs, last = state["preps"], state["marl"], state["bfs"], state["last"]
            g = torch.Generator(device="cuda").manual_seed(a.seed + 7919 * M + N + K)
            x = torch.randn(M, K, device="cuda", generator=g).to(torch.bfloat16)
            ref = torch.empty(M, N, device="cuda", dtype=torch.float64)
            xd = x.double()
            step = chunk_rows(K)
            for r0 in range(0, N, step):
                ref[:, r0:r0 + step] = xd @ deq_rows(torch, *last, r0, r0 + step, torch.float64).T
            del xd

            def run_new():
                y = None
                for prep in preps:
                    y = None
                    y = xl.linear(x, prep)
                    sep.add_(1.0)
                return [y]

            def run_marlin():
                y = None
                for layer in marl:
                    y = None
                    y = mu.apply_fp4_marlin_linear(x, layer.weight, layer.weight_scale, layer.weight_global_scale,
                                                   layer.workspace, N, K)
                    sep.add_(1.0)
                return [y]

            def run_cublas():
                y = None
                for wb in bfs:
                    y = None
                    y = F.linear(x, wb)
                    sep.add_(1.0)
                return [y]

            def run_sep():
                for _ in preps:
                    sep.add_(1.0)
                return [sep]

            arms = [C.Arm("new", "new", run_new, len(preps), ref=[ref], limit=LIMIT, code=[xl, rf],
                          note="config " + str(xl.config(N, K, M))),
                    C.Arm("marlin", "vendor", run_marlin, len(marl), ref=[ref], limit=LIMIT, code=[mu, "so:_C."]),
                    C.Arm("cublas", "vendor_alt", run_cublas, len(bfs), ref=[ref], limit=LIMIT,
                          code=["so:libcublas"], note="BF16 weights are the rounded dequantised weights"),
                    C.Arm("sep", "separator", run_sep, len(preps), energy=False)]
        except Exception as e:  # noqa: BLE001
            race.cell_failed(cell, e)
            continue
        race.run_cell(cell, arms, index)
        del arms, ref, x
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
