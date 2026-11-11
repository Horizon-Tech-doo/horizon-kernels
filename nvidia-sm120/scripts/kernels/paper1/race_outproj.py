# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 4c: FP8-weight output projection (W8A16) against the BF16 GEMM through PyTorch / cuBLAS.

Arms per cell (one process, the same activation rows):
  new     tilelang.contrib.sm120_fp8.decode_w8a16.linear, DEFAULT configuration for the row count (the first race
          swept seven configurations per row count and reported the fastest eligible one; no selection here)
  cublas  x @ W^T with the BF16 weights (the output head as vLLM runs it)
  ctl     added by the harness
  sep     the separator alone (one tiny add after every call in every arm)
Cells: 1, 4, 8, 16, 17, 32, 33, 64, 65, 128 rows of a 248,320 x 5,120 head (--vocab changes the row count of the
weight for a memory-reduced run; that is then NOT the paper geometry and is recorded).

Weights: synthetic seeded BF16 by default; --bf16-file opens a safetensors file read-only and takes --tensor from
it (a public checkpoint's output head). The FP8 weights are computed here from the BF16 weights, per row,
scale = max|w| / 448. Activations are always synthetic, so the top-1 agreement printed here is on random
activations and is context only.

Correctness per cell before timing: new against the fp64 product with the dequantised FP8 weights, cublas against
the fp64 product with the BF16 weights, relative L2 <= 0.0025; non-finite count; repeat equality; graph replay ==
eager.

usage: race_outproj.py --out FILE [--bf16-file F --tensor lm_head.weight] [--vocab N] [--reverse] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

TARGET = "outproj"
FIXED_MIB = 700
LIMIT = 0.0025
K = 5120
ROWS = [1, 4, 8, 16, 17, 32, 33, 64, 65, 128]
ENERGY_M = (1, 32, 128)
US = {1: (806, 1586), 4: (800, 1575), 8: (805, 1580), 16: (810, 1590), 17: (815, 1600), 32: (830, 1630),
      33: (950, 1640), 64: (960, 1660), 65: (1600, 1750), 128: (1626, 1781)}   # first race, for the time estimate


def geometry(a):
    N = a.vocab
    cb = a.copies or (C.copies_for(a.ws_mib, N * K * 2, 1) if N * K * 2 > a.ws_mib * 2**20 else
                      C.copies_for(a.ws_mib, N * K * 2))
    cq = a.copies or (C.copies_for(a.ws_mib, N * K, 1) if N * K > a.ws_mib * 2**20 else C.copies_for(a.ws_mib, N * K))
    return N, cb, cq


def build_cells(a):
    N, cb, cq = geometry(a)
    scale = N / 248320
    cells = []
    for i, M in enumerate(ROWS):
        by = cb * N * K * 2 + cq * N * K + N * 8 + 2 * M * N * 8 + 8 * M * N * 2 + 8192 * K * 16 + 128 * 2**20
        energy = M in ENERGY_M
        est = C.est_seconds(a, [(US[M][0] * scale, cq)] * 2 + [(US[M][1] * scale, cb), (3.0, cq)],
                            (40.0 * scale * (cb + cq) if i == 0 else 0) + 4.0, energy, 3)
        cells.append(C.Cell(f"M{M}", {"M": M, "N": N, "K": K, "copies_bf16": cb, "copies_fp8": cq,
                                      "paper_geometry": N == 248320}, by / 2**20, est, energy))
    return cells


def main():
    p = C.base_parser(__doc__, mem_budget_mib=5632)
    p.add_argument("--vocab", type=int, default=248320, help="rows of the weight (multiple of 128)")
    p.add_argument("--copies", type=int, default=0, help="rotated copies per format (0: from --ws-mib, at least 1)")
    p.add_argument("--bf16-file", help="safetensors file, opened read-only, with the BF16 output head")
    p.add_argument("--tensor", default="lm_head.weight")
    a = p.parse_args()
    if a.vocab % 128:
        p.error("--vocab must be a multiple of 128")
    race = C.Race(a, TARGET, FIXED_MIB, "W8A16 FP8 output projection vs BF16 cuBLAS GEMM")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    torch = race.start({"N": a.vocab, "K": K, "weights": "file" if a.bf16_file else "synthetic",
                        "activations": "synthetic", "limit_rel_l2": LIMIT, "fp8_config": "default, no selection"})
    from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
    dev = "cuda"
    N, cb, cq = geometry(a)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    if a.bf16_file:
        from safetensors import safe_open
        with safe_open(a.bf16_file, framework="pt", device="cpu") as f:
            W = f.get_tensor(a.tensor)
        assert W.dtype == torch.bfloat16 and tuple(W.shape) == (N, K), (W.dtype, tuple(W.shape))
        W = W.to(dev)
    else:
        W = torch.empty(N, K, device=dev, dtype=torch.bfloat16)
        for r0 in range(0, N, 8192):
            n = min(8192, N - r0)
            W[r0:r0 + n] = (torch.randn(n, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
    Q = torch.empty(N, K, device=dev, dtype=torch.float8_e4m3fn)
    S = torch.empty(N, device=dev, dtype=torch.float32)
    for r0 in range(0, N, 8192):
        w = W[r0:r0 + 8192].float()
        sc = w.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
        Q[r0:r0 + 8192] = (w / sc).to(torch.float8_e4m3fn)
        S[r0:r0 + 8192] = sc.reshape(-1)
    bfs = [W.clone() for _ in range(cb - 1)] + [W]          # the reference uses the last rotated copy
    qws = [Q.clone() for _ in range(cq - 1)] + [Q]
    preps = [w8.prepare(q, S) for q in qws]
    sep = torch.zeros(1, device=dev)
    hmax = max(c.params["M"] for c in cells)
    X = torch.randn(hmax, K, device=dev, generator=g).to(torch.bfloat16)

    for index, cell in enumerate(cells):
        if race.over_time(cell):
            break
        M = cell.params["M"]
        try:
            x = X[:M].contiguous()
            ref8 = refb = None
            if not a.warm_only:
                ref8 = torch.empty(M, N, device=dev, dtype=torch.float64)
                refb = torch.empty(M, N, device=dev, dtype=torch.float64)
                xd = x.double()
                for r0 in range(0, N, 8192):
                    r1 = min(N, r0 + 8192)
                    ref8[:, r0:r1] = xd @ (Q[r0:r1].float().double() * S[r0:r1, None].double()).T
                    refb[:, r0:r1] = xd @ W[r0:r1].double().T
                top = float((ref8.argmax(-1) == refb.argmax(-1)).float().mean())
                race.emit(t="check", cell=cell.id, arm="new", kind="top1_fp8_vs_bf16_fp64", top1_agreement=top,
                          note="synthetic activations: context only", **{"pass": True})
                del xd

            def run_new():
                y = None
                for prep in preps:
                    y = None
                    y = w8.linear(x, prep)
                    sep.add_(1.0)
                return [y]

            def run_cublas():
                y = None
                for wb in bfs:
                    y = None
                    y = x @ wb.T
                    sep.add_(1.0)
                return [y]

            def run_sep():
                for _ in preps:
                    sep.add_(1.0)
                return [sep]
            arms = [C.Arm("new", "new", run_new, cq, ref=[ref8] if ref8 is not None else None, limit=LIMIT,
                          code=[w8], note="config " + str(w8.config(N, K, M))),
                    C.Arm("cublas", "vendor", run_cublas, cb, ref=[refb] if refb is not None else None, limit=LIMIT,
                          code=["so:libcublas"]),
                    C.Arm("sep", "separator", run_sep, cq, energy=False)]
        except Exception as e:  # noqa: BLE001
            race.cell_failed(cell, e)
            continue
        race.run_cell(cell, arms, index)
        del arms, ref8, refb
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
