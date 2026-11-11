# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 4a / 4b: paged decode attention against FlashInfer XQA.

--format fp8
  new      horizon_ops.attn_decode.decode, pipelined split-KV form (pipe=1, block_N=32)
  nopipe   the same kernel without the pipeline (role new_alt)
  xqa      flashinfer.decode.xqa_batch_decode_with_kv_cache, called as vLLM calls it (64-token kernel page, HND)
  ctl      added by the harness
  All arms read the SAME rotated FP8 caches.
--format fp4   (4-bit KV cache, 288 B per token and KV head against 512 B)
  new      horizon_ops.attn_decode_fp4_v2.decode, NST 4, 4 CTAs per SM
  nst3     NST 3 (role new_alt);  cps2: 2 CTAs per SM (role new_alt)
  xqa      XQA on FP8 caches of the same geometry (a comparison across cache formats)
  ctl      added by the harness
  The 4-bit caches are written by the 4-bit cache-write kernel from seeded random K / V, so every byte is a valid
  code. The separate key-rotation launch that one variant of the format needs is not part of the timed call.
Cells: 1 / 8 / 32 requests x 1k, 4k, 16k, 32k, 128k tokens of context (requests x context <= 1.1 M) x 4 or 1 query
tokens: the full sweep, 28 cells per format.

Correctness per cell before timing, on the last rotated cache of each format: fp64 attention over the same
(dequantised) cache for every query row, online softmax in 4096-token chunks, relative L2 <= 0.0025 for our arms; the
vendor's error against the same limit is recorded and does not gate; non-finite count; repeat equality; graph replay
after poisoning == eager.

usage: race_attn_decode.py --out FILE --format fp8|fp4 [--pkg hzkv --pkg-dir DIR] [--reverse] [--dry-run]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

FIXED_MIB = 1300            # context, the 512 MiB vendor workspace, reference transients
PS, D = 64, 256
LIMIT = 0.0025
CTXS = (1024, 4096, 16384, 32768, 131072)
ENERGY = {(1, 1024, 1), (8, 16384, 4), (32, 32768, 4), (1, 131072, 1)}
GEOM = {"dense": (4, 6), "moe": (2, 8)}


def geometry(a, B, L):
    KVH = GEOM[a.geometry][0]
    NP = B * math.ceil(L / PS) + 3
    b8 = NP * KVH * PS * 2 * D
    b4 = NP * KVH * PS * 2 * (D // 2 + D // 16)
    n8 = C.copies_for(a.ws_mib, b8, minimum=1) if b8 > a.ws_mib * 2**20 else C.copies_for(a.ws_mib, b8)
    n4 = C.copies_for(a.ws_mib, b4, minimum=1) if b4 > a.ws_mib * 2**20 else C.copies_for(a.ws_mib, b4)
    return NP, b8, n8, b4, n4


def build_cells(a):
    cells = []
    KVH = GEOM[a.geometry][0]
    for B in (1, 8, 32):
        for L in CTXS:
            if B * L > 1_100_000:
                continue
            for QL in (4, 1):
                NP, b8, n8, b4, n4 = geometry(a, B, L)
                by = n8 * b8 + (n4 * b4 if a.format == "fp4" else 0) + 300 * 2**20
                floor = B * L * KVH * D * 2 / 1.63e6
                us8, us4 = floor * 1.3 + 15, floor * 0.5625 * 1.3 + 15
                if a.format == "fp8":
                    arms = [(us8 * 2, n8), (us8 * 2, n8), (us8, n8), (us8 * 2, n8)]
                else:
                    arms = [(us4, n4)] * 4 + [(us8, n8)]
                energy = (B, L, QL) in ENERGY
                setup = 4.0 + B * L / 1.5e5 * (n8 + (4 * n4 if a.format == "fp4" else 0))
                est = C.est_seconds(a, arms, setup, energy, 3)
                cells.append(C.Cell(f"B{B}_L{L}_Q{QL}", {"B": B, "ctx": L, "QL": QL, "geometry": a.geometry,
                                                         "copies_fp8": n8, "copies_fp4": n4 if a.format == "fp4" else 0,
                                                         "bytes_fp8": b8, "bytes_fp4": b4}, by / 2**20, est, energy))
    return cells


def reference(torch, q, gather, sl, QL, KVH, G, scale_k, scale_v, step=4096):
    """fp64 causal decode attention with an online softmax over `step`-token chunks.
    gather(b, c0, c1) -> K, V as (c1 - c0, KVH, D) fp64 for request b. Row t of a request of length L sees
    positions < L - QL + t + 1."""
    B, dev = sl.shape[0], q.device
    out = torch.empty(q.shape, dtype=torch.float64, device=dev)
    for b in range(B):
        L = int(sl[b])
        qq = q[b * QL:(b + 1) * QL].double().view(QL, KVH, G, D)
        lims = torch.tensor([L - QL + t + 1 for t in range(QL)], device=dev).view(QL, 1, 1, 1)
        m = torch.full((QL, KVH, G), float("-inf"), dtype=torch.float64, device=dev)
        den = torch.zeros((QL, KVH, G), dtype=torch.float64, device=dev)
        acc = torch.zeros((QL, KVH, G, D), dtype=torch.float64, device=dev)
        for c0 in range(0, L, step):
            c1 = min(L, c0 + step)
            K, V = gather(b, c0, c1)
            s = torch.einsum("thgd,chd->thgc", qq, K) * scale_k
            pos = torch.arange(c0, c1, device=dev).view(1, 1, 1, -1)
            s = s.masked_fill(pos >= lims, float("-inf"))
            nm = torch.maximum(m, s.amax(-1))
            w = torch.exp(s - nm.unsqueeze(-1))
            f = torch.exp(m - nm)
            acc = acc * f.unsqueeze(-1) + torch.einsum("thgc,chd->thgd", w, V)
            den = den * f + w.sum(-1)
            m = nm
        out[b * QL:(b + 1) * QL] = (acc / den.unsqueeze(-1) * scale_v).reshape(QL, KVH * G, D)
    return out


def main():
    p = C.base_parser(__doc__, mem_budget_mib=5120)
    p.add_argument("--format", required=True, choices=("fp8", "fp4"))
    p.add_argument("--geometry", default="dense", choices=tuple(GEOM))
    p.add_argument("--pkg", default="tilelang.contrib.horizon_ops", help="package holding the attention modules")
    a = p.parse_args()
    target = "attn_" + a.format
    race = C.Race(a, target, FIXED_MIB, f"paged decode attention ({a.format} cache) vs FlashInfer XQA, "
                  f"{a.geometry} geometry")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    torch = race.start({"format": a.format, "geometry": a.geometry, "page": PS, "limit_rel_l2": LIMIT})
    import importlib
    import flashinfer
    import flashinfer.decode as fid
    from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask
    A8 = importlib.import_module(a.pkg + ".attn_decode")
    if a.format == "fp4":
        V2 = importlib.import_module(a.pkg + ".attn_decode_fp4_v2")
        KW = importlib.import_module(a.pkg + ".kv_write_fp4")
        R = importlib.import_module(a.pkg + ".kv_fp4_ref")
    dev = torch.device("cuda")
    KVH, G = GEOM[a.geometry]
    HQ = KVH * G
    SCALE = D ** -0.5
    KS, VS = (0.05, 0.05) if a.format == "fp8" else (1.0, 1.0)
    WS = torch.zeros(512 * 2**20, dtype=torch.uint8, device=dev)
    inc = os.path.join(os.path.dirname(flashinfer.__file__), "data", "include")
    race.emit(t="vendor_source", flashinfer_include_tree_sha=C.tree_sha(inc),
              note="hash of the vendor's bundled kernel sources; the kernel is built from them at first use")
    code_xqa = [fid, "so:flashinfer", "so:xqa"]

    for index, cell in enumerate(cells):
        if race.over_time(cell):
            break
        B, L, QL = cell.params["B"], cell.params["ctx"], cell.params["QL"]
        try:
            NP, b8, n8, b4, n4 = geometry(a, B, L)
            NPB = math.ceil(L / PS)
            gen = torch.Generator(device=dev).manual_seed(a.seed + 31 * B + L + QL)
            perm = torch.randperm(NP - 1, device=dev, generator=gen) + 1
            bt = perm[:B * NPB].view(B, NPB).int().contiguous()
            sl = torch.full((B,), L, dtype=torch.int32, device=dev)
            q = torch.randn(B * QL, HQ, D, device=dev, generator=gen).to(torch.bfloat16)
            o = torch.empty_like(q)
            mask = (_make_xqa_draft_block_mask(QL, True, dev).unsqueeze(0).expand(B, -1, -1).contiguous()
                    if QL > 1 else None)
            caches8 = []
            for _ in range(n8):
                kv = torch.empty(NP, KVH, PS, 2 * D, device=dev, dtype=torch.float8_e4m3fn)
                for p0 in range(0, NP, 256):
                    n = min(256, NP - p0)
                    kv[p0:p0 + n] = (torch.randn(n, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(
                        torch.float8_e4m3fn)
                caches8.append((kv[..., :D], kv[..., D:]))

            def gather8(b, c0, c1):
                kc, vc = caches8[-1]
                pg = bt[b, c0 // PS: math.ceil(c1 / PS)].long()
                K = kc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:c1 - c0].double()
                V = vc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:c1 - c0].double()
                return K, V

            def run_xqa():
                for kc, vc in caches8:
                    fid.xqa_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace_buffer=WS,
                                                       block_tables=bt, seq_lens=sl, max_seq_len=L,
                                                       bmm1_scale=SCALE * KS, bmm2_scale=VS, window_left=-1, out=o,
                                                       sinks=None, kv_layout="HND", q_len_per_req=QL, mask=mask)
                return [o]

            ref8 = None if a.warm_only else reference(torch, q, gather8, sl, QL, KVH, G, SCALE * KS, VS)
            arms = []
            if a.format == "fp8":
                def run_new():
                    for kc, vc in caches8:
                        A8.decode(q, kc, vc, bt, sl, L, SCALE * KS, VS, QL, out=o, kv_layout="HND", pipe=1,
                                  block_N=32)
                    return [o]

                def run_nopipe():
                    for kc, vc in caches8:
                        A8.decode(q, kc, vc, bt, sl, L, SCALE * KS, VS, QL, out=o, kv_layout="HND", pipe=0,
                                  block_N=64)
                    return [o]
                arms = [C.Arm("new", "new", run_new, n8, ref=[ref8] if ref8 is not None else None, limit=LIMIT,
                              code=[A8]),
                        C.Arm("nopipe", "new_alt", run_nopipe, n8, ref=[ref8] if ref8 is not None else None,
                              limit=LIMIT, code=[A8], energy=False)]
            else:
                gk = torch.ones(KVH, device=dev)
                caches4 = []
                for _ in range(n4):
                    cache = R.alloc_cache(NP, PS, KVH, D, device=dev, fill=0x7F)
                    views = R.split_views(cache, KVH)
                    for b in range(B):
                        for c0 in range(0, L, 16384):
                            c1 = min(L, c0 + 16384)
                            pos = torch.arange(c0, c1, device=dev)
                            slots = bt[b, pos // PS].long() * PS + pos % PS
                            key = torch.randn(c1 - c0, KVH, D, device=dev, generator=gen).to(torch.bfloat16)
                            val = torch.randn(c1 - c0, KVH, D, device=dev, generator=gen).to(torch.bfloat16)
                            KW.write(key, val, *views, slots, gk, gk, None)
                    caches4.append(views)
                torch.cuda.synchronize()

                def gather4(b, c0, c1):
                    K, V = R.gather(caches4[-1], bt[b, c0 // PS:], c1 - c0)
                    return K.double(), V.double()

                ref4 = None if a.warm_only else reference(torch, q, gather4, sl, QL, KVH, G, SCALE, 1.0)

                def mk(cfg):
                    splits = V2.plan_splits_v2(B, KVH, L, cfg.get("ctas_per_sm", 4), cfg.get("min_chunk", 64))

                    def run():
                        for v in caches4:
                            V2.decode(q, *v, bt, sl, L, SCALE, 1.0, QL, gk, gk, out=o, splits=splits, **cfg)
                        return [o]
                    return run, splits
                for name, role, cfg in (("new", "new", dict(nst=4, ctas_per_sm=4)),
                                        ("nst3", "new_alt", dict(nst=3, ctas_per_sm=4)),
                                        ("cps2", "new_alt", dict(nst=4, ctas_per_sm=2))):
                    run, splits = mk(cfg)
                    arms.append(C.Arm(name, role, run, n4, ref=[ref4] if ref4 is not None else None, limit=LIMIT,
                                      code=[V2, KW], note=f"cfg {cfg} splits {splits}", energy=role == "new"))
            arms.append(C.Arm("xqa", "vendor", run_xqa, n8, ref=[ref8] if ref8 is not None else None, limit=LIMIT,
                              code=code_xqa, note="vendor error against the limit is recorded, it does not gate"))
        except Exception as e:  # noqa: BLE001
            race.cell_failed(cell, e)
            continue
        race.run_cell(cell, arms, index)
        del arms, caches8
        if a.format == "fp4":
            del caches4
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
