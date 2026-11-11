# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 4d: ragged prefill attention against FlashInfer FA2.

Arms per cell (one process, the same queries and the same rotated FP8 caches, 64-token kernel page, HND):
  new      attn_prefill.prefill_v3 with a prepared tile plan (default: 16 query rows per tile)
  q8w8     prefill_v3, 8 query rows, 32 keys, 8 warps, padded rows (role new_alt; skipped if the module lacks it)
  q4w4     prefill_v3, 4 query rows, 32 keys, 4 warps, padded rows (role new_alt; skipped if the module lacks it)
  fa2      flashinfer.BatchPrefillWithPagedKVCacheWrapper.run; its plan() runs before timing, like our tile plan
  ctl      added by the harness
Cells: --set short  the two shapes of the first race (1 query on 128 keys; 17 and 63 queries on 100 and 129 keys)
       --set long   graph-replay versions of the eager diagnostics and the 1k / 4k / 8k self-prefill shapes
                    (1024 on 1024, 1024 on 4096, 4096 on 4096, 8192 on 8192, 8192 on 16384); these cells use fewer
                    replays per cycle (recorded per record) because one replay takes tens of milliseconds
Correctness per cell before timing: fp64 causal attention over the same cache (last rotated copy) for every query
row up to 1024 rows, else 1024 rows spread evenly; relative L2 <= 0.0025 for our arms; the vendor's error is
recorded and does not gate; non-finite count; repeat equality; graph replay after poisoning == eager.

usage: race_prefill.py --out FILE [--set short|long] [--module-file CANDIDATE.py | --pkg PKG] [--reverse] [--dry-run]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

TARGET = "prefill"
FIXED_MIB = 1100
KVH, G, D, PS = 4, 6, 256, 64
HQ = KVH * G
LIMIT = 0.0025
SETS = {"short": [([1], [128]), ([17, 63], [100, 129])],
        "long": [([1024], [1024]), ([1024], [4096]), ([4096], [4096]), ([8192], [8192]), ([8192], [16384])]}
ENERGY = {"q1_k128", "q17+63_k100+129", "q1024_k4096"}


def cid(ql, kl):
    return "q" + "+".join(map(str, ql)) + "_k" + "+".join(map(str, kl))


def geometry(a, kl):
    NP = sum(math.ceil(x / PS) for x in kl) + 2
    by = NP * KVH * PS * 2 * D
    return NP, by, C.copies_for(a.ws_mib, by)


def build_cells(a):
    cells = []
    for ql, kl in SETS[a.set]:
        NP, by, copies = geometry(a, kl)
        work = sum(q * k for q, k in zip(ql, kl))
        us_new, us_fa2 = 20 + work * 1.46e-4, 12 + work * 0.7e-4          # rough, from the eager diagnostics
        per_replay = copies * us_new
        reps = max(3, min(a.reps, int(2.5e5 / per_replay))) if per_replay > 5000 else None
        warm = max(1, min(a.warm, (reps or a.reps) // 3)) if reps else None
        energy = cid(ql, kl) in ENERGY
        est = C.est_seconds(a, [(us_new, copies)] * 4 + [(us_fa2, copies)], 6.0 + work / 4e6, energy, 3, reps, warm)
        mib = (copies * by + by + 64 * HQ * max(kl) * 8 * 3 + max(kl) * KVH * D * 16) / 2**20 + 64
        cells.append(C.Cell(cid(ql, kl), {"qlens": ql, "kvlens": kl, "copies": copies, "set": a.set}, mib, est,
                            energy, reps, warm))
    return cells


def main():
    p = C.base_parser(__doc__, mem_budget_mib=2560)
    p.add_argument("--set", default="short", choices=tuple(SETS))
    p.add_argument("--pkg", default="tilelang.contrib.horizon_ops", help="package holding attn_prefill")
    p.add_argument("--module-file", help="candidate attn_prefill file loaded by path instead of --pkg")
    a = p.parse_args()
    race = C.Race(a, TARGET, FIXED_MIB, f"ragged prefill attention vs FlashInfer FA2, set {a.set}")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    torch = race.start({"set": a.set, "page": PS, "limit_rel_l2": LIMIT, "geometry": [KVH, G, D]})
    import importlib
    import importlib.util
    from functools import partial
    import flashinfer
    if a.module_file:
        spec = importlib.util.spec_from_file_location("p1_prefill_candidate", a.module_file)
        ap = importlib.util.module_from_spec(spec)
        sys.modules["p1_prefill_candidate"] = ap
        spec.loader.exec_module(ap)
    else:
        ap = importlib.import_module(a.pkg + ".attn_prefill")
    dev = torch.device("cuda")
    SCALE, KS, VS = D ** -0.5, 0.05, 0.05
    WS = torch.zeros(256 * 2**20, dtype=torch.uint8, device=dev)
    inc = os.path.join(os.path.dirname(flashinfer.__file__), "data", "include")
    race.emit(t="vendor_source", flashinfer_include_tree_sha=C.tree_sha(inc))

    def ref_rows(q, kc, vc, bt, qo, kvl, rows):
        """fp64 causal attention for the listed global query rows, 64 rows at a time. Returns (len(rows), HQ, D)."""
        out = torch.empty(len(rows), HQ, D, device=dev, dtype=torch.float64)
        B = kvl.shape[0]
        k = 0
        for b in range(B):
            L, q0, q1 = int(kvl[b]), int(qo[b]), int(qo[b + 1])
            mine = [r for r in rows if q0 <= r < q1]
            if not mine:
                continue
            pg = bt[b, :math.ceil(L / PS)].long()
            K = kc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * KS
            V = vc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * VS
            pos = torch.arange(L, device=dev).view(1, 1, 1, L)
            for i in range(0, len(mine), 64):
                rr = torch.tensor(mine[i:i + 64], device=dev)
                lim = (L - (q1 - q0) + (rr - q0) + 1).view(-1, 1, 1, 1)
                qq = q[rr].double().view(-1, KVH, G, D)
                s = torch.einsum("thgd,lhd->thgl", qq, K) * SCALE
                pr = torch.softmax(s.masked_fill(pos >= lim, float("-inf")), dim=-1)
                out[k:k + len(rr)] = torch.einsum("thgl,lhd->thgd", pr, V).reshape(-1, HQ, D)
                k += len(rr)
        return out

    for index, cell in enumerate(cells):
        if race.over_time(cell):
            break
        qlens, kvlens = cell.params["qlens"], cell.params["kvlens"]
        try:
            NP, by, copies = geometry(a, kvlens)
            B = len(qlens)
            gen = torch.Generator(device=dev).manual_seed(a.seed + sum(qlens) + 3 * sum(kvlens))
            pages = [math.ceil(x / PS) for x in kvlens]
            perm = (torch.randperm(NP - 1, generator=torch.Generator().manual_seed(a.seed + B)) + 1)
            bt = torch.zeros(B, max(pages), dtype=torch.int32)
            ind, o = [], 0
            for b, pc in enumerate(pages):
                bt[b, :pc] = perm[o:o + pc]
                ind += perm[o:o + pc].tolist()
                o += pc
            bt = bt.to(dev)
            q = torch.randn(sum(qlens), HQ, D, device=dev, generator=gen).to(torch.bfloat16)
            qo = torch.zeros(B + 1, dtype=torch.int32)
            qo[1:] = torch.cumsum(torch.tensor(qlens), 0)
            kvi = torch.zeros(B + 1, dtype=torch.int32)
            kvi[1:] = torch.cumsum(torch.tensor(pages), 0)
            last = torch.tensor([x - (pc - 1) * PS for x, pc in zip(kvlens, pages)], dtype=torch.int32)
            qo, kvi, last = qo.to(dev), kvi.to(dev), last.to(dev)
            kvl = torch.tensor(kvlens, dtype=torch.int32, device=dev)
            storage = [(torch.randn(NP, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
                       for _ in range(copies)]
            cache = [(x[..., :D], x[..., D:]) for x in storage]
            out = torch.empty_like(q)
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(WS, "HND")
            wrapper.plan(qo, kvi, torch.tensor(ind, dtype=torch.int32, device=dev), last, HQ, KVH, D, PS, causal=True,
                         sm_scale=SCALE, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
            total = sum(qlens)
            rows = list(range(total)) if total <= 1024 else sorted({int(i * (total - 1) / 1023) for i in range(1024)})
            ref = None if a.warm_only else ref_rows(q, *cache[-1], bt, qo, kvl, rows)
            ridx = torch.tensor(rows, device=dev)

            def mk(fn):
                def run():
                    for k, v in cache:
                        fn(k, v)
                    return [out]
                return run
            specs = [("new", "new", ap.prefill_v3, 16, {}),
                     ("q8w8", "new_alt", ap.prefill_v3, 8, dict(TQ=8, BN=32, num_warps=8, policy_qk=3, policy_pv=3,
                                                                pad_rows=True)),
                     ("q4w4", "new_alt", ap.prefill_v3, 4, dict(TQ=4, BN=32, num_warps=4, policy_qk=3, policy_pv=3,
                                                                pad_rows=True))]
            arms, gate = [], True
            runs = {}
            for name, role, fn, tq, kw in specs:
                try:
                    tiles = ap.plan_prefill_tiles(qo, tq, dev)
                    f = partial(fn, **kw) if kw else fn
                    one = (lambda k, v, f=f, tiles=tiles: f(q, k, v, qo, bt, kvl, SCALE * KS, VS, out=out,
                                                             tiles=tiles, kv_layout="HND"))
                    one(*cache[-1])
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    if role == "new":
                        raise
                    race.emit(t="arm_skipped", cell=cell.id, arm=name, error=type(e).__name__ + ": " + str(e)[:300])
                    continue
                runs[name] = (role, one)
            runs["fa2"] = ("vendor", lambda k, v: wrapper.run(q, (k, v), out=out, k_scale=KS, v_scale=VS))
            for name, (role, one) in runs.items():
                if ref is not None:
                    one(*cache[-1])
                    torch.cuda.synchronize()
                    rel = race.rel_l2(out[ridx], ref)
                    ok = rel <= LIMIT
                    if role == "new":
                        gate = bool(ok)
                    race.emit(t="check", cell=cell.id, arm=name, role=role, kind="rows_fp64", rows_checked=len(rows),
                              rows_total=total, rel_l2_fp64=rel, limit=LIMIT, **{"pass": bool(ok)},
                              gates=role == "new")
                arms.append(C.Arm(name, role, mk(one), copies, code=[ap] if role != "vendor" else
                                  [flashinfer, "so:flashinfer"], energy=role in ("new", "vendor"),
                                  note="backend " + str(getattr(wrapper, "_backend", "unknown")) if role == "vendor"
                                  else None))
        except Exception as e:  # noqa: BLE001
            race.cell_failed(cell, e)
            continue
        race.run_cell(cell, arms, index, gate=gate)
        del arms, runs, storage, cache
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
