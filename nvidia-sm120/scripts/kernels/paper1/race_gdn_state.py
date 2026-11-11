# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Target 2: gated-delta recurrent-state decode kernels against vLLM's CUDA kernel.

The gates below decide correctness: a block whose gates fail reports gate_pass false in every cell.

--variant state16 (default)
  new     gdn_state.fused_gdn_decode_post_conv_mtp with a float16 state
  vendor  vLLM fused_gdn_decode_post_conv_mtp with its float32 state (vLLM's default)
  vbf16   the same vLLM kernel with a bfloat16 state (role vendor_alt: the part of the gain that needs no new kernel)
  ctl     added by the harness
--variant ring
  new     gdn_state.replay_step: one checkpoint (--ring-dtype) plus the replay ring
  vendor, vbf16, ctl as above
Cells: requests 1 / 8 / 32 x 2 or 4 tokens each (the six shapes of the first run).

Correctness, before timing:
  cell "gates"   the complete gate sections of the first run's test (window: float32 and bfloat16 bit-identical to
                 vLLM, output and every state slot; window_fp16: float16 against our float32 kernel plus
                 round-to-nearest-even / saturation; replay_chain: ring against chained vLLM; graph: determinism and
                 replay == eager), executed from that file unchanged (--gate-module) so the criterion is the same.
  every cell     on fresh state of that exact shape: our float32 path against vLLM bit for bit; state16: float16
                 against float32 plus rounding; two runs from identical fresh state bit-equal; non-finite count.
Inputs are synthetic and seeded. The three per-layer parameter vectors are synthetic unless --params-dir names a
checkpoint directory, which is then opened read-only for those three small tensors of one layer.

usage: race_gdn_state.py --out FILE --pkg-dir DIR [--variant state16|ring] [--reverse] [--resume FILE...] [--dry-run]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
import p1_common as C  # noqa: E402

FIXED_MIB = 800
HV, H, DK, DV, W = 48, 16, 128, 128, 4
SLOT = 819200
S32 = HV * DV * DK * 4
SHAPES = [(1, 2), (1, 4), (8, 2), (8, 4), (32, 2), (32, 4)]


def geometry(a, nr):
    per32 = (nr * W + 1) * SLOT * 4
    n32 = C.copies_for(a.ws_mib, per32)
    n16 = C.copies_for(a.ws_mib, per32 // 2)
    half = 16
    isz = {"float32": 4, "bfloat16": 2, "float16": 2}[a.ring_dtype]
    perg = (nr + 1) * SLOT * isz + nr * 2 * half * (HV * DV + H * DK + HV) * 4
    ng = C.copies_for(a.ws_mib, perg)
    return per32, n32, n16, perg, ng


def build_cells(a):
    cells = []
    if not a.skip_gates:
        cells.append(C.Cell("gates", {"kind": "gates"}, 2600, 150.0, False))
    for nr, ql in SHAPES:
        per32, n32, n16, perg, ng = geometry(a, nr)
        rot = n32 * per32 + n16 * (per32 // 2) + (ng * perg if a.variant == "ring" else n16 * (per32 // 2))
        by = max(rot, int(3.5 * per32)) + 64 * 2**20       # the fresh-state checks are freed before the rotation
        us32 = nr * (1 + ql) * S32 / 1.63e6 * 1.2
        newc = ng if a.variant == "ring" else n16
        est = C.est_seconds(a, [(us32, n32), (us32 / 2, n16), (us32 / 2, newc), (us32 / 2, newc)], 12.0, True, 4)
        cells.append(C.Cell(f"nr{nr}_ql{ql}", {"nr": nr, "ql": ql, "variant": a.variant, "layers_f32": n32,
                                               "layers_16": n16, "layers_ring": ng}, by / 2**20, est, True))
    return cells


def main():
    p = C.base_parser(__doc__, mem_budget_mib=3584)
    p.add_argument("--variant", default="state16", choices=("state16", "ring"))
    p.add_argument("--ring-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--gate-module", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gdnstate",
                                                         "test_gdn_state_gpu.py"))
    p.add_argument("--skip-gates", action="store_true", help="only with --resume of a block whose gates passed")
    p.add_argument("--params-dir", help="checkpoint directory, read-only, for A_log / dt_bias / norm.weight")
    p.add_argument("--params-layer", type=int, default=0)
    a = p.parse_args()
    target = "gdn_" + a.variant
    race = C.Race(a, target, FIXED_MIB, f"gated-delta decode state ({a.variant}) vs vLLM's CUDA kernel")
    cells = race.plan(build_cells(a))
    if cells is None:
        return 0
    if not a.pkg_dir:
        raise SystemExit("--pkg-dir with the gated-delta state kernel package (hzgs/) is required")
    torch = race.start({"variant": a.variant, "ring_dtype": a.ring_dtype, "params": "checkpoint" if a.params_dir
                        else "synthetic"})
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("p1_gdn_gates", a.gate_module)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    from vllm import _custom_ops as ops
    from hzgs import gdn_state as GS
    try:
        from hzgs import gdn_decode as GD
    except Exception:  # noqa: BLE001
        GD = None
    dev = torch.device("cuda")
    T.ops, T.GS, T.dev, T.QUICK, T.OUT = ops, GS, dev, False, None
    T.log = lambda rec: race.emit(t="gate_detail", **rec)      # the gate file's records go through the same masking

    if a.params_dir:
        from safetensors import safe_open
        idx = json.load(open(os.path.join(a.params_dir, "model.safetensors.index.json")))["weight_map"]
        pre = f"model.language_model.layers.{a.params_layer}.linear_attn."
        with safe_open(os.path.join(a.params_dir, idx[pre + "A_log"]), "pt") as f:
            P = {"A_log": f.get_tensor(pre + "A_log").float().to(dev).contiguous(),
                 "dt_bias": f.get_tensor(pre + "dt_bias").to(torch.bfloat16).to(dev).contiguous(),
                 "norm_w": f.get_tensor(pre + "norm.weight").to(torch.bfloat16).to(dev).contiguous()}
    else:
        g = torch.Generator(device=dev).manual_seed(a.seed)
        P = {"A_log": (torch.rand(HV, device=dev, generator=g) * 4 - 4).contiguous(),
             "dt_bias": torch.randn(HV, device=dev, generator=g).to(torch.bfloat16).contiguous(),
             "norm_w": (1 + 0.1 * torch.randn(DV, device=dev, generator=g)).to(torch.bfloat16).contiguous()}
    code_new = [GS] + ([GD] if GD is not None else [])
    code_vendor = [ops, "so:_C."]
    gates_ok = None
    for fn in a.resume:                      # gates passed in an earlier block of this job?
        try:
            for ln in open(fn):
                if '"gate_summary"' in ln and json.loads(ln).get("pass") is True:
                    gates_ok = True
        except OSError:
            pass

    def cell_checks(cell, nr, ql):
        """Fresh-state exactness at this shape. Returns True if every comparison is exact."""
        gen = torch.Generator(device=dev).manual_seed(a.seed + 100 * nr + ql)
        mixed, aa, bb, z, cu = T.make_inputs([ql] * nr, gen)
        sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev).contiguous()
        nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
        L = mixed.shape[0]
        ok = True

        def out():
            return torch.full((L, HV, DV), 7.0, device=dev, dtype=torch.bfloat16)
        st_v, flat_v = T.make_state(nr * W + 1, torch.float32, gen)
        flat_o = flat_v.clone()
        o_v, o_o = out(), out()
        T.vllm_call(mixed, aa, bb, P, sidx, cu, nacc, st_v, z, o_v)
        T.g1s_call(mixed, aa, bb, P, sidx, cu, nacc, T.view_like(flat_o, st_v), z, o_o)
        torch.cuda.synchronize()
        d = {"out_diff": T.bits_diff(o_v, o_o), "state_diff": T.bits_diff(flat_v, flat_o)}
        ok = ok and d["out_diff"] == 0 and d["state_diff"] == 0
        race.emit(t="check", cell=cell.id, arm="new_f32_vs_vendor_f32", kind="bit_identity", **d,
                  nonfinite=race.nonfinite([o_o, flat_o]), **{"pass": d["out_diff"] == 0 and d["state_diff"] == 0})
        del st_v, flat_v, flat_o
        st16, flat16 = T.make_state(nr * W + 1, torch.float16, gen)
        flat16b = flat16.clone()
        flat32 = flat16.float()
        o16, o16b, o32 = out(), out(), out()
        T.g1s_call(mixed, aa, bb, P, sidx, cu, nacc, st16, z, o16)
        T.g1s_call(mixed, aa, bb, P, sidx, cu, nacc, T.view_like(flat16b, st16), z, o16b)
        T.g1s_call(mixed, aa, bb, P, sidx, cu, nacc, T.view_like(flat32, st16), z, o32)
        torch.cuda.synchronize()
        d = {"out_diff": T.bits_diff(o16, o32), "state_diff": T.bits_diff(T.store_round(flat32, torch.float16), flat16),
             "repeat_out_diff": T.bits_diff(o16, o16b), "repeat_state_diff": T.bits_diff(flat16, flat16b)}
        good = not any(d.values())
        ok = ok and good
        race.emit(t="check", cell=cell.id, arm="new_f16_vs_f32_rounded", kind="bit_identity", **d,
                  nonfinite=race.nonfinite([o16, flat16]), **{"pass": good})
        del st16, flat16, flat16b, flat32
        if a.variant == "ring":
            dt = getattr(torch, a.ring_dtype)
            cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
            ridx = torch.arange(nr, dtype=torch.int32, device=dev)
            ff = torch.zeros(nr, dtype=torch.int32, device=dev)
            res = []
            base = T.make_state(nr + 1, dt, gen)
            for _ in range(2):
                flat = base[1].clone()
                ring = GS.ring_buffers(nr, HV, H, 16, dev)
                o = out()
                T.g3_call(mixed, aa, bb, P, cidx, ridx, cu, nacc, ff, T.view_like(flat, base[0]), ring, z, o)
                torch.cuda.synchronize()
                res.append((o, flat) + tuple(ring))
            d = {"repeat_diff": sum(abs(T.bits_diff(x, y)) for x, y in zip(*res))}
            ok = ok and d["repeat_diff"] == 0
            race.emit(t="check", cell=cell.id, arm="ring_fresh_repeat", kind="repeat_equality", **d,
                      nonfinite=race.nonfinite(list(res[0][:2])), **{"pass": d["repeat_diff"] == 0},
                      note="bit-identity of the ring against chained vLLM is the replay_chain section of the gates")
        return ok

    for index, cell in enumerate(cells):
        if race.over_time(cell):
            break
        if cell.id == "gates":
            try:
                bad_w = T.section_window(P)
                bad_f = T.section_fp16(P)
                bad_r = T.section_replay(P)
                ok_g = T.section_graph(P)
                gates_ok = bad_w == 0 and bad_f == 0 and bad_r == 0 and bool(ok_g)
                race.emit(t="gate_summary", cell="gates", window_bad=bad_w, fp16_bad=bad_f, replay_bad=bad_r,
                          graph_ok=bool(ok_g), **{"pass": gates_ok},
                          gate_module_sha=C.file_sha(a.gate_module), code=C.code_fingerprint(code_new + code_vendor))
                race.emit(t="warm_done" if a.warm_only else "cell_done", cell="gates", gate_pass=gates_ok)
                race.done_cells.append("gates")
            except Exception as e:  # noqa: BLE001
                gates_ok = False
                race.cell_failed(cell, e)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            continue
        nr, ql = cell.params["nr"], cell.params["ql"]
        try:
            ok = True if a.warm_only else cell_checks(cell, nr, ql)
            torch.cuda.empty_cache()
            per32, n32, n16, perg, ng = geometry(a, nr)
            gen = torch.Generator(device=dev).manual_seed(a.seed + 7 * nr + ql)
            mixed, aa, bb, z, cu = T.make_inputs([ql] * nr, gen)
            o = torch.empty(mixed.shape[0], HV, DV, device=dev, dtype=torch.bfloat16)
            nacc = torch.tensor([(i % ql) + 1 for i in range(nr)], dtype=torch.int32, device=dev)
            sidx = (torch.arange(nr * W, dtype=torch.int32) + 1).view(nr, W).to(dev)
            s32 = [T.make_state(nr * W + 1, torch.float32, gen)[0] for _ in range(n32)]
            sbf = [T.make_state(nr * W + 1, torch.bfloat16, gen)[0] for _ in range(n16)]

            def run_vendor():
                for s in s32:
                    T.vllm_call(mixed, aa, bb, P, sidx, cu, nacc, s, z, o)
                return [o]

            def run_vbf16():
                for s in sbf:
                    T.vllm_call(mixed, aa, bb, P, sidx, cu, nacc, s, z, o)
                return [o]

            if a.variant == "state16":
                s16 = [T.make_state(nr * W + 1, torch.float16, gen)[0] for _ in range(n16)]

                def run_new():
                    for s in s16:
                        T.g1s_call(mixed, aa, bb, P, sidx, cu, nacc, s, z, o)
                    return [o]
                ncalls = n16
            else:
                dt = getattr(torch, a.ring_dtype)
                cidx = torch.arange(1, nr + 1, dtype=torch.int32, device=dev)
                ridx = torch.arange(nr, dtype=torch.int32, device=dev)
                ff = torch.zeros(nr, dtype=torch.int32, device=dev)
                sg = [(T.make_state(nr + 1, dt, gen)[0], GS.ring_buffers(nr, HV, H, 16, dev)) for _ in range(ng)]

                def run_new():
                    for s, ring in sg:
                        T.g3_call(mixed, aa, bb, P, cidx, ridx, cu, nacc, ff, s, ring, z, o)
                    return [o]
                ncalls = ng
                if not a.warm_only:
                    for _ in range(3):          # bring the rings to a steady state, as in the first run
                        run_new()
            arms = [C.Arm("new", "new", run_new, ncalls, stateful=True, code=code_new),
                    C.Arm("vendor", "vendor", run_vendor, n32, stateful=True, code=code_vendor),
                    C.Arm("vbf16", "vendor_alt", run_vbf16, n16, stateful=True, code=code_vendor,
                          note="vendor kernel, bfloat16 state")]
        except Exception as e:  # noqa: BLE001
            race.cell_failed(cell, e)
            continue
        race.run_cell(cell, arms, index, gate=bool(ok) and gates_ok is True)
        del arms
    return race.finish(cells)


if __name__ == "__main__":
    raise SystemExit(main())
