#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU unit tests for mtp_train_v4.py (served-numerics MTP retrain). No GPU, no vLLM, nothing written outside a temp dir.
#   CUDA_VISIBLE_DEVICES="" python test_mtp_train_v4.py -v
# The pure numerics tests need only torch. The mtp_quant cross-check imports the repo's horizon_kernels/mtp_quant.py
# by path (torch only at import). Tiny-model tests need transformers with qwen3_5 and run off the test server (gpushare
# aborts a GPU-less HF forward / backward there, see test_mtp_train_v2.py).
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402

torch.set_num_threads(int(os.environ.get("TEST_THREADS", "2")))
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    torch.utils.checkpoint.DefaultDeviceType.set_device_type("cpu")
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtp_train_v2 as V2  # noqa: E402
import mtp_train_v4 as M  # noqa: E402
from test_mtp_train_v2 import GPUSHARE_CPU, HAS_QWEN, needs_backward, tiny_cfg  # noqa: E402

if HAS_QWEN:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as QM

MTP_QUANT_PY = os.environ.get("MTP_QUANT_PY") or os.path.normpath(os.path.join(HERE, "..", "..", "horizon_kernels", "mtp_quant.py"))


def _load_mtp_quant():
    if not os.path.exists(MTP_QUANT_PY):
        return None
    spec = importlib.util.spec_from_file_location("hk_mtp_quant", MTP_QUANT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MQ = _load_mtp_quant()


# ---------------------------------------------------------------------------------------------------------------------
# brute-force references
def ref_e2m1(v):
    """nearest e2m1 value, ties to the even code, saturating."""
    pos = M.E2M1_POS
    m = min(abs(v), 6.0)
    best = min(range(8), key=lambda i: (abs(pos[i] - m), i % 2))     # equal distance -> even code first
    return -pos[best] if v < 0 else pos[best]


E4M3_VALUES = sorted(set(torch.arange(0, 127, dtype=torch.uint8).view(torch.float8_e4m3fn).float().tolist()))


def ref_ue4m3_ceil(x):
    if x <= 0:
        return 0.0
    if x >= 448.0:
        return 448.0
    return min(v for v in E4M3_VALUES if v >= x)


def ref_fp4x2_row(row, passes=2):
    """scalar loop re-implementation of nvfp4_quant_rs_kernel + the GEMM's dequantisation, one row, fp32 steps."""
    f32 = lambda v: torch.tensor(v, dtype=torch.float32)  # noqa: E731
    x = [f32(float(v)) for v in row]
    amax = max(abs(float(v)) for v in x)
    if amax > 0:
        rs, rsinv = f32(M.ROW_TARGET) / f32(amax), f32(amax) / f32(M.ROW_TARGET)
    else:
        rs, rsinv = f32(1.0), f32(1.0)
    out = []
    for g0 in range(0, len(x), 16):
        r = [v * rs for v in x[g0:g0 + 16]]
        acc = [f32(0.0)] * 16
        for _ in range(passes):
            am = max(abs(float(v)) for v in r)
            sc = f32(ref_ue4m3_ceil(float(f32(am) / f32(6.0))))
            inv = f32(1.0) / sc if float(sc) > 0 else f32(0.0)
            q = [f32(ref_e2m1(float(v * inv))) * sc for v in r]
            acc = [a + b for a, b in zip(acc, q)]
            r = [a - b for a, b in zip(r, q)]
        out += [a * rsinv for a in acc]
    return torch.stack(out)


def decode_rtn(packed, s, alpha):
    """mtp_quant.quantize_nvfp4_rtn outputs -> dequantised fp32 weight (what regfrag_xl multiplies)."""
    p = packed.view(torch.uint8).to(torch.int32)
    codes = torch.stack((p & 0xF, p >> 4), -1).reshape(p.shape[0], -1)
    mag = torch.tensor(M.E2M1_POS)[codes & 7]
    val = torch.where((codes & 8) != 0, -mag, mag)
    N, K = val.shape
    return (val.view(N, K // 16, 16) * s.float().unsqueeze(-1)).view(N, K) * alpha[:, None]


# ---------------------------------------------------------------------------------------------------------------------
class TestE2M1(unittest.TestCase):
    def test_grid_ties_and_saturation(self):
        vals = [i / 16 for i in range(-160, 161)] + [6.9, -7.5, 100.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        got = M.e2m1_rtne(torch.tensor(vals, dtype=torch.float32)).tolist()
        self.assertEqual(got, [ref_e2m1(v) for v in vals])
        # the tie points round to the even code: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4
        self.assertEqual(M.e2m1_rtne(torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])).tolist(),
                         [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])


class TestUE4M3Ceil(unittest.TestCase):
    def test_against_enumeration(self):
        g = torch.Generator().manual_seed(0)
        xs = torch.cat([torch.rand(2000, generator=g) * 460, torch.rand(500, generator=g) * 2 ** -5,
                        torch.tensor(E4M3_VALUES), torch.tensor(E4M3_VALUES) * (1 + 2 ** -20),
                        torch.tensor([0.0, -1.0, 448.0, 449.0, 1e9, 2 ** -10, 2 ** -6, 7 * 2 ** -9 + 1e-6])])
        got = M.ue4m3_ceil(xs.float()).tolist()
        exp = [ref_ue4m3_ceil(float(x)) for x in xs.float()]
        self.assertEqual(got, exp)


class TestFp4x2Act(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randn(4, 64, generator=g)
        x[1] = 0.0                                          # zero row
        x[2, 5] = 40.0                                      # outlier channel
        x[3] *= 1e-3
        self.x = x.to(torch.bfloat16)

    def test_bit_exact_vs_scalar_reference(self):
        got = M.fp4x2_act_qdq(self.x)
        for i in range(self.x.shape[0]):
            self.assertTrue(torch.equal(got[i], ref_fp4x2_row(self.x[i].float())), f"row {i}")

    def test_properties(self):
        x = torch.randn(16, 256, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
        q2, q1 = M.fp4x2_act_qdq(x, 2), M.fp4x2_act_qdq(x, 1)
        e2 = float((q2 - x.float()).norm() / x.float().norm())
        e1 = float((q1 - x.float()).norm() / x.float().norm())
        self.assertLess(e2, 0.02)
        self.assertLess(e2, e1)
        self.assertTrue(torch.equal(M.fp4x2_act_qdq(x[5:6]), q2[5:6]))          # row-invariant
        self.assertTrue(torch.equal(M.fp4x2_act_qdq(self.x)[1], torch.zeros(64)))

    def test_linear(self):
        g = torch.Generator().manual_seed(3)
        x = torch.randn(10, 256, generator=g).to(torch.bfloat16)
        w = (torch.randn(128, 256, generator=g) * 0.05).to(torch.bfloat16)
        y = M.fp4x2_linear(x, w, chunk=4)
        ref = (M.fp4x2_act_qdq(x).double() @ w.double().t())
        self.assertLess(float((y.double() - ref).norm() / ref.norm()), 5e-3)     # bf16 output rounding only
        plain = torch.nn.functional.linear(x, w).double()
        d = float((y.double() - plain).norm() / plain.norm())
        self.assertGreater(d, 1e-4)                                              # it is not W4A16
        self.assertLess(d, 0.03)
        self.assertEqual(M.fp4x2_linear(x[:0], w).shape, (0, 128))


class TestNVFP4RTN(unittest.TestCase):
    @unittest.skipIf(MQ is None, "horizon_kernels/mtp_quant.py not found next to the scripts")
    def test_bit_exact_vs_mtp_quant_merged(self):
        g = torch.Generator().manual_seed(4)
        parts = [(torch.randn(128, 256, generator=g) * s).to(torch.bfloat16) for s in (0.02, 0.05, 0.01)]
        parts[1][3, :16] = 0.0                               # a zero block
        parts[2][7, 40] = 0.9                                # an outlier
        packed, s, alpha = MQ.quantize_nvfp4_rtn(torch.cat(parts))
        served = decode_rtn(packed, s, alpha)
        gs = M.nvfp4_group_gs(parts)
        mine = torch.cat([M.nvfp4_rtn_unit(p, gs, rows=48) * gs for p in parts])
        self.assertTrue(torch.equal(mine, served))
        self.assertTrue(torch.equal(gs.reshape(1), alpha[:1]))

    @unittest.skipIf(MQ is None, "horizon_kernels/mtp_quant.py not found next to the scripts")
    def test_zero_matrix(self):
        z = torch.zeros(128, 256, dtype=torch.bfloat16)
        packed, s, alpha = MQ.quantize_nvfp4_rtn(z)
        gs = M.nvfp4_group_gs([z])
        self.assertTrue(torch.equal(M.nvfp4_rtn_unit(z, gs) * gs, decode_rtn(packed, s, alpha)))
        self.assertTrue(bool((M.nvfp4_rtn_unit(z, gs) == 0).all()))

    def test_unit_exact_in_bf16(self):
        w = torch.randn(128, 256, generator=torch.Generator().manual_seed(5)) * 0.03
        u = M.nvfp4_rtn_unit(w, M.nvfp4_group_gs([w]))
        self.assertTrue(torch.equal(u.to(torch.bfloat16).float(), u))


class _FakeMTP(torch.nn.Module):
    """The MTP module's linear layout (names as in mtp_train_v2.make_mtp) at quantisable toy sizes."""
    def __init__(self, H=256):
        super().__init__()
        L = lambda o, i: torch.nn.Linear(i, o, bias=False)  # noqa: E731
        self.fc = L(H, 2 * H)
        att = torch.nn.Module(); att.q_proj, att.k_proj, att.v_proj, att.o_proj = L(2 * H, H), L(128, H), L(128, H), L(H, 2 * H)
        mlp = torch.nn.Module(); mlp.gate_proj, mlp.up_proj, mlp.down_proj = L(H, H), L(H, H), L(H, H)
        layer = torch.nn.Module(); layer.self_attn, layer.mlp = att, mlp
        self.layers = torch.nn.ModuleList([layer])


class TestQAT(unittest.TestCase):
    @needs_backward
    def test_forward_value_and_ste_gradient(self):
        torch.manual_seed(6)
        m = _FakeMTP()
        keys = sorted(m.state_dict())
        M.install_mtp_qat(m)
        self.assertEqual(sorted(m.state_dict()), keys)                  # nothing registered
        q = m.layers[0].self_attn
        gs = M.nvfp4_group_gs([q.q_proj.weight, q.k_proj.weight, q.v_proj.weight])
        self.assertTrue(torch.equal(q.k_proj._hz_gs, gs))
        x = torch.randn(5, 256, requires_grad=True)
        y = q.k_proj(x)
        ref = torch.nn.functional.linear(x, q.k_proj._hz_unit.float() * gs)
        self.assertTrue(torch.allclose(y, ref, rtol=1e-5, atol=1e-6))
        y.sum().backward()
        # straight-through: the master weight gets exactly the gradient of a plain linear, d(sum y)/dW = 1^T x
        exp = torch.ones(5, 128).t() @ x.detach()
        self.assertTrue(torch.allclose(q.k_proj.weight.grad, exp, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(x.grad, torch.ones(5, 128) @ (q.k_proj._hz_unit.float() * gs), rtol=1e-5,
                                       atol=1e-6))

    def test_refresh_follows_the_master(self):
        m = _FakeMTP()
        M.install_mtp_qat(m)
        u0 = m.fc._hz_unit.clone()
        with torch.no_grad():
            m.fc.weight.mul_(3.0)
        M.refresh_mtp_qat(m)
        self.assertFalse(torch.equal(u0, m.fc._hz_unit))

    def test_refuses_unquantisable_shapes(self):
        m = _FakeMTP()
        m.fc = torch.nn.Linear(100, 64, bias=False)
        with self.assertRaises(RuntimeError):
            M.install_mtp_qat(m)


class TestFP8(unittest.TestCase):
    def test_head_argmax_matches_brute_force(self):
        g = torch.Generator().manual_seed(7)
        W = (torch.randn(300, 64, generator=g) * 0.1).to(torch.bfloat16)
        x = torch.randn(9, 64, generator=g).to(torch.bfloat16)
        kind, head = M.head_for_eval(torch.nn.Parameter(W.float()), "fp8")
        self.assertEqual(kind, "fp8")
        got = M.fp8_head_argmax(x, head, chunk=4)
        wq, sw = M.fp8_rows(W)
        xq, sa = M.fp8_rows(x)
        ref = ((xq.double() @ wq.double().t()) * sa.double()[:, None] * sw.double()[None, :]).argmax(-1)
        self.assertTrue(torch.equal(got, ref))
        self.assertTrue(torch.equal(sa, x.float().abs().amax(-1).clamp(min=1e-8) / 448.0))

    @needs_backward
    def test_ste_and_qdq(self):
        x = torch.randn(3, 16, requires_grad=True)
        y = M.fp8_rows_ste(x)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))
        self.assertEqual(M.fp8_qdq(torch.tensor([500.0, -1000.0, 1.0])).tolist(), [448.0, -448.0, 1.0])


class TestDequant(unittest.TestCase):
    def test_matches_manual(self):
        g = torch.Generator().manual_seed(8)
        packed = torch.randint(0, 256, (4, 16), generator=g, dtype=torch.int32).to(torch.uint8)
        scale = (torch.rand(4, 2, generator=g) + 0.5).to(torch.float8_e4m3fn)
        gsc = torch.tensor([3.0])
        w = M.dequant_nvfp4(packed, scale, gsc, torch.float32)
        lut = list(M.E2M1_POS) + [-v for v in M.E2M1_POS]
        for r in range(4):
            for c in range(32):
                b = int(packed[r, c // 2])
                code = (b & 0xF) if c % 2 == 0 else (b >> 4)
                exp = torch.tensor(lut[code]) * (scale[r, c // 16].float() / torch.tensor(3.0))   # fp32, same order
                self.assertEqual(float(w[r, c]), float(exp))


class TestRowMasks(unittest.TestCase):
    def test_modes(self):
        resp = [0] * 5 + [1] * 4 + [0] * 3 + [1] * 2
        self.assertIsNone(M.fp4_row_mask(resp, "w4a16"))
        self.assertEqual(M.fp4_row_mask(resp, "pi").tolist(), [not r for r in resp])
        self.assertTrue(bool(M.fp4_row_mask(resp, "pi_all").all()))
        self.assertEqual(M.segments([0, 1, 1, 0, 1]), [(1, 3), (4, 5)])
        # prepi: prompt runs cut into chunks of 2 rows, fp4x2 iff the chunk has >= 2 rows
        m = M.fp4_row_mask(resp, "prepi", prepi_min_rows=2, chunk=2).tolist()
        self.assertEqual(m, [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
        self.assertIsNone(M.fp4_row_mask([1] * 6, "prepi"))
        with self.assertRaises(ValueError):
            M.fp4_row_mask(resp, "x")

    def test_gen_frac_deterministic(self):
        resp = [0] * 10 + [1] * 200
        a = M.fp4_row_mask(resp, "pi", gen_frac=0.25, seed=3)
        b = M.fp4_row_mask(resp, "pi", gen_frac=0.25, seed=3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(bool(a[:10].all()))
        self.assertTrue(20 < int(a[10:].sum()) < 80)


class _Lin(torch.nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.a = torch.nn.Linear(k, n, bias=False).to(torch.bfloat16)


class TestRowEmulation(unittest.TestCase):
    def test_rows_and_eligibility(self):
        torch.manual_seed(9)
        mod = _Lin(128, 256)
        small = _Lin(96, 256)
        n, nf = M.install_row_emulation(mod, select=lambda nm, m: True)
        M.install_row_emulation(small, select=lambda nm, m: True)
        self.assertEqual((n, nf), (1, 1))
        x = torch.randn(1, 6, 256).to(torch.bfloat16)
        plain = torch.nn.functional.linear(x, mod.a.weight)
        self.assertTrue(torch.equal(mod.a(x), plain))
        rows = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
        M.Emu.rows = rows
        try:
            y = mod.a(x)
            ys = small.a(x)
            with self.assertRaises(RuntimeError):
                mod.a(x[:, :5])
        finally:
            M.Emu.rows = None
        fp = M.fp4x2_linear(x[0], mod.a.weight)
        for i in range(6):
            self.assertTrue(torch.equal(y[0, i], fp[i] if rows[i] else plain[0, i]), i)
        self.assertTrue(torch.equal(ys, torch.nn.functional.linear(x, small.a.weight)))   # N % 128: W4A16


class TestData(unittest.TestCase):
    def test_domains(self):
        cases = {"math": "maths", "code_gen": "code", "code_edit": "code", "long_code": "code", "code": "code",
                 "chat": "prose", "long_prose": "prose", "multi": "multi", "tools": "tools", "traffic": "traffic",
                 "weird": "other"}
        for k, d in cases.items():
            self.assertEqual(M.domain_of({"kind": k}), d, k)
        self.assertEqual(M.domain_of({"kind": "long", "prompt": "Review ```python\nx```"}), "code")
        self.assertEqual(M.domain_of({"kind": "long", "prompt": "Summarise these articles"}), "prose")

    def test_excludes_and_load(self):
        with tempfile.TemporaryDirectory() as td:
            a, b = os.path.join(td, "gen_v3.jsonl"), os.path.join(td, "gen_v4.jsonl")
            rows = lambda kinds: "".join(json.dumps({"prompt_text": "p", "gen_text": "g", "kind": k}) + "\n" for k in kinds)  # noqa: E731
            with open(a, "w") as f:
                f.write(rows(["chat", "math", "code_gen", "tools"]))
            with open(b, "w") as f:
                f.write(rows(["chat", "tools", "math"]) + "not json\n")
            got, st = M.load_rows_multi(f"{a},{b}", "gen_v3.jsonl:chat,code_gen,gen_v3.jsonl:tools")
            self.assertEqual(sorted(r["kind"] for r in got),
                             ["maths:math", "maths:math", "prose:chat", "tools:tools"])
            self.assertEqual(st["gen_v4.jsonl"], {"rows": 3, "bad": 1, "kept": 3})
            self.assertEqual(M.parse_excludes(" chat , f.jsonl:x ,"), [(None, "chat"), ("f.jsonl", "x")])


def _ev(acc, dom):
    return {"acc_len": acc, "by_domain": {d: {"acc_len": v} for d, v in dom.items()}}


class TestGates(unittest.TestCase):
    def test_gate(self):
        ref = _ev({"1": 1.80, "2": 2.4, "3": 2.80}, {"prose": {"3": 2.5}, "maths": {"3": 3.4}, "code": {"3": 3.2}})
        good = _ev({"1": 1.81, "2": 2.5, "3": 2.85}, {"prose": {"3": 2.6}, "maths": {"3": 3.395}, "code": {"3": 3.2}})
        ok, det = M.gate_v4(good, ref)
        self.assertTrue(ok, det)
        bad_dom = _ev({"1": 1.81, "2": 2.5, "3": 2.85}, {"prose": {"3": 2.6}, "maths": {"3": 3.38}, "code": {"3": 3.2}})
        self.assertFalse(M.gate_v4(bad_dom, ref)[0])
        bad_1 = _ev({"1": 1.80, "2": 2.5, "3": 2.9}, {})
        self.assertFalse(M.gate_v4(bad_1, ref)[0])
        fr, fn = {"acc_len": {"1": 1.70}}, {"acc_len": {"1": 1.694}}
        self.assertFalse(M.gate_v4(good, ref, ev_full=fn, ref_full=fr)[0])          # load path -0.006 < -0.005
        ok, det = M.gate_v4(good, ref, ev_full={"acc_len": {"1": 1.697}}, ref_full=fr)
        self.assertTrue(ok)
        self.assertEqual(det["load_path_1"], -0.003)
        self.assertTrue(M.better_v4(good, None))
        self.assertTrue(M.better_v4(good, ref))
        self.assertFalse(M.better_v4(ref, good))
        t = M.diag_table({"pi": good, "prepi": ref})
        self.assertEqual(t["prose"], {"pi": 2.6, "prepi": 2.5})
        self.assertEqual(t["overall"], {"pi": 2.85, "prepi": 2.80})


class TestFidelityHelpers(unittest.TestCase):
    def test_parse_and_compare(self):
        ids = [5, 7, 9]
        pl = [None, {"7": {"logprob": -0.5, "rank": 2}, "3": {"logprob": -0.1, "rank": 1}},
              {"9": {"logprob": -0.2, "rank": 1}}]
        lp, top1 = M.parse_prompt_logprobs(pl, ids)
        self.assertEqual((lp, top1), ([-0.5, -0.2], [3, 9]))
        c = M.compare_lp([(lp, top1)], [([-0.4, -0.2], [3, 8])])
        self.assertEqual(c["tokens"], 2)
        self.assertAlmostEqual(c["mean_abs_dlogp"], 0.05)
        self.assertAlmostEqual(c["dnll"], -0.05)
        self.assertEqual(c["top1_agree"], 0.5)


class TestArgs(unittest.TestCase):
    def test_defaults(self):
        a = M.parse_args([])
        self.assertEqual((a.nsteps, a.step_w, a.gate_steps, a.eval_nsteps), (3, [1.0, 0.7, 0.5], (1, 3), 4))
        self.assertEqual((a.serve_rows, a.shard_scales, a.kv_fp8, a.mtp_quant, a.head_quant, a.init),
                         ("pi", "served", 1, "nvfp4", "fp8", "prod"))
        self.assertAlmostEqual(a.lr_head, 3e-6)
        b = M.parse_args(["--nsteps", "4", "--gate-steps", "3,1,4"])
        self.assertEqual((b.nsteps, b.gate_steps, b.eval_nsteps), (4, (1, 3, 4), 4))
        with self.assertRaises(SystemExit):
            M.parse_args(["--smoke", "--eval-only"])


# ---------------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAS_QWEN, "transformers with qwen3_5 not available")
@unittest.skipIf(GPUSHARE_CPU, "the HF Qwen3.5 CPU forward probes CUDA (gpushare); run off the server")
class TestTinyQwen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = tiny_cfg()
        cls.lm = QM.Qwen3_5TextModel(cls.cfg).to(torch.bfloat16).eval().requires_grad_(False)
        cls.lm_head = (torch.randn(cls.cfg.vocab_size, cls.cfg.hidden_size) * 0.5).to(torch.bfloat16)
        M.install_row_emulation(cls.lm, select=lambda nm, m: "mlp" in nm or "proj" in nm, n_mult=16, k_mult=16)
        M.install_kv_fp8()

    def _ctx(self, vocab_ids):
        d2t, t2d = V2.build_vocab_maps(vocab_ids, self.cfg.vocab_size)
        torch.manual_seed(1)
        mtp = V2.make_mtp(self.cfg).float()
        head = torch.nn.Parameter(self.lm_head[d2t].float().clone())
        return V2.Ctx(self.lm, self.lm_head, mtp, head, d2t, t2d, topk=8)

    def _docs(self, n=3, T=40, seed=0):
        g = torch.Generator().manual_seed(seed)
        out = []
        for j in range(n):
            ids = torch.randint(0, self.cfg.vocab_size, (T,), generator=g).tolist()
            resp = [0] * 6 + [1] * (T - 7) + [0]
            out.append(V2.Doc(ids, resp, ["prose:chat", "maths:math", "code:code_gen"][j % 3]))
        return out

    def test_row_modes_change_only_what_they_should(self):
        ctx = self._ctx(torch.arange(64))
        d = self._docs(1)[0]
        a = V2.main_pass(ctx, d.ids)
        b = M.main_pass_v4(ctx, d.ids, None)
        self.assertTrue(torch.equal(a["h"], b["h"]))
        c = M.main_pass_v4(ctx, d.ids, M.fp4_row_mask(d.resp, "pi"))
        self.assertFalse(torch.equal(a["h"], c["h"]))
        rel = float((a["h"].float() - c["h"].float()).norm() / a["h"].float().norm())
        self.assertLess(rel, 0.1)
        self.assertIsNone(M.Emu.rows)

    def test_kv_fp8_switch(self):
        ctx = self._ctx(torch.arange(64))
        d = self._docs(1)[0]
        a = M.main_pass_v4(ctx, d.ids, None)["h"]
        M.Emu.kv_fp8 = True
        try:
            b = M.main_pass_v4(ctx, d.ids, None)["h"]
        finally:
            M.Emu.kv_fp8 = False
        self.assertTrue(bool(torch.isfinite(b.float()).all()))
        c = M.main_pass_v4(ctx, d.ids, None)["h"]
        self.assertTrue(torch.equal(a, c))
        M.install_kv_fp8()                                   # idempotent

    def test_bf16_eval_equals_v2_draft_eval(self):
        ids = torch.randperm(64, generator=torch.Generator().manual_seed(6))[:40]
        ctx = self._ctx(ids)
        docs = self._docs(3)
        cache2, _ = V2.cache_eval_docs(ctx, docs)
        ev2 = V2.evaluate(ctx, cache2, "draft", 3, chunk=8)
        a = M.parse_args([])
        cache4, ds = M.cache_eval_docs_v4(ctx, docs, "w4a16", a)
        ev4 = M.evaluate_v4(ctx, cache4, 3, "bf16", chunk=8)
        self.assertEqual(ev2["acc_len"], ev4["acc_len"])
        self.assertEqual(set(ev4["by_domain"]), {"prose", "maths", "code"})
        full2 = V2.evaluate(ctx, cache2, "full", 1, chunk=8)
        full4 = M.evaluate_v4(ctx, cache4, 1, "full", chunk=8)
        self.assertEqual(full2["acc_len"], full4["acc_len"])
        ev8 = M.evaluate_v4(ctx, cache4, 4, "fp8", chunk=8)
        self.assertEqual(len(ev8["acc_len"]), 4)
        for m in range(1, 5):
            self.assertTrue(1.0 <= ev8["acc_len"][str(m)] <= 1.0 + m)

    @needs_backward
    def test_training_with_fp8_head_ste_reduces_loss(self):
        ids = torch.randperm(64, generator=torch.Generator().manual_seed(6))[:40]
        ctx = self._ctx(ids)
        docs = self._docs(3)
        pids, presp = V2.pack_docs(docs, 128)[0]
        rows = M.fp4_row_mask(presp, "pi")
        mp = M.main_pass_v4(ctx, pids, rows)
        resp = torch.tensor(presp, dtype=torch.bool)
        params = list(ctx.mtp.parameters()) + [ctx.head_w]
        opt = torch.optim.AdamW(params, lr=3e-3)
        losses = []
        M.Emu.kv_fp8 = True
        try:
            for _ in range(12):
                outs = V2.run_chain(ctx.mtp, mp["h"], mp["e"], mp["pe"], 3, grad_ckpt=True)
                oh = [M.fp8_rows_ste(o) for o in outs]
                loss, _ = V2.mtp_losses(oh, M.fp8_rows_ste(ctx.head_w.to(torch.bfloat16)), ctx.d2t, mp["q"], mp["di"],
                                        mp["valid"], mp["targ"], resp, [1.0, 0.7, 0.5], chunk=16)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
        finally:
            M.Emu.kv_fp8 = False
        self.assertLess(losses[-1], losses[0])
        self.assertIsNotNone(ctx.head_w.grad)


# ---------------------------------------------------------------------------------------------------------------------
class _FakeTok:
    """character tokenizer: id = ord(c) % 64, one offset per character (what V2.build_docs needs)."""
    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=True):
        return {"input_ids": [[ord(c) % 64 for c in t] for t in texts],
                "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts]}

    def convert_tokens_to_ids(self, name):
        return -1


@unittest.skipUnless(HAS_QWEN, "transformers with qwen3_5 not available")
@unittest.skipIf(GPUSHARE_CPU, "backward + HF forward on CPU: run off the server")
class TestRunEndToEnd(unittest.TestCase):
    """run() on a tiny Qwen3.5 with every server dependency stubbed: vLLM stop, the NVFP4 loader, the served endpoint,
    /proc/meminfo, the tokenizer. Exercises the wiring: data, fidelity, caches, baselines, training, gate, save."""

    def setUp(self):
        from types import SimpleNamespace
        from unittest import mock
        from safetensors.torch import save_file
        torch.manual_seed(0)
        self.td = tempfile.TemporaryDirectory()
        td = self.td.name
        cfg = tiny_cfg()
        lm = QM.Qwen3_5TextModel(cfg).to(torch.bfloat16).eval().requires_grad_(False)
        lm_head = (torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.5).to(torch.bfloat16)
        self.model_dir, self.prod = os.path.join(td, "model"), os.path.join(td, "model", "horizon_mtp")
        os.makedirs(self.prod)
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump({"text_config": {"vocab_size": cfg.vocab_size}}, f)
        mtp = V2.make_mtp(cfg)
        sd = {"mtp." + k: v.to(torch.bfloat16).contiguous() for k, v in mtp.state_dict().items()}
        save_file(sd, os.path.join(self.model_dir, "model_mtp.safetensors"))
        save_file(sd, os.path.join(self.prod, "model_mtp.safetensors"))
        vocab = torch.randperm(64, generator=torch.Generator().manual_seed(2))[:48].long()
        torch.save(vocab, os.path.join(self.prod, "draft_vocab.pt"))
        save_file({"weight": torch.zeros(48, cfg.hidden_size, dtype=torch.bfloat16)},
                  os.path.join(self.prod, "draft_head.safetensors"))
        self.tmpl = os.path.join(td, "tmpl.jinja")
        with open(self.tmpl, "w") as f:
            f.write("x")
        g = random.Random(3)
        kinds = ["chat", "math", "code_gen", "multi", "tools"]

        def rows(n, seed):
            r = random.Random(seed)
            return [{"kind": kinds[i % 5], "prompt_text": "".join(r.choice("abcdefgh ") for _ in range(30)),
                     "gen_text": "".join(r.choice("ijklmnop.") for _ in range(90))} for i in range(n)]
        self.data, self.evald = os.path.join(td, "gen.jsonl"), os.path.join(td, "eval.jsonl")
        for path, rs in ((self.data, rows(24, 4)), (self.evald, rows(10, 5))):
            with open(path, "w") as f:
                for r in rs:
                    f.write(json.dumps(r) + "\n")
        del g
        fake = SimpleNamespace(model=SimpleNamespace(language_model=lm, visual=None),
                               config=SimpleNamespace(text_config=cfg), lm_head=SimpleNamespace(weight=lm_head))
        orig_install = M.install_row_emulation
        ctx0 = V2.Ctx(lm, lm_head, None, None, *V2.build_vocab_maps(vocab, 64), topk=8)

        def served(url, ids, **kw):      # the "server": this very model with every row on fp4x2 and FP8 KV
            orig_install(lm, select=lambda n, m: True, n_mult=16, k_mult=16)
            M.install_kv_fp8()
            M.Emu.kv_fp8 = True
            try:
                return M.teacher_prompt_logprobs(ctx0, ids, torch.ones(len(ids), dtype=torch.bool))
            finally:
                M.Emu.kv_fp8 = False
        self.patches = [
            mock.patch.object(V2, "stop_vllm", lambda: None),
            mock.patch.object(V2, "mem_snapshot", lambda: {k: 40.0 for k in ("avail_gb", "rss_gb", "anon_gb", "file_gb", "hwm_gb")}),
            mock.patch.object(V2, "WATCHDOG", SimpleNamespace(phase="", min_avail_seen=40.0), create=True),
            mock.patch.object(V2, "load_main_model", lambda d: (fake, 7, 0)),
            mock.patch.object(M, "install_row_emulation",
                              lambda m: orig_install(m, select=lambda n, x: True, n_mult=16, k_mult=16)),
            mock.patch.object(M, "served_prompt_logprobs", served),
            mock.patch("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _FakeTok()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        M.Emu.kv_fp8, M.Emu.rows = False, None
        self.td.cleanup()

    def _args(self, *extra):
        return M.parse_args(["--model", self.model_dir, "--data", self.data, "--eval-data", self.evald,
                             "--vocab", os.path.join(self.prod, "draft_vocab.pt"), "--prod-dir", self.prod,
                             "--outdir", os.path.join(self.td.name, "out"), "--shard-scales", "exact",
                             "--mtp-quant", "none", "--max-packs", "4", "--maxlen", "512", "--heldout", "8",
                             "--eval-every", "2", "--log-every", "1", "--chat-template", self.tmpl, "--fidelity-n", "2",
                             "--lr", "3e-3", "--topk", "8", "--min-greedy-frac", "0", *extra])

    def test_eval_only(self):
        M.run(self._args("--eval-only"))                 # fidelity requirement passes: pi_all emulation == "server"

    def test_train_and_save(self):
        from unittest import mock
        with mock.patch.object(M, "gate_v4", lambda ev, ref, *a, **k: (True, {"forced": True})):
            M.run(self._args("--init", "orig"))
        out = os.path.join(self.td.name, "out")
        self.assertTrue(os.path.exists(os.path.join(out, "meta.json")))
        with open(os.path.join(out, "meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["fidelity"]["pi_all+exact_shard+fp8kv"]["mean_abs_dlogp"], 0.0)
        self.assertGreater(meta["fidelity"]["v2"]["mean_abs_dlogp"], 0.0)
        self.assertIn("load_path", meta["trained"])
        self.assertTrue(torch.equal(torch.load(os.path.join(out, "draft_vocab.pt")),
                                    torch.load(os.path.join(self.prod, "draft_vocab.pt"))))
        with self.assertRaises(RuntimeError):            # never into a directory that already holds a head
            M.run(self._args("--init", "orig"))

    def test_real_gate_runs(self):
        M.run(self._args("--smoke"))
        M.run(self._args())                              # saves or not, but completes


if __name__ == "__main__":
    unittest.main()
