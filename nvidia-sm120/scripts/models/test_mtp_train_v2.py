#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU unit tests for mtp_train_v2.py. No GPU, no vLLM, nothing written outside a temp dir.
#   CUDA_VISIBLE_DEVICES="" $HK_VENV/bin/python test_mtp_train_v2.py -v
# Pure tests need only torch. Tiny-model tests need transformers with qwen3_5 (skipped otherwise). Real-artifact tests
# (checkpoint header vs module layout, served chat template) run only where the model directory exists; they read the
# safetensors header and the tokenizer, never the weights.
import json
import math
import os
import struct
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402

torch.set_num_threads(int(os.environ.get("TEST_THREADS", "2")))
if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    # CPU-only run. On the test server the gpushare GPU client aborts the process when anything calls cuInit with no
    # visible device, so no test may even probe CUDA: pin checkpoint's default device to CPU and stub the probes.
    torch.utils.checkpoint.DefaultDeviceType.set_device_type("cpu")
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0


def _gpushare_preloaded():
    try:
        with open("/etc/ld.so.preload") as f:
            return "gpushare" in f.read()
    except OSError:
        return False


# The first backward() starts autograd's device threads, which ask the CUDA runtime for the device count (cuInit) in
# C++; gpushare (in /etc/ld.so.preload) kills the process when cuInit finds no visible device. Gradient tests
# therefore run only where gpushare is absent (e.g. a laptop); everything forward-only runs on the server too.
GPUSHARE_CPU = os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and _gpushare_preloaded()
needs_backward = unittest.skipIf(GPUSHARE_CPU, "gpushare aborts backward() on a GPU-less CPU run; run off the server")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_train_v2 as M  # noqa: E402

try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as QM  # noqa: F401
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    HAS_QWEN = True
except Exception:  # noqa: BLE001
    HAS_QWEN = False

MODEL = os.environ.get("MODEL", M.MODEL_DEFAULT)
HAS_MODEL = os.path.exists(os.path.join(MODEL, "model_mtp.safetensors"))
REAL_MTP_KEYS = sorted([
    "mtp.fc.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight", "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight", "mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight"])


def B(xs):
    return torch.tensor(xs, dtype=torch.bool)


# ---------------------------------------------------------------------------------------------------------------------
class TestVocabMaps(unittest.TestCase):
    def test_roundtrip(self):
        d2t, t2d = M.build_vocab_maps(torch.tensor([5, 2, 9]), 12)
        self.assertEqual(d2t.tolist(), [5, 2, 9])
        self.assertEqual(t2d[d2t].tolist(), [0, 1, 2])
        self.assertEqual(t2d[5].item(), 0)
        self.assertEqual(t2d[2].item(), 1)
        self.assertEqual(t2d[9].item(), 2)
        others = [i for i in range(12) if i not in (5, 2, 9)]
        self.assertTrue(bool((t2d[others] == -1).all()))
        self.assertEqual(d2t.dtype, torch.long)
        self.assertEqual(t2d.dtype, torch.long)

    def test_int32_and_list_inputs(self):
        d2t, _ = M.build_vocab_maps(torch.tensor([3, 1], dtype=torch.int32), 4)
        self.assertEqual(d2t.dtype, torch.long)
        d2t, _ = M.build_vocab_maps([0, 3], 4)
        self.assertEqual(d2t.tolist(), [0, 3])

    def test_errors(self):
        for bad in (torch.tensor([1, 1]), torch.tensor([0, 12]), torch.tensor([-1, 0]),
                    torch.tensor([[1, 2]]), torch.tensor([1.0, 2.0]), torch.tensor([], dtype=torch.long)):
            with self.assertRaises(ValueError):
                M.build_vocab_maps(bad, 12)


# ---------------------------------------------------------------------------------------------------------------------
class TestTeacher(unittest.TestCase):
    def test_topk_probs_are_full_softmax(self):
        lg = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
        p, i, a = M.teacher_from_logits(lg, 2)
        full = lg.softmax(-1)
        self.assertEqual(i.tolist(), [[1, 2]])
        self.assertTrue(torch.allclose(p, full[:, [1, 2]]))
        self.assertEqual(a.tolist(), [1])
        self.assertLess(float(p.sum()), 1.0)

    def test_argmax_on_ties_is_first_index_like_vllm(self):
        lg = torch.tensor([[1.0, 5.0, 5.0, 0.0]], dtype=torch.bfloat16)
        _, _, a = M.teacher_from_logits(lg, 3)
        self.assertEqual(a.tolist(), [1])

    def test_restrict_drop_and_renormalise(self):
        _, t2d = M.build_vocab_maps(torch.tensor([2, 5, 9]), 12)
        p = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]])
        ids = torch.tensor([[5, 7, 2], [7, 8, 11]])          # row 1: nothing in the draft vocab
        q, di, valid, mass = M.restrict_to_vocab(p, ids, t2d)
        self.assertTrue(torch.allclose(q[0], torch.tensor([0.5 / 0.7, 0.0, 0.2 / 0.7])))
        self.assertEqual(di[0].tolist(), [1, 0, 0])           # 5 -> 1, 7 dropped (q = 0), 2 -> 0
        self.assertEqual(valid.tolist(), [True, False])
        self.assertAlmostEqual(float(mass[0]), 0.7, places=6)
        self.assertEqual(float(mass[1]), 0.0)
        self.assertTrue(bool((q[1] == 0).all()))
        self.assertAlmostEqual(float(q[0].sum()), 1.0, places=6)
        self.assertFalse(bool(torch.isnan(q).any()))


# ---------------------------------------------------------------------------------------------------------------------
class TestKL(unittest.TestCase):
    def test_matches_manual(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 6)
        q = torch.tensor([[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.2, 0.5, 0.3]])
        di = torch.tensor([[4, 1, 0], [2, 0, 0], [0, 5, 3]])
        logp = logits.log_softmax(-1)
        manual = 0.0
        for r in range(3):
            for j in range(3):
                if q[r, j] > 0:
                    manual += float(q[r, j] * (q[r, j].log() - logp[r, di[r, j]]))
        self.assertAlmostEqual(float(M.kl_rows(logits, q, di)), manual, places=5)

    def test_zero_when_student_equals_teacher_and_matches_kl_div(self):
        torch.manual_seed(1)
        V = 7
        qfull = torch.rand(4, V).softmax(-1)
        di = torch.arange(V).expand(4, V)
        self.assertAlmostEqual(float(M.kl_rows(qfull.log(), qfull, di)), 0.0, places=5)
        s = torch.randn(4, V)
        ref = torch.nn.functional.kl_div(s.log_softmax(-1), qfull, reduction="sum")
        self.assertAlmostEqual(float(M.kl_rows(s, qfull, di)), float(ref), places=4)

    @needs_backward
    def test_checkpointed_chunk_gradient_equals_direct(self):
        torch.manual_seed(2)
        x = torch.randn(5, 8, requires_grad=True)
        w = torch.randn(6, 8, requires_grad=True)
        q = torch.rand(5, 3).softmax(-1)
        di = torch.randint(0, 6, (5, 3))
        kl1, am1 = M._kl_chunk(x, w, q, di)
        g1 = torch.autograd.grad(kl1, (x, w))
        kl2, am2 = torch.utils.checkpoint.checkpoint(M._kl_chunk, x, w, q, di, use_reentrant=False,
                                                   preserve_rng_state=False)
        g2 = torch.autograd.grad(kl2, (x, w))
        self.assertTrue(torch.allclose(kl1, kl2))
        self.assertTrue(torch.equal(am1, am2))
        for a, b in zip(g1, g2):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))


# ---------------------------------------------------------------------------------------------------------------------
class _FakeEmb(torch.nn.Module):          # out = embedding input  -> reveals which token each slot consumed
    def forward(self, emb, h, pe):
        self.seen_pe = getattr(self, "seen_pe", []) + [pe[0].shape[-2]]
        return emb.clone()


class _FakeHid(torch.nn.Module):          # out = hidden input + 1 -> reveals which hidden each slot consumed
    def forward(self, emb, h, pe):
        return h + 1


class TestAlignment(unittest.TestCase):
    def setUp(self):
        self.T, self.K = 9, 3
        t = torch.arange(self.T, dtype=torch.float32).view(1, -1, 1)
        self.e, self.h = t.clone(), 100 + t.clone()
        self.pe = (torch.zeros(1, self.T, 4), torch.zeros(1, self.T, 4))

    def test_step_k_slot_i_consumes_token_i_plus_k_plus_1(self):
        f = _FakeEmb()
        outs = M.run_chain(f, self.h, self.e, self.pe, self.K)
        self.assertEqual(len(outs), self.K)
        for k, out in enumerate(outs):
            n = self.T - 2 - k
            self.assertEqual(out.shape[1], n)
            self.assertEqual(out[0, :, 0].tolist(), [float(i + k + 1) for i in range(n)])
            self.assertEqual(f.seen_pe[k], n)                  # rotary sliced to the n_k slots

    def test_step_k_consumes_previous_output_at_same_slot(self):
        outs = M.run_chain(_FakeHid(), self.h, self.e, self.pe, self.K)
        for k, out in enumerate(outs):
            n = self.T - 2 - k
            self.assertEqual(out[0, :, 0].tolist(), [100.0 + i + k + 1 for i in range(n)])

    def test_teacher_index(self):
        idx = torch.tensor([0, 3, 5])
        self.assertEqual(M.teacher_index(idx, 0).tolist(), [1, 4, 6])
        self.assertEqual(M.teacher_index(idx, 2).tolist(), [3, 6, 8])
        self.assertEqual(M.n_slots(10, 0), 8)
        self.assertEqual(M.n_slots(10, 4), 4)

    def test_masks(self):
        resp = B([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])   # prompt 0-2, generated 3-8, trailing newline 9
        # slot i valid at step k iff x_{i+1} and x_{i+k+2} are generated
        self.assertEqual(M.step_mask(resp, 0, 8).nonzero().flatten().tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(M.step_mask(resp, 2, 6).nonzero().flatten().tolist(), [2, 3, 4])
        self.assertEqual(M.root_mask(resp, 3, M.n_slots(10, 2)).nonzero().flatten().tolist(), [2, 3, 4])
        # a gap in the generated part (packed docs) breaks roots whose chain would cross it
        resp2 = B([0, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        self.assertEqual(M.root_mask(resp2, 2, M.n_slots(10, 1)).nonzero().flatten().tolist(), [0, 4, 5, 6])

    def _greedy_setup(self, T=20, V=11, seed=0):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, V, (T,), generator=g)
        targ = torch.roll(ids, -1)                  # teacher greedy token at p = data token p+1 (greedy data)
        resp = torch.ones(T, dtype=torch.bool)
        resp[:3] = False
        return ids, targ, resp

    def test_perfect_student_accepts_everything_and_off_by_one_does_not(self):
        ids, targ, resp = self._greedy_setup()
        K = 4
        n_last = M.n_slots(ids.numel(), K - 1)
        ridx = M.root_mask(resp, K, n_last).nonzero().flatten()
        good = torch.stack([ids[ridx + k + 2] == targ[M.teacher_index(ridx, k)] for k in range(K)])
        ev = M.finalize_counts(M.chain_counts(good), K)
        self.assertEqual(ev["acc_len"][str(K)], 1.0 + K)
        self.assertEqual(ev["per_pos"], [1.0] * K)
        bad = torch.stack([ids[ridx + k + 1] == targ[M.teacher_index(ridx, k)] for k in range(K)])
        self.assertLess(M.finalize_counts(M.chain_counts(bad), K)["acc_len"][str(K)], 2.0)

    def _loss_setup(self, shift, requires_grad=False):
        ids, targ, resp = self._greedy_setup(T=24, V=50, seed=3)
        V, K = 50, 3
        d2t, t2d = M.build_vocab_maps(torch.randperm(V, generator=torch.Generator().manual_seed(4)), V)
        W = (torch.eye(V) * 30).to(torch.bfloat16)          # head: logits = 30 * one-hot(out)
        T = ids.numel()
        q = torch.zeros(T, 4)
        q[:, 0] = 1.0
        di = torch.zeros(T, 4, dtype=torch.long)
        di[:, 0] = t2d[targ]                                  # restricted teacher: one-hot on the greedy token
        valid = torch.ones(T, dtype=torch.bool)
        outs = []
        for k in range(K):
            n = M.n_slots(T, k)
            want = ids[torch.arange(n) + k + 2 - shift]      # shift=0: x_{i+k+2}, the right target
            o = torch.nn.functional.one_hot(t2d[want], V).float().unsqueeze(0)
            outs.append(o.requires_grad_(requires_grad))
        return outs, W, d2t, q, di, valid, targ, resp, K

    def test_mtp_losses_alignment(self):
        outs, W, d2t, q, di, valid, targ, resp, K = self._loss_setup(shift=0)
        with torch.no_grad():
            loss, st = M.mtp_losses(outs, W, d2t, q, di, valid, targ, resp, [1.0, 0.8, 0.6], chunk=5)
        self.assertLess(float(loss), 1e-6)
        self.assertEqual(st["agree"], [1.0] * K)
        ch = M.finalize_counts(M.train_chain_counts(st["matches"], resp, K), K)
        self.assertEqual(ch["acc_len"][str(K)], 1.0 + K)
        outs, W, d2t, q, di, valid, targ, resp, K = self._loss_setup(shift=1)
        with torch.no_grad():
            loss_bad, st_bad = M.mtp_losses(outs, W, d2t, q, di, valid, targ, resp, [1.0, 0.8, 0.6], chunk=5)
        self.assertGreater(float(loss_bad), 10.0)
        self.assertLess(max(st_bad["agree"]), 0.5)

    @needs_backward
    def test_mtp_losses_counts_and_grad(self):
        outs, W, d2t, q, di, valid, targ, resp, K = self._loss_setup(shift=1, requires_grad=True)
        loss, st = M.mtp_losses(outs, W, d2t, q, di, valid, targ, resp, [1.0, 0.8, 0.6], chunk=4)
        T = targ.numel()
        for k in range(K):                                    # only real decode slots are trained
            n = M.n_slots(T, k)
            self.assertEqual(st["count"][k], int(M.step_mask(resp, k, n).sum()))
        loss.backward()
        for o in outs:
            self.assertTrue(bool(torch.isfinite(o.grad).all()))
            self.assertGreater(float(o.grad.abs().sum()), 0.0)
            self.assertEqual(float(o.grad[0, :2].abs().sum()), 0.0)   # prompt slots get no gradient

    def test_out_of_vocab_teacher_token_can_never_be_accepted(self):
        # vocab lacks id 7; the teacher's greedy token at the aligned position of draft 1 is 7
        d2t, t2d = M.build_vocab_maps(torch.tensor([10, 11, 12]), 20)
        draft_argmax = torch.tensor([[1], [0], [2]])          # K x R, draft indices -> full ids 11, 10, 12
        targ_aligned = torch.tensor([[11], [7], [12]])
        Mx = d2t[draft_argmax] == targ_aligned
        ev = M.finalize_counts(M.chain_counts(Mx), 3)
        self.assertEqual(ev["acc_len"], {"1": 2.0, "2": 2.0, "3": 2.0})
        self.assertEqual(ev["per_pos"], [1.0, 0.0, 0.0])
        self.assertEqual(ev["agree"], [1.0, 0.0, 1.0])        # draft 2 matched but comes after a rejection
        # and in the loss path: a position whose top-k is entirely out of vocab is dropped, not NaN
        q, di, valid, _ = M.restrict_to_vocab(torch.tensor([[0.9, 0.1]]), torch.tensor([[7, 8]]), t2d)
        self.assertEqual(valid.tolist(), [False])


# ---------------------------------------------------------------------------------------------------------------------
class TestChain(unittest.TestCase):
    def test_hand_computed(self):
        Mx = B([[1, 1, 0, 1],      # step 0
                [1, 0, 1, 1],      # step 1
                [1, 1, 1, 0]])     # step 2      chains: 3, 1 (early stop), 0, 2
        c = M.chain_counts(Mx)
        self.assertEqual(c, {"n": 4, "reach": [3, 2, 1], "agree": [3, 3, 3]})
        ev = M.finalize_counts(c, 3)
        self.assertEqual(ev["acc_len"], {"1": 1.75, "2": 2.25, "3": 2.5})
        self.assertEqual(ev["per_pos"], [0.75, 0.5, 0.25])
        self.assertEqual(ev["agree"], [0.75, 0.75, 0.75])
        ev3 = M.finalize_counts(M.chain_counts(Mx[:, 1:]), 3)     # drop root 0: chains 1, 0, 2
        self.assertEqual(ev3["acc_len"]["3"], 2.0)

    def test_accumulate_and_empty(self):
        a = M.new_counts(2)
        M.add_counts(a, M.chain_counts(B([[1, 0], [1, 1]])))
        M.add_counts(a, M.chain_counts(B([[1], [0]])))
        self.assertEqual(a, {"n": 3, "reach": [2, 1], "agree": [2, 2]})
        e = M.finalize_counts(M.new_counts(2), 2)
        self.assertEqual(e["n_roots"], 0)
        self.assertTrue(math.isnan(e["acc_len"]["2"]))
        self.assertFalse(M.gate_passes(e, {"acc_len": {"2": 1.0}}, [2]))

    def test_train_chain_counts_uses_roots(self):
        resp = B([0, 1, 1, 1, 1, 1, 1, 1])
        T, K = 8, 2
        matches = [torch.ones(M.n_slots(T, k), dtype=torch.bool) for k in range(K)]
        matches[1][2] = False
        c = M.train_chain_counts(matches, resp, K)
        roots = M.root_mask(resp, K, M.n_slots(T, K - 1)).nonzero().flatten().tolist()
        self.assertEqual(roots, [0, 1, 2, 3, 4])
        self.assertEqual(c, {"n": 5, "reach": [5, 4], "agree": [5, 4]})


class TestGate(unittest.TestCase):
    def test_gate_steps(self):
        self.assertEqual(M.gate_steps(5), [3, 5])
        self.assertEqual(M.gate_steps(3), [3])
        self.assertEqual(M.gate_steps(4), [3, 4])
        self.assertEqual(M.gate_steps(6), [3, 5, 6])
        self.assertEqual(M.gate_steps(2), [2])

    def test_gate_and_best(self):
        b = {"acc_len": {"3": 2.5, "5": 2.9}}
        self.assertTrue(M.gate_passes({"acc_len": {"3": 2.6, "5": 3.0}}, b, [3, 5]))
        self.assertFalse(M.gate_passes({"acc_len": {"3": 2.6, "5": 2.9}}, b, [3, 5]))   # equal is not better
        self.assertFalse(M.gate_passes({"acc_len": {"3": 2.4, "5": 3.5}}, b, [3, 5]))
        e1, e2 = {"acc_len": {"3": 2.6, "5": 3.0}}, {"acc_len": {"3": 2.7, "5": 3.0}}
        self.assertTrue(M.better(e1, None, 5))
        self.assertTrue(M.better(e2, e1, 5))
        self.assertFalse(M.better(e1, e2, 5))


class TestPacking(unittest.TestCase):
    def test_whole_docs_in_order(self):
        docs = [M.Doc([1] * 3, [0, 1, 1], "a"), M.Doc([2] * 4, [0, 0, 1, 1], "a"),
                M.Doc([3] * 2, [1, 1], "b"), M.Doc([4] * 5, [0, 1, 1, 1, 1], "b")]
        packs = M.pack_docs(docs, 6)
        self.assertEqual([p[0] for p in packs], [[1, 1, 1], [2, 2, 2, 2, 3, 3], [4] * 5])
        self.assertEqual([p[1] for p in packs], [[0, 1, 1], [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
        self.assertEqual(sum(len(p[0]) for p in packs), 14)
        self.assertTrue(all(len(p[0]) <= 6 and len(p[0]) == len(p[1]) for p in packs))
        self.assertEqual(M.pack_docs([], 6), [])

    def test_exact_fit_and_errors(self):
        packs = M.pack_docs([M.Doc([1] * 3, [1] * 3, "a"), M.Doc([2] * 3, [1] * 3, "a")], 6)
        self.assertEqual(len(packs), 1)
        with self.assertRaises(ValueError):
            M.pack_docs([M.Doc([1] * 7, [1] * 7, "a")], 6)
        with self.assertRaises(ValueError):
            M.pack_docs([M.Doc([1] * 3, [1] * 2, "a")], 6)


class TestDataHelpers(unittest.TestCase):
    def test_response_mask(self):
        offs = [(0, 5), (5, 10), (10, 12), (12, 20), (20, 21)]
        self.assertEqual(M.response_mask(offs, 10, 20), [0, 0, 1, 1, 0])
        self.assertEqual(M.response_mask([(8, 12)], 10, 20), [0])     # straddles the boundary: prompt side

    def test_data_stat_counts(self):
        _, t2d = M.build_vocab_maps(torch.tensor([1, 2, 3]), 10)
        ids = torch.tensor([0, 1, 2, 3, 9, 5])
        resp = B([0, 1, 1, 1, 1, 0])
        targ = torch.tensor([1, 2, 7, 9, 5, 0])      # teacher greedy at p (for x_{p+1})
        mass = torch.tensor([1.0, 0.5, 0.25, 0.0, 1.0, 1.0])
        valid = B([1, 1, 1, 0, 1, 1])
        c = M.data_stat_counts(ids, resp, targ, t2d, mass, valid)
        # counted positions p = 1, 2, 3 (x_2, x_3, x_4 are drafted tokens)
        self.assertEqual(c["n"], 3)
        self.assertEqual(c["oov"], 2)                 # targ 7 and 9 are not in {1, 2, 3}
        self.assertEqual(c["greedy"], 2)              # p=1: 2==2, p=3: 9==9; p=2: 3 != 7
        self.assertAlmostEqual(c["mass"], 0.75)
        self.assertEqual(c["nomass"], 1)
        f = M.finalize_data_stats(c)
        self.assertAlmostEqual(f["teacher_argmax_oov_frac"], 0.66667, places=4)

    def test_lr_scale(self):
        self.assertAlmostEqual(M.lr_scale(0, 100, 10), 0.1)
        self.assertAlmostEqual(M.lr_scale(9, 100, 10), 0.5 * (1 + math.cos(math.pi * 0.09)))
        self.assertAlmostEqual(M.lr_scale(100, 100, 10), 0.0)
        self.assertAlmostEqual(M.lr_scale(0, 2, 0), 1.0)


class TestSystemHelpers(unittest.TestCase):
    def test_parse_meminfo(self):
        t = "MemTotal:       50331648 kB\nMemAvailable:   44431360 kB\nHugePages_Total:       0\n"
        d = M.parse_kb_table(t)
        self.assertEqual(d["MemAvailable"], 44431360)
        self.assertAlmostEqual(d["MemAvailable"] / 2 ** 20, 42.37, places=2)

    def test_vllm_cmdline_match(self):
        self.assertTrue(M.is_vllm_cmdline("$HK_VENV/bin/python $HK_VENV/bin/"
                                          "vllm serve $HK_DATA/models/Qwen3.8-27B-NVFP4 --port 8000"))
        self.assertTrue(M.is_vllm_cmdline("VLLM::EngineCore"))
        for cmd in ("supervisorctl stop vllm", "python mtp_train_v2.py --smoke", "grep vllm", ""):
            self.assertFalse(M.is_vllm_cmdline(cmd), cmd)

    def test_parse_args(self):
        for k in ("NSTEPS", "STEP_W", "LR", "LR_HEAD"):
            os.environ.pop(k, None)
        a = M.parse_args([])
        self.assertEqual(a.nsteps, 5)
        self.assertEqual(a.step_w, [1.0, 0.8, 0.6, 0.45, 0.35])
        self.assertAlmostEqual(a.lr_head, 5e-6)
        self.assertEqual(M.parse_args(["--nsteps", "3"]).step_w, [1.0, 0.8, 0.6])
        with open(os.devnull, "w") as dn:
            err, sys.stderr = sys.stderr, dn
            try:
                with self.assertRaises(SystemExit):
                    M.parse_args(["--nsteps", "6"])
                with self.assertRaises(SystemExit):
                    M.parse_args(["--smoke", "--eval-only"])
            finally:
                sys.stderr = err


class TestSave(unittest.TestCase):
    def _snap(self):
        return {"mtp": {"fc.weight": torch.randn(4, 8, dtype=torch.bfloat16), "norm.weight": torch.zeros(4)},
                "head": torch.randn(3, 4)}

    def test_build_save_dicts(self):
        meta = {"mtp.fc.weight": (torch.bfloat16, (4, 8)), "mtp.norm.weight": (torch.bfloat16, (4,))}
        m, h = M.build_save_dicts(self._snap(), meta)
        self.assertEqual(sorted(m), sorted(meta))
        self.assertTrue(all(v.dtype == torch.bfloat16 for v in m.values()))
        self.assertEqual(h["weight"].dtype, torch.bfloat16)
        with self.assertRaises(RuntimeError):
            M.build_save_dicts(self._snap(), {"mtp.fc.weight": (torch.bfloat16, (4, 8))})
        with self.assertRaises(RuntimeError):
            M.build_save_dicts(self._snap(), {"mtp.fc.weight": (torch.bfloat16, (8, 4)),
                                              "mtp.norm.weight": (torch.bfloat16, (4,))})

    def test_save_outputs_and_refuse_original(self):
        from safetensors.torch import load_file
        meta = {"mtp.fc.weight": (torch.bfloat16, (4, 8)), "mtp.norm.weight": (torch.bfloat16, (4,))}
        m, h = M.build_save_dicts(self._snap(), meta)
        with tempfile.TemporaryDirectory() as d:
            model_dir, out = os.path.join(d, "model"), os.path.join(d, "model", "horizon_mtp")
            os.makedirs(model_dir)
            paths = M.save_outputs(out, model_dir, m, h, torch.tensor([3, 1, 2]), {"date": "x"})
            self.assertEqual(sorted(load_file(paths["model_mtp.safetensors"])), sorted(meta))
            self.assertEqual(tuple(load_file(paths["draft_head.safetensors"])["weight"].shape), (3, 4))
            self.assertEqual(torch.load(paths["draft_vocab.pt"]).tolist(), [3, 1, 2])
            with open(paths["meta.json"]) as f:
                self.assertEqual(json.load(f)["date"], "x")
            self.assertFalse(any(f.endswith(".tmp") for f in os.listdir(out)))
            with self.assertRaises(RuntimeError):
                M.save_outputs(model_dir, model_dir, m, h, torch.tensor([3, 1, 2]), {})
            self.assertFalse(os.path.exists(os.path.join(model_dir, "model_mtp.safetensors")))


# ---------------------------------------------------------------------------------------------------------------------
def tiny_cfg(vocab=64):
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"], num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    cfg._attn_implementation = "sdpa"
    return cfg


@unittest.skipUnless(HAS_QWEN, "transformers with qwen3_5 not available")
@unittest.skipIf(GPUSHARE_CPU, "the HF Qwen3.5 CPU forward probes CUDA, which gpushare turns into a fatal hang; "
                               "run off the server (same torch/transformers versions)")
class TestTinyQwen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.cfg = tiny_cfg()
        # BF16 main model and lm_head, fp32 MTP master weights: the same dtypes as the real run
        cls.lm = QM.Qwen3_5TextModel(cls.cfg).to(torch.bfloat16).eval().requires_grad_(False)
        cls.lm_head = (torch.randn(cls.cfg.vocab_size, cls.cfg.hidden_size) * 0.5).to(torch.bfloat16)

    def _ctx(self, vocab_ids):
        d2t, t2d = M.build_vocab_maps(vocab_ids, self.cfg.vocab_size)
        torch.manual_seed(1)
        mtp = M.make_mtp(self.cfg).float()
        head = torch.nn.Parameter(self.lm_head[d2t].float().clone())
        return M.Ctx(self.lm, self.lm_head, mtp, head, d2t, t2d, topk=8)

    def _docs(self, n=3, T=40, seed=0):
        g = torch.Generator().manual_seed(seed)
        out = []
        for j in range(n):
            ids = torch.randint(0, self.cfg.vocab_size, (T,), generator=g).tolist()
            resp = [0] * 6 + [1] * (T - 7) + [0]
            out.append(M.Doc(ids, resp, "k%d" % (j % 2)))
        return out

    def test_mtp_layout_matches_real_checkpoint_keys(self):
        mtp = M.make_mtp(self.cfg)
        self.assertEqual(sorted("mtp." + k for k in mtp.state_dict()), REAL_MTP_KEYS)

    def test_main_pass_shapes_and_teacher(self):
        ctx = self._ctx(torch.arange(64))
        mp = M.main_pass(ctx, self._docs(1)[0].ids, chunk=16)
        T = 40
        self.assertEqual(tuple(mp["h"].shape), (1, T, 32))
        self.assertEqual(tuple(mp["q"].shape), (T, 8))
        ref = torch.nn.functional.linear(mp["h"][0], self.lm_head).float().argmax(-1)
        self.assertTrue(torch.equal(mp["targ"], ref))
        self.assertEqual(mp["h"].dtype, torch.bfloat16)
        self.assertTrue(bool(mp["valid"].all()))                 # full vocab: nothing dropped
        self.assertTrue(torch.allclose(mp["q"].sum(-1), torch.ones(T), atol=1e-5))
        self.assertEqual(mp["pe"][0].shape[-2], T)

    def test_draft_head_over_full_vocab_equals_full_head(self):
        # a "draft vocabulary" holding every id in the original order: the draft path (b) must equal (a) exactly
        ctx = self._ctx(torch.arange(64))
        cache, ds = M.cache_eval_docs(ctx, self._docs(3))
        a = M.evaluate(ctx, cache, "full", 3, chunk=8)
        b = M.evaluate(ctx, cache, "draft", 3, chunk=8)
        self.assertEqual(ds["teacher_argmax_oov_frac"], 0.0)
        self.assertEqual(a["acc_len"], b["acc_len"])
        self.assertEqual(a["per_pos"], b["per_pos"])
        self.assertIn("by_kind", a)
        # roots i with x_{i+1} .. x_{i+4} generated: generated tokens are 6..38 -> i in 5..34 -> 30 per doc
        self.assertEqual(a["n_roots"], 3 * 30)
        # a permuted full vocabulary maps back through d2t; only exact BF16 logit ties may break differently
        ctx_p = self._ctx(torch.randperm(64, generator=torch.Generator().manual_seed(5)))
        cache_p, _ = M.cache_eval_docs(ctx_p, self._docs(3))
        bp = M.evaluate(ctx_p, cache_p, "draft", 3, chunk=8)
        for x, y in zip(a["agree"], bp["agree"]):
            self.assertLess(abs(x - y), 0.1)

    def test_trimmed_vocab_eval(self):
        ids = torch.randperm(64, generator=torch.Generator().manual_seed(6))[:40]
        ctx = self._ctx(ids)
        cache, ds = M.cache_eval_docs(ctx, self._docs(3))
        self.assertGreater(ds["teacher_argmax_oov_frac"], 0.0)
        ev_b = M.evaluate(ctx, cache, "draft", 3, chunk=8)
        for m in ("1", "2", "3"):
            self.assertTrue(1.0 <= ev_b["acc_len"][m] <= 1.0 + int(m))
        self.assertTrue(all(k >= 0 for k in ev_b["kl"]))
        self.assertEqual(len(ev_b["kl"]), 3)
        # the (b) head over a trimmed vocab can only lose matches where the teacher token is out of vocab (or on
        # exact ties): its per-step agreement cannot exceed (a)'s by more than tie noise
        ev_a = M.evaluate(ctx, cache, "full", 3, chunk=8)
        for x, y in zip(ev_b["agree"], ev_a["agree"]):
            self.assertLessEqual(x, y + 0.05)
        # training-mode forward (no backward): losses finite, only decode slots counted
        mp = M.main_pass(ctx, self._docs(1)[0].ids, chunk=16)
        resp = torch.tensor(self._docs(1)[0].resp, dtype=torch.bool)
        with torch.no_grad():
            outs = M.run_chain(ctx.mtp, mp["h"], mp["e"], mp["pe"], 3)
            loss, st = M.mtp_losses(outs, ctx.head_w.to(torch.bfloat16), ctx.d2t, mp["q"], mp["di"], mp["valid"],
                                    mp["targ"], resp, [1.0, 0.8, 0.6], chunk=16)
        self.assertTrue(bool(torch.isfinite(loss)))
        for k in range(3):
            n = M.n_slots(40, k)
            self.assertEqual(st["count"][k], int((M.step_mask(resp, k, n) & mp["valid"][k + 1:k + 1 + n]).sum()))

    @needs_backward
    def test_training_reduces_loss_and_save_dry_run(self):
        ids = torch.randperm(64, generator=torch.Generator().manual_seed(6))[:40]
        ctx = self._ctx(ids)
        docs = self._docs(3)
        cache, _ = M.cache_eval_docs(ctx, docs)
        ev_b = M.evaluate(ctx, cache, "draft", 3, chunk=8)
        packs = M.pack_docs(docs, 128)
        self.assertEqual(len(packs), 1)
        pids, presp = packs[0]
        params = list(ctx.mtp.parameters()) + [ctx.head_w]
        opt = torch.optim.AdamW(params, lr=3e-3)
        resp = torch.tensor(presp, dtype=torch.bool)
        mp = M.main_pass(ctx, pids, chunk=32)
        losses = []
        for _ in range(12):
            outs = M.run_chain(ctx.mtp, mp["h"], mp["e"], mp["pe"], 3, grad_ckpt=True)
            loss, st = M.mtp_losses(outs, ctx.head_w.to(torch.bfloat16), ctx.d2t, mp["q"], mp["di"], mp["valid"],
                                    mp["targ"], resp, [1.0, 0.8, 0.6], chunk=16)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])
        self.assertIsNotNone(ctx.head_w.grad)
        ev_c = M.evaluate(ctx, cache, "draft", 3, chunk=8)
        self.assertEqual(ev_c["n_roots"], ev_b["n_roots"])
        # save dry run against the tiny "original" layout
        orig_meta = {"mtp." + k: (torch.bfloat16, tuple(v.shape)) for k, v in ctx.mtp.state_dict().items()}
        md, hd = M.build_save_dicts(M.snapshot(ctx), orig_meta)
        self.assertEqual(sorted(md), sorted(orig_meta))
        self.assertEqual(tuple(hd["weight"].shape), (40, 32))


# ---------------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAS_QWEN and HAS_MODEL, "real model directory not available")
class TestRealArtifacts(unittest.TestCase):
    def test_real_checkpoint_header_matches_module(self):
        from transformers import AutoConfig
        with open(os.path.join(MODEL, "model_mtp.safetensors"), "rb") as f:
            hdr = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        hdr.pop("__metadata__", None)
        tcfg = AutoConfig.from_pretrained(MODEL).text_config
        with torch.device("meta"):                       # shapes only, no memory
            mtp = M.make_mtp(tcfg)
        mine = {"mtp." + k: list(v.shape) for k, v in mtp.state_dict().items()}
        self.assertEqual(sorted(mine), sorted(hdr))
        for k, v in hdr.items():
            self.assertEqual(mine[k], v["shape"], k)
            self.assertEqual(v["dtype"], "BF16", k)

    @unittest.skipUnless(os.path.exists(M.TEMPLATE_DEFAULT), "served chat template not available")
    def test_render_with_served_template(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        with open(M.TEMPLATE_DEFAULT) as f:
            tmpl = f.read()
        rows = [{"prompt": "What is 2+2?", "think": True, "reasoning": "  Two plus two.\n", "content": "4.",
                 "kind": "math"},
                {"prompt": "Say hi", "think": False, "reasoning": "", "content": " Hi! ", "kind": "chat"}]
        p_think, g_think, _ = M.render_texts(tok, tmpl, rows[0])
        p_plain, g_plain, _ = M.render_texts(tok, tmpl, rows[1])
        self.assertIn("<|im_start|>system\nReasoning effort is set to xhigh", p_think)
        self.assertTrue(p_think.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertNotIn("<|im_start|>system", p_plain)
        self.assertEqual(p_plain, "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(g_think, "Two plus two.\n</think>\n\n4.<|im_end|>")
        self.assertEqual(g_plain, "Hi!<|im_end|>")
        docs, st = M.build_docs(rows, tok, tmpl, 4096, min_len=0, min_resp=1)
        self.assertEqual(st["docs"], 2)
        for d, g in zip(docs, (g_think, g_plain)):
            gen_ids = [t for t, r in zip(d.ids, d.resp) if r]
            self.assertEqual(tok.decode(gen_ids), g)             # the mask covers exactly the generated text
            first = d.resp.index(1)
            self.assertTrue(all(d.resp[first:first + len(gen_ids)]))
            self.assertEqual(d.resp[-1], 0)                      # trailing newline is not generated
        # v1 fallback (no template) reproduces v1's literal text for non-thinking rows
        p_v1, _, _ = M.render_texts(tok, None, rows[1])
        self.assertEqual(p_v1, p_plain)


if __name__ == "__main__":
    unittest.main()
