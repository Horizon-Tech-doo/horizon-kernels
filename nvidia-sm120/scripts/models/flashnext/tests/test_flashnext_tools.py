# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Flash-Next pruning tools (03-10-2026). numpy required; the hook tests also need torch.
  python -m unittest discover -s scripts/models/flashnext/tests -v

A tiny synthetic checkpoint with Qwen4Exp tensor names: 3 MoE layers + 1 MTP layer, 16 experts, top-4, hidden 32,
expert width 16. Main experts are real NVFP4 (packed E2M1 + E4M3 block-16 scales + FP32 global scale + input_scale),
the MTP experts FP8 128-block (as NVIDIA's checkpoint), router BF16, shards split so that one expert's tensors sit in
different files."""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fn_common as fc  # noqa: E402
import prune_checkpoint as pc  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

E, K, H, I, L = 16, 4, 32, 16, 3


def _tensor_bytes(arr, dtype):
    if dtype == "BF16":
        return fc.f32_to_bf16(arr).tobytes()
    return np.ascontiguousarray(arr).tobytes()


def build_ckpt(root, expert_format="nvfp4", seed=0, quant_names=None, fused=False):
    """-> (dir, dense float32 weights {name: array} for checks)."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    tensors = []          # (name, dtype, shape, bytes)
    dense = {}

    def add(name, dtype, arr):
        arr = np.asarray(arr)
        tensors.append((name, dtype, list(arr.shape), _tensor_bytes(arr, dtype)))

    add("model.language_model.embed_tokens.weight", "BF16", rng.normal(size=(64, H)).astype(np.float32))
    for li in range(L):
        p = f"model.language_model.layers.{li}"
        add(f"{p}.linear_attn.in_proj_qkv.weight", "BF16", rng.normal(size=(48, H)).astype(np.float32))
        gw = rng.normal(size=(E, H)).astype(np.float32)
        add(f"{p}.mlp.gate.weight", "BF16", gw)
        dense[f"{p}.mlp.gate.weight"] = fc.bf16_to_f32(fc.f32_to_bf16(gw))
        add(f"{p}.mlp.shared_expert_gate.weight", "BF16", rng.normal(size=(1, H)).astype(np.float32))
        for proj, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            add(f"{p}.mlp.shared_expert.{proj}.weight", "BF16", rng.normal(size=(o, i)).astype(np.float32) * 0.1)
        if fused:
            for proj, (o, i) in (("gate_up_proj", (2 * I, H)), ("down_proj", (H, I))):
                add(f"{p}.mlp.experts.{proj}", "BF16", rng.normal(size=(E, o, i)).astype(np.float32))
            continue
        for e in range(E):
            for proj, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                w = rng.normal(size=(o, i)).astype(np.float32) * (0.05 + 0.02 * e)
                b = f"{p}.mlp.experts.{e}.{proj}"
                if expert_format == "nvfp4":
                    packed, sc, g = fc.quant_nvfp4_reference(w)
                    add(b + ".weight", "U8", packed)
                    add(b + ".weight_scale", "F8_E4M3", sc)
                    add(b + ".weight_scale_2", "F32", np.array(g, dtype=np.float32))
                    add(b + ".input_scale", "F32", np.array(0.01 * (e + 1), dtype=np.float32))
                    dense[b] = fc.dequant_nvfp4(packed, sc, g)
                else:
                    add(b + ".weight", "BF16", w)
                    dense[b] = fc.bf16_to_f32(fc.f32_to_bf16(w))
    # MTP layer: FP8 128x128-block experts (NVIDIA layout)
    add("mtp.fc_hidden.weight", "BF16", rng.normal(size=(H, 2 * H)).astype(np.float32))
    add("mtp.layers.0.mlp.gate.weight", "BF16", rng.normal(size=(E, H)).astype(np.float32))
    tab = fc.e4m3_table()
    good = np.where(np.isfinite(tab))[0]
    for e in range(E):
        for proj, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            b = f"mtp.layers.0.mlp.experts.{e}.{proj}"
            add(b + ".weight", "F8_E4M3", rng.choice(good, size=(o, i)).astype(np.uint8))
            add(b + ".weight_scale_inv", "F32", np.full((1, 1), 0.01 * (e + 1), dtype=np.float32))
    # shards: round-robin by tensor so experts straddle files
    shards = {}
    for n, t in enumerate(tensors):
        shards.setdefault(f"model-{n % 3 + 1:05d}-of-00003.safetensors", []).append(t)
    wmap = {}
    for sh, ts in shards.items():
        path = os.path.join(root, sh)
        tmp_src = path + ".src"
        outs, off = [], 0
        with open(tmp_src, "wb") as f:
            for (name, dt, shape, b) in ts:
                f.write(b)
                outs.append(fc.OutTensor(name, dt, shape, [(tmp_src, off, len(b))]))
                off += len(b)
                wmap[name] = sh
        fc.write_safetensors(path, outs, {"format": "pt"})
        os.remove(tmp_src)
    json.dump({"metadata": {"total_size": sum(len(t[3]) for t in tensors)}, "weight_map": wmap},
              open(os.path.join(root, "model.safetensors.index.json"), "w"))
    qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
          "quantized_layers": {f"model.language_model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4",
                                                                                  "group_size": 16}
                               for i in range(L)}}
    if quant_names:
        qc["quantized_layers"].update(quant_names)
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
           "text_config": {"num_experts": E, "num_experts_per_tok": K, "num_hidden_layers": L, "hidden_size": H,
                           "moe_intermediate_size": I, "mtp_num_hidden_layers": 1, "mtp": {"num_hidden_layers": 1},
                           "norm_topk_prob": True},
           "quantization_config": qc}
    json.dump(cfg, open(os.path.join(root, "config.json"), "w"))
    json.dump(qc, open(os.path.join(root, "hf_quant_config.json"), "w"))
    open(os.path.join(root, "tokenizer.json"), "w").write("{}")
    open(os.path.join(root, "LICENSE"), "w").write("Qwen Community License 1.0 (test copy)\n")
    return root, dense


def fake_scores(path, prefixes, seed=1, zero=()):
    rng = np.random.default_rng(seed)
    arrays = {"meta": np.array(json.dumps({"test": True}))}
    for p in prefixes:
        count = rng.integers(1, 1000, size=E).astype(np.float64)
        for z in zero:
            count[z] = 0
        gsum = count * rng.uniform(0.05, 0.3, size=E)
        reap = rng.uniform(0.1, 2.0, size=E)
        arrays[f"{p}|count"] = count
        arrays[f"{p}|gsum"] = gsum
        arrays[f"{p}|reap_sum"] = np.where(count > 0, reap * count, 0)
        arrays[f"{p}|ean_sum"] = count * 3.0
    np.savez(path, **arrays)
    return arrays


PREFIXES = [f"model.language_model.layers.{i}.mlp" for i in range(L)] + ["mtp.layers.0.mlp"]


class TestFormats(unittest.TestCase):
    def test_e4m3_e2m1_tables(self):
        t = fc.e4m3_table()
        self.assertEqual(t[0x38], 1.0)
        self.assertEqual(t[0x7E], 448.0)
        self.assertEqual(t[0x01], 2.0 ** -9)
        self.assertTrue(np.isnan(t[0x7F]))
        self.assertEqual(list(fc.e2m1_table()[:8]), [0, .5, 1, 1.5, 2, 3, 4, 6])

    def test_nvfp4_nibble_order_and_scales(self):
        packed = np.full((1, 8), 0x21, dtype=np.uint8)                 # low nibble 1 (0.5), high 2 (1.0)
        sc = np.full((1, 1), 0x38, dtype=np.uint8)                    # e4m3 1.0
        d = fc.dequant_nvfp4(packed, sc, 2.0)
        self.assertEqual(list(d[0, :2]), [1.0, 2.0])
        rng = np.random.default_rng(0)
        w = rng.normal(size=(8, 64)).astype(np.float32)
        p, s, g = fc.quant_nvfp4_reference(w)
        rel = np.linalg.norm(fc.dequant_nvfp4(p, s, g) - w) / np.linalg.norm(w)
        self.assertLess(rel, 0.15)                                    # 4-bit: coarse but structurally right

    def test_bf16_roundtrip(self):
        x = np.array([1.0, -2.5, 3.14159, 1e-3], dtype=np.float32)
        y = fc.bf16_to_f32(fc.f32_to_bf16(x))
        self.assertTrue(np.allclose(x, y, rtol=1e-2))

    def test_vllm_name_mapping(self):
        self.assertEqual(fc.vllm_layer_to_prefix("language_model.model.layers.7.mlp.experts", 48),
                         "model.language_model.layers.7.mlp")
        self.assertEqual(fc.vllm_layer_to_prefix("model.layers.48.mlp.experts", 48), "mtp.layers.0.mlp")
        self.assertEqual(fc.vllm_layer_to_prefix("mtp.layers.0.mlp.experts", 48), "mtp.layers.0.mlp")


class TestPrune(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fn-test-")
        self.src, self.dense = build_ckpt(os.path.join(self.tmp, "src"))
        self.scores = os.path.join(self.tmp, "scores.npz")
        self.arr = fake_scores(self.scores, PREFIXES, zero=(3,))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def prune(self, *extra, out="out"):
        o = os.path.join(self.tmp, out)
        pc.main(["--src", self.src, "--out", o, *extra])
        return o

    def test_uniform_prune_bytes_and_structure(self):
        out = self.prune("--scores", self.scores, "--prune-ratio", "0.25", "--round", "4", "--verify", "full",
                         "--emit-keep", os.path.join(self.tmp, "keep.json"))
        cfg = json.load(open(os.path.join(out, "config.json")))
        self.assertEqual(cfg["text_config"]["num_experts"], 12)
        self.assertEqual(cfg["horizon_pruning"]["num_experts_src"], 16)
        keep = json.load(open(os.path.join(self.tmp, "keep.json")))["keep"]
        for p in PREFIXES:
            self.assertEqual(len(keep[p]), 12)
            self.assertNotIn(3, keep[p])                      # never-routed expert goes first
            st = {s: self.arr[f"{p}|{s}"] for s in ("count", "reap_sum")}
            reap = np.where(st["count"] > 0, st["reap_sum"] / np.maximum(st["count"], 1), 0)
            self.assertEqual(sorted(keep[p]), sorted(np.argsort(-reap, kind="stable")[:12].tolist()))
        rd_src, rd_out = _Reader(self.src), _Reader(out)
        p = PREFIXES[1]
        for new, old in enumerate(keep[p]):
            for t in ("gate_proj.weight", "gate_proj.weight_scale", "gate_proj.weight_scale_2", "down_proj.input_scale"):
                self.assertEqual(rd_out.raw(f"{p}.experts.{new}.{t}"), rd_src.raw(f"{p}.experts.{old}.{t}"))
        g_src = rd_src.raw(f"{p}.gate.weight")
        g_out = rd_out.raw(f"{p}.gate.weight")
        row = H * 2
        self.assertEqual(g_out, b"".join(g_src[o * row:(o + 1) * row] for o in keep[p]))
        self.assertFalse(rd_out.has(f"{p}.experts.12.gate_proj.weight"))
        self.assertEqual(rd_out.raw("model.language_model.embed_tokens.weight"),
                         rd_src.raw("model.language_model.embed_tokens.weight"))
        self.assertEqual(rd_out.raw(f"{p}.shared_expert_gate.weight"), rd_src.raw(f"{p}.shared_expert_gate.weight"))
        mt = "mtp.layers.0.mlp"
        self.assertEqual(rd_out.raw(f"{mt}.experts.0.up_proj.weight_scale_inv"),
                         rd_src.raw(f"{mt}.experts.{keep[mt][0]}.up_proj.weight_scale_inv"))
        for f in ("LICENSE", "NOTICE", "horizon_pruning.json", "tokenizer.json", "hf_quant_config.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
        total = 0
        for sh in set(idx["weight_map"].values()):
            t, _, _ = fc.read_header(os.path.join(out, sh))
            total += sum(x.nbytes for x in t.values())
        self.assertEqual(total, idx["metadata"]["total_size"])

    def test_verify_detects_corruption(self):
        out = self.prune("--scores", self.scores, "--prune-ratio", "0.5", "--round", "1", "--verify", "none")
        keep = json.load(open(os.path.join(out, "horizon_pruning.json")))["keep"]
        name = "model.language_model.layers.0.mlp.experts.2.up_proj.weight"
        ti = _Reader(out).info(name)
        with open(ti.path, "r+b") as f:
            f.seek(ti.start)
            b = f.read(1)
            f.seek(ti.start)
            f.write(bytes([b[0] ^ 0xFF]))
        with self.assertRaises(AssertionError):
            pc.verify(self.src, out, keep, E, 8, "score", full=True)

    def test_mtp_policy(self):
        no_mtp = os.path.join(self.tmp, "s2.npz")
        fake_scores(no_mtp, PREFIXES[:-1])
        with self.assertRaises(SystemExit):
            self.prune("--scores", no_mtp, "--prune-ratio", "0.5", out="o1")
        out = self.prune("--scores", no_mtp, "--prune-ratio", "0.5", "--mtp", "drop", "--verify", "full", out="o2")
        cfg = json.load(open(os.path.join(out, "config.json")))
        self.assertEqual(cfg["text_config"]["mtp_num_hidden_layers"], 0)
        idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
        self.assertFalse(any(n.startswith("mtp.") for n in idx["weight_map"]))

    def test_keep_file_transfers_to_another_format(self):
        kf = os.path.join(self.tmp, "keep.json")
        pc.main(["--src", self.src, "--scores", self.scores, "--prune-ratio", "0.25", "--round", "4", "--dry-run",
                 "--emit-keep", kf, "--emit-mask", os.path.join(self.tmp, "mask.npz")])
        other, _ = build_ckpt(os.path.join(self.tmp, "bf16src"), expert_format="bf16", seed=5)
        out = os.path.join(self.tmp, "bf16out")
        pc.main(["--src", other, "--out", out, "--keep-file", kf, "--verify", "full"])
        self.assertEqual(json.load(open(os.path.join(out, "config.json")))["text_config"]["num_experts"], 12)
        m = np.load(os.path.join(self.tmp, "mask.npz"))
        self.assertEqual(int(m[f"{PREFIXES[0]}|uniform"].sum()), 12)
        nu = sum(int(m[f"{p}|nonuniform"].sum()) for p in PREFIXES[:-1])
        self.assertEqual(nu, 12 * (len(PREFIXES) - 1))           # same total budget as uniform

    def test_refuses_per_expert_quant_entries(self):
        bad, _ = build_ckpt(os.path.join(self.tmp, "bad"),
                            quant_names={"model.language_model.layers.0.mlp.experts.3.up_proj": {"quant_algo": "FP8"}})
        with self.assertRaises(SystemExit):
            pc.main(["--src", bad, "--out", os.path.join(self.tmp, "bo"), "--scores", self.scores])

    def test_fused_expert_tensors_are_sliced(self):
        fsrc, _ = build_ckpt(os.path.join(self.tmp, "fused"), fused=True)
        out = os.path.join(self.tmp, "fout")
        pc.main(["--src", fsrc, "--out", out, "--scores", self.scores, "--prune-ratio", "0.25", "--round", "4",
                 "--verify", "full"])
        t = _Reader(out).info("model.language_model.layers.0.mlp.experts.gate_up_proj")
        self.assertEqual(t.shape, [12, 2 * I, H])

    def test_notice_carries_nvidia_attribution(self):
        open(os.path.join(self.src, "README.md"), "w").write("---\nlicense_name: nvidia-open-model-license\n---\n")
        out = self.prune("--scores", self.scores, "--prune-ratio", "0.25", "--round", "4", "--verify", "none")
        notice = open(os.path.join(out, "NOTICE")).read()
        self.assertIn("Licensed by NVIDIA Corporation under the NVIDIA Open Model License", notice)
        self.assertIn("Qwen Community License", notice)
        self.assertTrue(os.path.exists(os.path.join(out, "README.md")))

    def test_rejects_bad_counts(self):
        with self.assertRaises(SystemExit):           # fewer kept experts than top-k
            pc.main(["--src", self.src, "--out", os.path.join(self.tmp, "x"), "--scores", self.scores,
                     "--keep-experts", "2"])


class _Reader:
    def __init__(self, d):
        self.wmap, _ = fc.load_index(d)
        self.d = d
        self.cache = {}

    def has(self, n):
        return n in self.wmap

    def info(self, n):
        sh = self.wmap[n]
        if sh not in self.cache:
            self.cache[sh] = fc.read_header(os.path.join(self.d, sh))[0]
        return self.cache[sh][n]

    def raw(self, n):
        return fc.read_bytes(self.info(n))


# ---------------------------------------------------------------------------------------------------------------------
# hooks (torch): a fake vLLM MoE runner with the 0.30 method names
# ---------------------------------------------------------------------------------------------------------------------

@unittest.skipIf(torch is None, "torch not installed")
class TestHooks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fn-hooks-")
        self.src, self.dense = build_ckpt(os.path.join(self.tmp, "src"))
        sys.path.insert(0, os.path.dirname(HERE))
        from horizon_flashnext import hooks  # noqa: PLC0415
        self.hooks = hooks
        hooks._STATE.update(calib=None, mask=None)

    def tearDown(self):
        shutil.rmtree(self.tmp)
        self.hooks._STATE.update(calib=None, mask=None)

    def runner(self, layer, shared=False):
        dense = self.dense
        prefix = f"model.language_model.layers.{layer}.mlp"
        Wg = torch.tensor(dense[prefix + ".gate.weight"])

        class Router:
            def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, input_ids=None):
                p = torch.softmax(router_logits[:, :E].float(), dim=-1)
                w, ids = torch.topk(p, K, dim=-1)
                w = w / w.sum(-1, keepdim=True)
                if shared:     # a fused shared expert as id E with its own weight (FSE layout)
                    ids = torch.cat([ids, torch.full_like(ids[:, :1], E)], 1)
                    w = torch.cat([w, torch.full_like(w[:, :1], 0.5)], 1)
                return w, ids.to(topk_indices_dtype or torch.int32)

        class QM:
            topk_indices_dtype = torch.int32

        class Experts:
            quant_method = QM()

            def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
                out = torch.zeros_like(x, dtype=torch.float32)
                for t in range(x.shape[0]):
                    for s in range(topk_ids.shape[1]):
                        e = int(topk_ids[t, s])
                        b = f"{prefix}.experts.{e}"
                        xt = x[t:t + 1].float()
                        g = xt @ torch.tensor(dense[b + ".gate_proj"]).T
                        u = xt @ torch.tensor(dense[b + ".up_proj"]).T
                        out[t] += float(topk_weights[t, s]) * ((torch.nn.functional.silu(g) * u) @
                                                               torch.tensor(dense[b + ".down_proj"]).T)[0]
                return out.to(x.dtype)

        class Runner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_name = f"language_model.model.layers.{layer}.mlp.experts"
                self.router = Router()
                self.routed_experts = Experts()
                self.seen = []

            def _apply_quant_method(self, hidden_states, router_logits, shared_experts_input, input_ids=None,
                                    shared_experts_overlapping=False):
                w, ids = self.router.select_experts(hidden_states, router_logits)
                self.seen.append(ids)
                return None, self.routed_experts.forward_modular(hidden_states, w[:, :K], ids[:, :K])

            def forward(self, x):
                return self._apply_quant_method(x, x.float() @ Wg.T, None)

        return Runner()

    def brute(self, layer, x):
        prefix = f"model.language_model.layers.{layer}.mlp"
        logits = x @ self.dense[prefix + ".gate.weight"].T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        ids = np.argsort(-p, axis=1)[:, :K]
        cnt, gs, rs = np.zeros(E), np.zeros(E), np.zeros(E)
        for t in range(len(x)):
            w = p[t, ids[t]] / p[t, ids[t]].sum()
            for s, e in enumerate(ids[t]):
                b = f"{prefix}.experts.{e}"
                g = x[t] @ self.dense[b + ".gate_proj"].T
                u = x[t] @ self.dense[b + ".up_proj"].T
                y = (g / (1 + np.exp(-g)) * u) @ self.dense[b + ".down_proj"].T
                cnt[e] += 1
                gs[e] += w[s]
                rs[e] += w[s] * np.linalg.norm(y)
        return cnt, gs, rs

    def test_calibration_matches_bruteforce_and_ignores_shared_slot(self):
        h = self.hooks
        r0, r1 = self.runner(0), self.runner(1, shared=True)
        h.install_instance_wrappers([r0, r1], nhl=L)
        h._STATE["calib"] = h.Calibration(E, "kernel", sample_tokens=8, sample_dir=os.path.join(self.tmp, "samples"))
        h._STATE["calib"].active = True
        rng = np.random.default_rng(3)
        x = rng.normal(size=(20, H)).astype(np.float32)
        for r in (r0, r1):
            r(torch.tensor(x))
            r(torch.tensor(x[:5]))
        self.assertEqual(h._STATE["calib"].errors, [])
        res = h._STATE["calib"].results()
        xx = np.concatenate([x, x[:5]])
        for layer in (0, 1):
            cnt, gs, rs = self.brute(layer, xx)
            got = res[f"model.language_model.layers.{layer}.mlp"]
            np.testing.assert_allclose(got["count"], cnt)
            np.testing.assert_allclose(got["gsum"], gs, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(got["reap_sum"], rs, rtol=1e-3, atol=1e-5)
            self.assertEqual(got["tokens"], 25)
        # offline verify on the dumped samples agrees with the checkpoint
        sys.path.insert(0, os.path.dirname(HERE))
        import reap_calibrate as rc  # noqa: PLC0415
        rd = rc.CkptReader(self.src)
        s = dict(np.load(os.path.join(self.tmp, "samples", "model.language_model.layers.0.mlp.npz")))
        v = rc.verify_block(rd, "model.language_model.layers.0.mlp", s, K)
        self.assertGreaterEqual(v["routing_set_agreement"], 0.99)
        self.assertLess(v["norm_rel_err_median"], 0.02)               # bf16 hidden rounding only
        s["norms"] = s["norms"] * 1.5
        self.assertGreater(rc.verify_block(rd, "model.language_model.layers.0.mlp", s, K)["norm_rel_err_median"], 0.3)

    def test_calibration_refuses_inactive_and_records_errors(self):
        h = self.hooks
        r = self.runner(0)
        h.install_instance_wrappers([r], nhl=L)
        h._STATE["calib"] = h.Calibration(E, "kernel")
        r(torch.zeros(3, H))                                           # inactive: nothing recorded
        self.assertEqual(h._STATE["calib"].calls, 0)
        h._STATE["calib"].active = True
        r.routed_experts.forward_modular = lambda *a, **k: ("not", "a tensor")
        r(torch.ones(3, H))
        self.assertTrue(h._STATE["calib"].errors)
        self.assertFalse(h._STATE["calib"].active)

    def test_runtime_mask_blocks_experts(self):
        h = self.hooks
        keep = np.ones(E, dtype=bool)
        keep[[0, 1, 2, 5, 7]] = False
        path = os.path.join(self.tmp, "mask.npz")
        np.savez(path, **{"meta": np.array("{}"), "model.language_model.layers.2.mlp|uniform": keep})
        h.load_mask(path)
        r = self.runner(2)
        h.install_instance_wrappers([r], nhl=L)
        x = torch.tensor(np.random.default_rng(1).normal(size=(64, H)).astype(np.float32))
        r(x)
        used = set(torch.cat(r.seen).flatten().tolist())
        self.assertTrue(used.isdisjoint({0, 1, 2, 5, 7}))
        self.assertTrue(used)

    def test_ckpt_prefix(self):
        self.assertEqual(self.hooks.ckpt_prefix("model.layers.48.mlp.experts", 48), "mtp.layers.0.mlp")
        self.assertEqual(self.hooks.ckpt_prefix("language_model.model.layers.0.mlp.experts", 48),
                         "model.language_model.layers.0.mlp")


if __name__ == "__main__":
    unittest.main()
