# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch tests: capture must not construct persistent kernels/workspaces."""
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = pathlib.Path(os.environ.get(
    "HORIZON_CAPTURE_TEST_SOURCE",
    pathlib.Path(__file__).resolve().parents[3] / "horizon_kernels" / "k4seg.py"))


class Tensor:
    shape = (4, 16384)
    device = "cuda"
    dtype = "float32"

    def stride(self, index=None):
        return (100, 4, 1) if index is None else 100

    def contiguous(self):
        return self


class PersistentCaptureTests(unittest.TestCase):
    def setUp(self):
        self.capturing = True
        self.torch = types.SimpleNamespace(
            Tensor=Tensor, float32="float32", cuda=types.SimpleNamespace(
                is_current_stream_capturing=lambda: self.capturing))
        spec = importlib.util.spec_from_file_location("persistent_under_test", SOURCE)
        self.mod = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": self.torch}):
            spec.loader.exec_module(self.mod)
        self.mod._log = lambda: types.SimpleNamespace(warning_once=Mock())
        self.g = dict(conv_state=Tensor(), ssm=Tensor(), dt_bias=Tensor(),
                      norm_w=Tensor(), act_sigmoid=0, R=1, lmax=4)
        self.mod._layer = lambda *args: object()
        self.mod._k4_ready = lambda *args: True
        self.mod._in_lin = lambda *args: types.SimpleNamespace(output_size_per_partition=16384)
        self.mod._is_gdn = lambda *args: True
        self.mod._weights = lambda *args: ({}, {})

    def test_segment_cache_miss_does_not_construct_or_warm(self):
        self.mod._segment = Mock(side_effect=AssertionError("constructed during capture"))
        self.mod.warm = Mock(side_effect=AssertionError("warmed during capture"))
        self.mod._fallback = Mock(return_value="per-op")
        self.assertEqual(self.mod._seg_impl(Tensor(), Tensor(), 0, 0, 1), "per-op")
        self.mod._fallback.assert_called_once()
        self.assertFalse(self.mod._WARMED)

    def test_segment_cache_hit_uses_persistent_kernel(self):
        self.mod._SEGS[(4, 16384, "cuda")] = True
        kernel = Mock(return_value=("q", "ba", "res"))
        self.mod._segment = Mock(return_value=kernel)
        self.mod._fallback = Mock(side_effect=AssertionError("unexpected fallback"))
        self.assertEqual(self.mod._seg_impl(Tensor(), Tensor(), 0, 0, 1), ("q", "ba", "res"))
        kernel.assert_called_once()

    def test_gdn_cache_miss_returns_without_tilelang_import(self):
        self.assertIsNone(self.mod._gdn_layer(4, 1, 16384, self.g, "cuda"))
        self.assertFalse(self.mod._GDN_LAYERS)

    def test_gdn_cache_hit_reuses_workspace(self):
        key = (4, 1, 16384, "cuda", (100, 4, 1), 100, "float32",
               "float32", 0, ())
        sentinel = object()
        self.mod._GDN_LAYERS[key] = sentinel
        self.assertIs(self.mod._gdn_layer(4, 1, 16384, self.g, "cuda"), sentinel)

    def test_eager_gdn_constructs_once_then_capture_reuses(self):
        self.capturing = False
        constructor = Mock(return_value=object())
        self.mod._watch = Mock()
        sg = types.ModuleType("tilelang.contrib.horizon_k4")
        sg.segment = types.SimpleNamespace(GdnLayer=constructor)
        with patch.dict(sys.modules, {"tilelang.contrib.horizon_k4": sg}):
            eager = self.mod._gdn_layer(4, 1, 16384, self.g, "cuda")
        self.capturing = True
        self.assertIs(self.mod._gdn_layer(4, 1, 16384, self.g, "cuda"), eager)
        constructor.assert_called_once()
        self.mod._watch.assert_called_once_with(eager)

    def test_gdn_metadata_rejects_bf16_state_before_constructing_kernel_inputs(self):
        q = types.SimpleNamespace(is_conv_state_dim_first=Mock(
            side_effect=AssertionError("unsupported state reached kernel inputs")))
        package = types.ModuleType("vllm.model_executor.layers.mamba.gdn")
        package.qwen_gdn_linear_attn = q
        attn = types.SimpleNamespace(
            _can_use_fused_gdn_mtp_decode=lambda md: True,
            conv1d=types.SimpleNamespace(bias=None), tp_size=1,
            kv_cache=[None, types.SimpleNamespace(dtype="bfloat16")])
        with patch.dict(sys.modules, {package.__name__: package}):
            self.assertIsNone(self.mod._gdn_inputs(attn, types.SimpleNamespace(num_prefills=0)))
        q.is_conv_state_dim_first.assert_not_called()

    def test_gdn_metadata_keeps_fp32_state_on_supported_path(self):
        q = types.SimpleNamespace(is_conv_state_dim_first=lambda: True)
        package = types.ModuleType("vllm.model_executor.layers.mamba.gdn")
        package.qwen_gdn_linear_attn = q
        weight = Mock()
        conv_state, ssm = object(), types.SimpleNamespace(dtype="float32")
        indices = Mock()
        indices.__getitem__ = Mock(return_value=indices)
        indices.size = Mock(return_value=4)
        md = types.SimpleNamespace(num_prefills=0, num_spec_decodes=1,
                                   spec_state_indices_tensor=indices,
                                   spec_query_start_loc=[0, 4], num_accepted_tokens=[1])
        attn = types.SimpleNamespace(
            _can_use_fused_gdn_mtp_decode=lambda md: True, tp_size=1,
            conv1d=types.SimpleNamespace(bias=None, weight=weight),
            kv_cache=[conv_state, ssm], A_log=object(), dt_bias=object(),
            norm=types.SimpleNamespace(weight=object(), activation="silu"),
            head_k_dim=128, layer_norm_epsilon=1e-6)
        with patch.dict(sys.modules, {package.__name__: package}):
            result = self.mod._gdn_inputs(attn, md)
        self.assertIsNotNone(result)
        self.assertIs(result["ssm"], ssm)
        self.assertEqual(result["R"], 1)
        self.assertEqual(result["lmax"], 4)

    def test_gdn_cache_miss_runs_core_and_segment_fallback(self):
        self.mod._layer = lambda *args: types.SimpleNamespace(linear_attn=object())
        self.mod._gdn_inputs = lambda *args: self.g
        self.mod._gdn_core = Mock(return_value="core")
        segment = Mock(return_value=("q", "ba", "res"))
        self.torch.ops = types.SimpleNamespace(horizon=types.SimpleNamespace(k4_segment=segment))
        context = types.ModuleType("vllm.forward_context")
        context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=None)
        with patch.dict(sys.modules, {"vllm.forward_context": context}):
            self.assertEqual(self.mod._gdn_impl(Tensor(), Tensor(), Tensor(), 0, 0, 1),
                             ("q", "ba", "res"))
        self.mod._gdn_core.assert_called_once()
        segment.assert_called_once()
        self.assertFalse(self.mod._GDN_LAYERS)


if __name__ == "__main__":
    unittest.main()
