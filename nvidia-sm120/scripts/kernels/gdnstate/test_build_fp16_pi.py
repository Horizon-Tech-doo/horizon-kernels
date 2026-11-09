# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression for candidate isolation, including the old symlink overwrite risk."""
from pathlib import Path
import tempfile
import unittest

from build_fp16_pi import build
from fp16_pi_gsm_compare import compare


class BuildTests(unittest.TestCase):
    def test_quality_comparison_fails_closed(self):
        baseline = {"correct": [True] * 1319}
        self.assertEqual(compare(baseline, baseline)["result"], "pass")
        self.assertEqual(compare(baseline, {"correct": [True] * 1318})["result"], "fail")
        # Six paired losses are statistically significant even below the 1pp budget.
        self.assertEqual(compare(baseline, {"correct": [False] * 6 + [True] * 1313})["result"], "fail")
        self.assertEqual(compare(baseline, {"correct": [False] * 5 + [True] * 1314})["result"], "pass")
        self.assertEqual(compare(baseline, {"correct": [1] * 1319})["result"], "fail")

    def test_isolation_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tl, hk, kernels = root / "base-tl", root / "base-hk", root / "sources"
            ops = tl / "tilelang/contrib/horizon_ops"
            ops.mkdir(parents=True)
            (tl / "tilelang/__init__.py").write_text("")
            (ops / "gdn_state.py").write_text("BASE STATE\n")
            (ops / "gdn_decode.py").write_text("BASE DECODE\n")
            hp = hk / "horizon_kernels"
            hp.mkdir(parents=True)
            (hp / "vllm_plugin.py").write_text("import os\ndef register():\n    if True:\n        gdn_recover.install()\n")
            # Production plugin modules can themselves be symlinks.
            (root / "base-plugin.py").write_text("BASE PLUGIN\n")
            (hp / "gdn_state.py").symlink_to(root / "base-plugin.py")
            (hp / "prefill_invariant.py").write_text("PI\n")
            kernels.mkdir()
            (kernels / "gdn_state.py").write_text("NEW STATE\n")
            (kernels / "gdn_decode.py").write_text("NEW DECODE\n")
            plugin = root / "plugin.py"
            plugin.write_text("NEW PLUGIN\n")
            env = root / "base.env"
            env.write_text(f'PYTHONPATH={tl}:{hk}:/another-prefix\nHORIZON_PI=1\nHORIZON_GDN_CKPT=1\n'
                           'HORIZON_GDN_CKPT_STOP=fill\nHORIZON_TAIL_MIN_BLOCK=1\nHORIZON_GDN_G3=1\n'
                           'VLLM_EXTRA_ARGS="--prefix-match-unit 64 --enable-mamba-fine-grained-prefix-cache"\n')
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            out = root / "candidate"
            m = build(env, out, kernels, plugin)
            for path, contents in before.items():
                self.assertEqual(path.read_bytes(), contents, str(path))
            for name in ("gdn_state.py", "gdn_decode.py"):
                target = out / "tl/tilelang/contrib/horizon_ops" / name
                self.assertFalse(target.is_symlink())
                self.assertEqual(target.read_bytes(), (kernels / name).read_bytes())
            self.assertFalse((out / "hk/horizon_kernels/gdn_state.py").is_symlink())
            self.assertEqual(m["set"]["HORIZON_GDN_G3"], "0")
            self.assertEqual(m["set"]["HORIZON_GDN_STATE"], "fp16")
            self.assertEqual(m["set"]["PYTHONPATH"], f"{out}/tl:{out}/hk:/another-prefix")
            self.assertEqual((out / "hk/horizon_kernels/prefill_invariant.py").read_text(), "PI\n")
            with self.assertRaisesRegex(ValueError, "already exists"):
                build(env, out, kernels, plugin)
            # Rejected bases must leave no partial candidate.
            env.write_text(env.read_text().replace("HORIZON_PI=1", "HORIZON_PI=0"))
            with self.assertRaisesRegex(ValueError, "HORIZON_PI"):
                build(env, root / "bad", kernels, plugin)
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
