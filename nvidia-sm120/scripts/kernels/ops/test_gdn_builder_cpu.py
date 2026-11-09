# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GDN builder metadata regression without a CUDA or vLLM dependency."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

source = Path(__file__).resolve().parents[3] / 'horizon_kernels/gdn_recover.py'
tree = ast.parse(source.read_text())
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'HorizonGDNRecoverBuilder')
cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', '_ctx')]

class Base:
    def __init__(self, spec, names, config, device):
        self.vllm_config = config
        self.num_spec = 0

ns = {'Base': Base, '_BY_ALOG': {}, '_CommitContext': lambda *args: args}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)

class Tests(unittest.TestCase):
    def test_owns_layer_names_and_refreshes_state(self):
        ptr = [42]
        layer = NS(kv_cache=[None, NS(data_ptr=lambda: ptr[0])], A_log=NS(data_ptr=lambda: 7))
        config = NS(cache_config=NS(use_kda_recoverssm=False),
                    compilation_config=NS(static_forward_context={'layer0': layer}))
        names = ['layer0']
        b = ns['HorizonGDNRecoverBuilder'](None, names, config, None)
        names[0] = 'mutated'
        self.assertEqual(b.layer_names, ('layer0',))
        b.hz_max_reqs = 32; b.hz_buf = {}
        first = b._ctx()
        self.assertIs(first, b._ctx())
        self.assertEqual(ns['_BY_ALOG'][7], 'layer0')
        ptr[0] = 43
        self.assertIsNot(first, b._ctx())
        self.assertEqual(b.hz_ctx_ptr, 43)

if __name__ == '__main__':
    unittest.main()
