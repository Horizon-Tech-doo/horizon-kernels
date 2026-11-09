# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual pinned finder with fake cache blocks; no torch/CUDA imports."""
import ast
import __future__
from collections.abc import Sequence
import importlib.util
import itertools
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[3] / 'horizon_kernels/prefix_drop_lookup.py'


def adapter():
    spec = importlib.util.spec_from_file_location('prefix_drop_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Spec:
    def __init__(self, block_size):
        self.block_size = block_size


class HashView:
    def __init__(self, hashes, unit, block_size):
        self.hashes, self.ratio = hashes, block_size // unit

    def __iter__(self):
        return iter(self.hashes[self.ratio - 1::self.ratio])


class Pool:
    hash_block_size = 64

    def get_cached_block(self, block_hash, ids):
        return tuple(block_hash for _ in ids)


def vendor_method(path, class_name='FullAttentionManager'):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'find_longest_cache_hit')
    fn.decorator_list = []
    namespace = dict(FullAttentionSpec=Spec, ChunkedLocalAttentionSpec=Spec, MambaSpec=Spec, Sequence=Sequence,
                     itertools=itertools, BlockHashListWithBlockSize=HashView,
                     cdiv=lambda a, b: (a+b-1)//b,
                     resolve_block_hashes=lambda h, unit, block, **kw:
                     h if kw['alignment_tokens'] < block else h[block//unit-1::block//unit])
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'pinned_cache_finder', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace['find_longest_cache_hit']


class DropLookup(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('GDN_VENDOR_CACHE_SOURCE'), 'set pinned vendor cache source')
    def test_sparse_attention_tail_reaches_exported_mamba_checkpoint(self):
        path = os.environ['GDN_VENDOR_CACHE_SOURCE']
        attention = vendor_method(path)
        mamba = vendor_method(path, 'MambaManager')
        class Manager:
            supports_fine_grained_hash_lookup = True
        class SparsePool(Pool):
            null_block = 0
            def get_cached_block(self, block_hash, ids):
                # Full attention has physical-block hashes and one append-only tail.
                # Mamba exports floor((prompt-1)/64)*64 - 64 = 14,208.
                available = {0: set(range(832, 14337, 832)) | {14336}, 1: {14208}}
                return tuple(block_hash for _ in ids) if all(block_hash in available[i] for i in ids) else None
        kwargs = dict(block_hashes=list(range(64, 14337, 64)), block_pool=SparsePool(),
                      kv_cache_spec=Spec(832), drop_eagle_block=True, alignment_tokens=64)
        old_hit = attention(Manager, max_length=14335, kv_cache_group_ids=[0], **kwargs)[1]
        self.assertEqual(old_hit, 14080)
        self.assertEqual(mamba(Manager, max_length=old_hit, kv_cache_group_ids=[1], **kwargs)[1], 0)
        ceiling = adapter().lookup_ceiling(14335, 832, 64, 64, True, True)
        new_hit = attention(Manager, max_length=ceiling, kv_cache_group_ids=[0], **kwargs)[1]
        self.assertEqual(new_hit, 14272)
        self.assertEqual(mamba(Manager, max_length=new_hit, kv_cache_group_ids=[1], **kwargs)[1], 14208)

    def test_ceiling_scope_and_post_drop_cap(self):
        m = adapter()
        for cap in range(4097):
            ceiling = m.lookup_ceiling(cap, 832, 64, 64, True, True)
            self.assertLessEqual(max(0, ceiling-64), cap)
            self.assertLess(ceiling-cap, 64)
            for drop, supports, block, alignment in ((False, True, 832, 64),
                                                     (True, False, 832, 64),
                                                     (True, True, 64, 64),
                                                     (True, True, 832, 128)):
                self.assertEqual(m.lookup_ceiling(cap, block, 64, alignment, drop, supports), cap)

    def test_registration_default_off(self):
        m = adapter()
        before = set(sys.modules)
        with patch.dict(os.environ, {}, clear=True):
            m.register()
        self.assertFalse(m.STATE['installed'])
        self.assertFalse(any(n.startswith(('torch', 'vllm', 'triton')) for n in set(sys.modules)-before))

    @unittest.skipUnless(os.environ.get('GDN_VENDOR_CACHE_SOURCE'), 'set pinned vendor cache source')
    def test_actual_finder_boundary_and_plain_cache(self):
        path = os.environ['GDN_VENDOR_CACHE_SOURCE']
        method = vendor_method(path)
        class Manager:
            supports_fine_grained_hash_lookup = True
            find_longest_cache_hit = classmethod(method)
        kwargs = dict(block_hashes=list(range(64, 20001, 64)), kv_cache_group_ids=[0],
                      block_pool=Pool(), kv_cache_spec=Spec(832), drop_eagle_block=True,
                      alignment_tokens=64)
        self.assertEqual(Manager.find_longest_cache_hit(max_length=14335, **kwargs)[1], 14208)
        m = adapter()
        logger = types.ModuleType('vllm.logger')
        logger.init_logger = lambda _: types.SimpleNamespace(info=lambda *a: None)
        module = types.SimpleNamespace(FullAttentionManager=Manager, __file__=path)
        with patch.dict(sys.modules, {'vllm.logger': logger}):
            m.patch_module(module)
        self.assertEqual(Manager.find_longest_cache_hit(max_length=14335, **kwargs)[1], 14272)
        for cap in (0, 1, 63, 64, 65, 831, 832, 833, 14334, 14335, 14336, 14337):
            self.assertLessEqual(Manager.find_longest_cache_hit(max_length=cap, **kwargs)[1], cap)
            plain = dict(kwargs, drop_eagle_block=False)
            self.assertEqual(Manager.find_longest_cache_hit(max_length=cap, **plain), method(Manager, max_length=cap, **plain))
        for block in (64, 128, 832, 1600):
            plain = dict(kwargs, kv_cache_spec=Spec(block), alignment_tokens=block)
            self.assertEqual(Manager.find_longest_cache_hit(max_length=14335, **plain), method(Manager, max_length=14335, **plain))


if __name__ == '__main__':
    unittest.main()
