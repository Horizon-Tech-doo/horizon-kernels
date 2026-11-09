# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Default-off fine-prefix lookup ceiling before the speculative prefix drop.

The caller retains a final token for logits. Rounding that cap down before
dropping another hash unit can skip a valid exported Mamba checkpoint. Probe
the ceiling hash boundary, then let the existing finder drop/trim normally.
This adapter selects cache capabilities and geometry, never a model name.

vLLM 0.32 (port 24-10-2026): upstream now does this itself. HybridKVCacheCoordinator.find_longest_cache_hit caps an
EAGLE/MTP group's lookup at len(block_hashes) * hash_block_size (the whole prompt) before dropping the hash unit
(kv_cache_utils.eagle_proof_margin), and get_mamba_prefill_checkpoint_position / FullAttentionManager's tail logic use
the same proof limit. FullAttentionManager.find_longest_cache_hit is byte-identical to 0.30, but the module source hash
below no longer matches, and the old guard raised inside the import of the module (engine crash). On a vLLM that has
eagle_proof_margin the patch is therefore skipped with a log line; on the pinned 0.29/0.30 source it behaves as before.
"""
import functools
import hashlib
import importlib.abc
import os
from pathlib import Path
import sys

TARGET = 'vllm.v1.core.single_type_kv_cache_manager'
SOURCE_SHA256 = '0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf'
STATE = {'installed': False, 'patched': False}


def lookup_ceiling(max_length, block_size, hash_size, alignment, drop, supports):
    if (max_length > 0 and drop and supports and alignment == hash_size
            and 0 < alignment < block_size and block_size % alignment == 0):
        return (max_length + alignment - 1) // alignment * alignment
    return max_length


def upstream_has_eagle_proof_margin():
    """True on vLLM >= 0.32, whose coordinator already looks up the whole prompt before the EAGLE hash-unit drop.
    Checked on the source file (no import), so it is safe to call while vllm.v1.core is being imported."""
    f = Path(__file__)
    try:
        import importlib.util
        spec = importlib.util.find_spec('vllm.v1.core.kv_cache_utils')
        return spec is not None and spec.origin is not None and 'def eagle_proof_margin' in Path(spec.origin).read_text()
    except Exception:  # noqa: BLE001
        return False


def patch_module(module, check_source=True):
    cls = module.FullAttentionManager
    original = cls.find_longest_cache_hit.__func__
    if getattr(original, '_horizon_drop_ceiling', False):
        return
    if upstream_has_eagle_proof_margin():
        from vllm.logger import init_logger
        init_logger('vllm.horizon_kernels').info(
            'horizon_kernels prefix drop: SKIPPED, this vLLM already caps the EAGLE lookup at the whole prompt '
            '(kv_cache_utils.eagle_proof_margin)')
        STATE['patched'] = False
        return
    if check_source and hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('cache finder source changed; review prefix-drop adapter')

    @functools.wraps(original)
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                               block_pool, kv_cache_spec, drop_eagle_block, alignment_tokens,
                               dcp_world_size=1, pcp_world_size=1):
        ceiling = lookup_ceiling(max_length, kv_cache_spec.block_size * dcp_world_size,
                                 block_pool.hash_block_size, alignment_tokens,
                                 drop_eagle_block, cls.supports_fine_grained_hash_lookup)
        blocks, hit = original(cls, block_hashes, ceiling, kv_cache_group_ids,
                               block_pool, kv_cache_spec, drop_eagle_block, alignment_tokens,
                               dcp_world_size, pcp_world_size)
        if hit > max_length:
            raise RuntimeError('speculative prefix drop returned a hit above the caller cap')
        return blocks, hit

    find_longest_cache_hit._horizon_drop_ceiling = True
    cls.find_longest_cache_hit = classmethod(find_longest_cache_hit)
    STATE['patched'] = True
    from vllm.logger import init_logger
    init_logger('vllm.horizon_kernels').info(
        'horizon_kernels prefix drop: fine-hash lookup ceiling before speculative drop')


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        patch_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
                return spec
        return None


def register():
    if os.environ.get('HORIZON_PREFIX_DROP_LOOKUP', '0') != '1' or STATE['installed']:
        return
    if TARGET in sys.modules:
        patch_module(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
    STATE['installed'] = True
