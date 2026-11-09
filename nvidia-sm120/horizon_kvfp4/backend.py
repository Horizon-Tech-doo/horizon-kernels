# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""HorizonFp4Backend: vLLM 0.29 attention backend for the Horizon NVFP4 KV cache (hz-kvfp4 v1) on SM120.

Selected with `--kv-cache-dtype nvfp4 --attention-backend CUSTOM` and HORIZON_KVFP4=1 (horizon_kvfp4.register).
vLLM's own nvfp4 plumbing is reused: KVQuantMode.NVFP4, uint8 storage, 2 * KVH head slots of 144 bytes per token
(customize_spec), hybrid page alignment, block allocation and prefix caching. Only our backend lifts the SM100-only
check; the byte meaning inside a slot is ours: [128 bytes E2M1 data | 16 bytes E4M3 block scales].

Per layer (full attention + the MTP draft layer; GDN layers are untouched):
  write     every KV write (prefill, decode, draft layer) -> kv_write_fp4 (bit-exact to kv_fp4_ref; scale search on)
  decode    every request with <= 1 + num_speculative_tokens query tokens (decode, MTP verify, draft steps, short
            prefill tails, zero-length CUDA-graph padding) -> kv_fp4_serving.decode_ragged over ALL requests of the
            batch in one launch (exact GPU seq_lens and query offsets; longer rows are skipped on the device), so
            batch order, padding and async-scheduling CPU upper bounds do not matter. CUDA-graph safe.
  prefill   requests with more query tokens -> the cached context is dequantised EXACTLY to bf16 in chunks of
            HORIZON_KVFP4_CHUNK tokens (default 65536) and FlashInfer FA2 single-request prefill runs over each chunk
            (the chunk holding the new tokens causal, bottom-right aligned; earlier chunks fully visible); chunk
            results are merged with FlashInfer merge_state. Prefill computes in fp16 (HORIZON_KVFP4_PF16=1, default;
            the dequantised K / V are exact in fp16 as in bf16) so P, partial outputs and merges keep 11-bit mantissas;
            only the final store rounds to bf16 (29-09-2026: bf16 prefill reached 0.0026..0.0029 vs fp64 with 2..3
            chunks). Batched path: all prefill rows whose contexts fit one 64k scratch -> one dequant_multi + one
            FlashInfer ragged FA2 call per layer. GPU 29-09-2026 vs fp64: one FA2 call 0.0021..0.0026 (FA2's own bf16
            precision), 2 chunks ~0.0026, 3 chunks ~0.0029, 6 tiny chunks 0.0037; hence 64k chunks (contexts up to
            64k: a single FA2 call, like the FP8 production prefill). Eager (vLLM never captures mixed batches here).
The keys are optionally Hadamard-rotated (HORIZON_KVFP4_HAD=1, default 0). The rotation is fused
(HORIZON_KVFP4_FUSED_ROT=1, default): the CUDA write rotates keys before quantising, the decode kernel rotates the
query rows while loading them (bit-identical to rotate_q + the plain kernel), and the prefill rotates q straight to
fp16 in the launch that otherwise converts it. FUSED_ROT=0: the separate rotate_q launch + TileLang write (old path).
HORIZON_KVFP4_DUMP=DIR saves bf16 K / V / Q of layers 3, 31, 63 from the first prefill of >= 2048 tokens after DIR/GO
exists, once (real-data format analysis; eager only; GO keeps vLLM's dummy warm-up prefill out).
"""
import os
import re
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionMetadata,
                                       AttentionMetadataBuilder, AttentionType, CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

logger = init_logger("vllm.horizon_kvfp4")
D = 256
DATA = D // 2          # 128 bytes E2M1
SCL = D // 16          # 16 bytes E4M3
SLOT = DATA + SCL      # 144 bytes per (token, head slot)
K8_SLOT = D + DATA + SCL   # K8V4 (HORIZON_KVFP4_K8=1): 400 bytes per (token, kv head) = [256 K e4m3 | 128 V | 16 V sf]


def _k8() -> bool:
    return os.environ.get("HORIZON_KVFP4_K8", "0") == "1"
# decode configuration by batch size and decode-row length (HORIZON_KVFP4_DECODE_TABLE overrides; see __init__).
# From the 29-09-2026 standalone sweeps (bench_serving_gpu, t3 / t4): 4 token groups, no split partials beyond one CTA
# per SM, is the fastest or near-fastest everywhere except >16 requests with 4-token rows at 4k+ context, where the
# original one-group kernel wins (the draft schedule used here sends 4-token rows only up to 10 requests).
DECODE_TABLE = "16:128:4:2:1,1024x2:128:4:2:1,1024:256:1:3:4"


def _parse_entry(e, cps):
    k, *rest = e.split(":")
    maxb, _, maxql = k.partition("x")
    v = [int(x) for x in rest]
    if len(v) == 3:
        v.append(cps)
    return (int(maxb), int(maxql or 8), *v)
_STATE = {}            # per device: workspace, prefill scratch, signs; warm-up flags


def _K():
    from horizon_kvfp4.kernels import attn_decode_fp4 as AD
    from horizon_kvfp4.kernels import kv_fp4_ref as R
    from horizon_kvfp4.kernels import kv_fp4_serving as SV
    from horizon_kvfp4.kernels import kv_write_fp4 as KW
    return AD, R, SV, KW


def _env(name, default):
    return os.environ.get(name, default)


class HorizonFp4Backend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list] = ["nvfp4"]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
        assert spec.head_size == D and spec.head_size_v == D, "horizon_kvfp4 needs head_dim 256"
        if _k8():               # FP8 keys (vLLM's fp8 KV bytes) + FP4 values: one 400-byte slot per kv head
            return replace(spec, num_head_slots=spec.num_kv_heads, state_content_bytes=K8_SLOT)
        return replace(spec, num_head_slots=2 * spec.num_kv_heads, state_content_bytes=SLOT)

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"          # vLLM maps the name back to AttentionBackendEnum (registered as CUSTOM)

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16, 32, 64]

    @classmethod
    def supported_kv_cache_layouts(cls):
        # head-major blocks keep each head's 16-token K / V tile contiguous (the kernels accept any strides)
        return (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [D]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype == "nvfp4"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls():
        return HorizonFp4Impl

    @staticmethod
    def get_builder_cls():
        return HorizonFp4Builder


@dataclass
class HorizonFp4Metadata(AttentionMetadata):
    num_reqs: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor          # (num_reqs + 1,) int32, device
    seq_lens: torch.Tensor                 # (num_reqs,) int32, device (exact)
    block_table: torch.Tensor              # (num_reqs, max_blocks) int32, device
    slot_mapping: torch.Tensor
    has_decode: bool                       # some request has <= decode_threshold query tokens
    prefill_rows: list                     # [(row, q0, q1, seq_len)] from the CPU copies (exact for prefill rows)
    decode_threshold: int
    qlmax: int = 4                         # decode kernel variant: smallest of 1 / 2 / 4 >= the longest decode row


class HorizonFp4Builder(AttentionMetadataBuilder[HorizonFp4Metadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # MTP draft steps: the decode path reads only device tensors that the speculator advances in place (seq_lens,
    # query offsets, block table, slot mapping), so one build serves every draft step (as vLLM's Triton backend)
    supports_draft_decode_metadata_update: bool = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False,
              **kwargs) -> HorizonFp4Metadata:
        cam = common_attn_metadata
        n = cam.num_reqs
        thr = int(self.reorder_batch_threshold or 1)
        qsl = cam.query_start_loc_cpu[: n + 1].tolist()
        qlens = [qsl[i + 1] - qsl[i] for i in range(n)]
        rows = [i for i in range(n) if qlens[i] > thr]
        dec = [q for q in qlens if 0 < q <= thr]
        mx = max(dec) if dec else 1
        qlmax = next((c for c in (1, 2, 4, 8) if c >= mx and c <= thr), thr)
        prefill_rows = []
        if rows:
            ub = cam.seq_lens_cpu_upper_bound
            if ub is None:                      # V1 runner without the upper bound: exact copy (prefill steps only)
                ub = cam.seq_lens[:n].cpu()
            prefill_rows = [(i, qsl[i], qsl[i + 1], int(ub[i])) for i in rows]
        return HorizonFp4Metadata(
            num_reqs=n, num_actual_tokens=cam.num_actual_tokens, query_start_loc=cam.query_start_loc[: n + 1],
            seq_lens=cam.seq_lens[:n], block_table=cam.block_table_tensor[:n], slot_mapping=cam.slot_mapping,
            has_decode=len(rows) < n, prefill_rows=prefill_rows, decode_threshold=thr, qlmax=qlmax)


    def update_draft_decode_metadata(self, metadata: HorizonFp4Metadata) -> None:
        pass


def _layer_index(name: str):
    m = re.search(r"layers\.(\d+)\.", name or "")
    return int(m.group(1)) if m else -1


class HorizonFp4Impl(AttentionImpl[HorizonFp4Metadata]):
    supports_quant_query_input: bool = False
    can_return_lse_for_decode: bool = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype="auto", logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        assert head_size == D, "horizon_kvfp4 needs head_dim 256"
        assert kv_cache_dtype == "nvfp4", kv_cache_dtype
        assert alibi_slopes is None and sliding_window is None and not logits_soft_cap
        assert kv_sharing_target_layer_name is None, "KV sharing is not supported"
        assert kwargs.get("sinks") is None, "attention sinks are not supported"
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = None
        self.sliding_window = None
        self.G = self.num_heads // self.num_kv_heads
        self.had = _env("HORIZON_KVFP4_HAD", "0") == "1"
        self.k8 = _k8()
        assert not (self.k8 and self.had), "HORIZON_KVFP4_K8=1 keeps FP8 keys: the Hadamard (for FP4 keys) must be off"
        self.fused_rot = _env("HORIZON_KVFP4_FUSED_ROT", "1") == "1"
        self.search = _env("HORIZON_KVFP4_SEARCH", "1") == "1"
        self.cps = int(_env("HORIZON_KVFP4_CPS", "4"))
        self.chunk = int(_env("HORIZON_KVFP4_CHUNK", "65536"))
        # decode kernel configuration by batch size (graph-safe: it depends on the number of requests only):
        # "maxB[xmaxQL]:minc:tw:nst[:cps],..." = up to maxB requests (and decode rows of up to maxQL tokens, default
        # 8) use min-chunk minc, tw token groups, nst pipeline stages and a split plan for cps CTAs per SM (default
        # HORIZON_KVFP4_CPS); the first matching entry in (maxB, maxQL) order wins
        self.dtable = sorted(_parse_entry(e, self.cps)
                             for e in _env("HORIZON_KVFP4_DECODE_TABLE", DECODE_TABLE).split(","))
        self.batch_prefill = _env("HORIZON_KVFP4_BATCH_PREFILL", "1") == "1"
        self.cuda_write = _env("HORIZON_KVFP4_CUDA_WRITE", "1") == "1"
        # prefill compute dtype: fp16 (default) or bf16. The dequantised K / V are exact in both; fp16 keeps 3 more
        # mantissa bits in FA2's P matrix, its partial outputs and the chunk merges (29-09-2026 S3: bf16 0.0026..0.0029
        # with 2..3 chunks vs fp64, over the 0.0025 gate); only the final store rounds to bf16. Queries are converted
        # (Qwen's q is RMS-normed, far inside fp16's range).
        self.pdt = torch.float16 if _env("HORIZON_KVFP4_PF16", "1") == "1" else torch.bfloat16
        self._gk = self._gv = None
        self._views = None
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        spec = cfg.speculative_config
        nspec = (spec.num_speculative_tokens or 0) if spec is not None else 0
        self.qlmax = 1 + nspec
        assert self.G * self.qlmax <= 32, "G * (1 + drafts) must be <= 32"
        self.max_reqs = cfg.scheduler_config.max_num_seqs
        self.max_batched = cfg.scheduler_config.max_num_batched_tokens
        if not _STATE.get("logged"):
            _STATE["logged"] = True
            logger.info("horizon_kvfp4: NVFP4 KV cache backend active (heads %d, kv heads %d, decode rows up to %d "
                        "query tokens, hadamard=%d, k8v4=%d, scale search=%d, decode table %s, ctas/SM=%d, prefill "
                        "chunk %d)",
                        num_heads, self.num_kv_heads, self.qlmax, self.had, int(self.k8), self.search, self.dtable,
                        self.cps,
                        self.chunk)

    # ------------------------------------------------------------------------------------------------ helpers
    def _dhad(self):
        """1 when the decode kernel rotates q itself"""
        return int(self.had and self.fused_rot)

    def _dcfg(self, B, ql):
        """(minc, tw, nst, cps) for B requests and decode rows of up to ql tokens; token groups that do not fit one
        SM's shared memory for this head geometry fall back to 1 group and the default split plan"""
        AD, R, SV, KW = _K()
        minc, tw, nst, cps = next((e[2:] for e in self.dtable if B <= e[0] and ql <= e[1]), self.dtable[-1][2:])
        if SV.smem_bytes(self.G, ql, SV.default_ncw(self.G, ql), nst, tw, self._dhad(), int(self.k8)) > 99 * 1024:
            tw, nst, cps = 1, 3, self.cps
        return minc, tw, nst, cps

    def _state(self, device):
        st = _STATE.get(device)
        if st is None:
            AD, R, SV, KW = _K()
            need = 0
            for B in range(1, self.max_reqs + 1):
                S = SV.plan_splits(B, self.num_kv_heads, max([self.cps] + [e[5] for e in self.dtable]))
                need = max(need, B * self.num_kv_heads * S * 32 * (D + 1))
            st = _STATE[device] = {
                "ws": torch.empty(need, dtype=torch.float32, device=device),
                "signs": R.hadamard_signs(D).to(device) if self.had else None,
                "scratch": None,
            }
            logger.info("horizon_kvfp4: decode workspace %.1f MiB", need * 4 / 2 ** 20)
        return st

    def _cache_views(self, kv_cache: torch.Tensor):
        """(num_blocks, 2*KVH, PS, 144) uint8 logical view -> [page, pos, head, byte] data / scale views + flat"""
        v = self._views
        if v is not None and v[0] == kv_cache.data_ptr() and v[1] == tuple(kv_cache.stride()):
            return v[2]
        AD, R, SV, KW = _K()
        KVH = self.num_kv_heads
        t = kv_cache.permute(0, 2, 1, 3)                      # (blocks, PS, slots, bytes)
        if self.k8:                                           # (blocks, KVH, PS, 400)
            assert kv_cache.dtype == torch.uint8 and kv_cache.shape[1] == KVH and kv_cache.shape[3] == K8_SLOT, \
                kv_cache.shape
            kd, vd, vsf = t[..., :D], t[..., D:D + DATA], t[..., D + DATA:]
            ksf = vsf                                         # unused by the K8 kernels
        else:
            assert kv_cache.dtype == torch.uint8 and kv_cache.shape[1] == 2 * KVH
            assert kv_cache.shape[3] == SLOT, kv_cache.shape
            kd, ksf = t[:, :, :KVH, :DATA], t[:, :, :KVH, DATA:]
            vd, vsf = t[:, :, KVH:, :DATA], t[:, :, KVH:, DATA:]
        flat, strides = SV.flat_views(kd, ksf, vd, vsf)
        out = dict(kd=kd, ksf=ksf, vd=vd, vsf=vsf, flat=flat, strides=strides, PS=kv_cache.shape[2])
        self._views = (kv_cache.data_ptr(), tuple(kv_cache.stride()), out)
        return out

    def _scales(self, layer, device):
        if self._gk is None:
            self._gk = torch.full((self.num_kv_heads,), float(layer._k_scale_float), device=device)
            self._gv = torch.full((self.num_kv_heads,), float(layer._v_scale_float), device=device)
            if not _STATE.get("scales_logged"):
                _STATE["scales_logged"] = True
                logger.info("horizon_kvfp4: k_scale %g, v_scale %g (layer %s)", float(layer._k_scale_float),
                            float(layer._v_scale_float), getattr(layer, "layer_name", "?"))
        return self._gk, self._gv

    def process_weights_after_loading(self, act_dtype):
        """compile every kernel variant and JIT FlashInfer's bf16 FA2 prefill now, never inside a CUDA graph"""
        if _STATE.get("warm"):
            return
        _STATE["warm"] = True
        AD, R, SV, KW = _K()
        try:
            for ql in (1, 2, 4, 8):
                if ql <= self.qlmax:
                    for e in self.dtable:
                        if ql <= e[1]:
                            minc, tw, nst, _ = self._dcfg(e[0], ql)
                            SV.get_decode(self.num_kv_heads, self.G, ql, None, nst, minc, tw, self._dhad(),
                                          int(self.k8))
                    SV.get_combine(self.num_kv_heads, self.G, ql)
            f16 = self.pdt == torch.float16
            SV.get_dequant(self.num_kv_heads, f16, self.k8)
            SV.get_dequant_multi(self.num_kv_heads, f16, self.k8)
            if self._cuda_write():
                SV.get_write(self.num_kv_heads, self.search, self.had, self.k8)
            if self.had:
                AD.get_rotate(self.num_heads, D, 128, 0)
                AD.get_rotate(self.num_heads, D, 128, f16)
            from flashinfer import merge_state, single_prefill_with_kv_cache
            dev = torch.device("cuda", torch.cuda.current_device())
            q = torch.zeros(4, self.num_heads, D, dtype=self.pdt, device=dev)
            k = torch.zeros(8, self.num_kv_heads, D, dtype=self.pdt, device=dev)
            for causal in (True, False):
                o, lse = single_prefill_with_kv_cache(q, k, k, causal=causal, sm_scale=self.scale, backend="fa2",
                                                      return_lse=True)
            merge_state(o, lse, o, lse)
            if self.batch_prefill:
                w = self._ragged(dev)
                ind = torch.tensor([0, 4], dtype=torch.int32)
                w.plan(ind, torch.tensor([0, 8], dtype=torch.int32), self.num_heads, self.num_kv_heads, D, causal=True,
                       sm_scale=self.scale, q_data_type=self.pdt, kv_data_type=self.pdt)
                w.run(q, k, k)
            torch.cuda.synchronize()
            logger.info("horizon_kvfp4: kernels compiled, FlashInfer FA2 prefill JIT warm")
        except Exception:  # noqa: BLE001
            logger.exception("horizon_kvfp4: warm-up failed (kernels will compile on first use)")

    def _ragged(self, device, tag="main"):
        """one FlashInfer ragged FA2 prefill wrapper per (device, model: target / MTP draft), re-planned once per
        step by the first layer that needs it (the draft model has its own metadata, so its own wrapper)"""
        w = _STATE.get(("ragged", device, tag))
        if w is None:
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
            ws = _STATE.get(("ragged_ws", device))
            if ws is None:
                # FlashInfer's split-KV plan puts tmp_v = qo_heads * padded_batch * cta_tile_q * D * 4 bytes here
                # (29-09-2026: 138 MB for 8 short prefill rows among 32 requests overflowed the old 128 MB); 512 MB
                # covers padded_batch 128 at cta_tile_q 128 with 24 heads. vLLM's own FlashInfer backend (not used
                # with CUSTOM) would allocate 394 MB.
                mb = int(_env("HORIZON_KVFP4_PREFILL_WS_MB", "512"))
                ws = _STATE[("ragged_ws", device)] = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device=device)
            w = _STATE[("ragged", device, tag)] = BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
        return w

    def _dump(self, layer, name, t):
        d = os.environ.get("HORIZON_KVFP4_DUMP")
        # only after the arm creates DIR/GO (vLLM's warm-up runs prefill dummy tokens first, which are not real K/V)
        if not d or torch.cuda.is_current_stream_capturing() or not os.path.exists(os.path.join(d, "GO")):
            return
        li = _layer_index(getattr(layer, "layer_name", ""))
        if li not in (3, 31, 63) or "mtp" in getattr(layer, "layer_name", ""):
            return
        key = f"{li}_{name}"
        done = _STATE.setdefault("dumped", set())
        if key in done or t.shape[0] < 2048:
            return
        os.makedirs(d, exist_ok=True)
        torch.save(t[:8192].detach().to("cpu", torch.bfloat16).clone(), os.path.join(d, f"layer{li}_{name}.pt"))
        done.add(key)
        logger.info("horizon_kvfp4: dumped %s of layer %d (%d tokens) to %s", name, li, min(t.shape[0], 8192), d)

    # ------------------------------------------------------------------------------------------------- write
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        AD, R, SV, KW = _K()
        st = self._state(key.device)
        v = self._cache_views(kv_cache)
        gk, gv = self._scales(layer, key.device)
        n = slot_mapping.shape[0]
        self._dump(layer, "k", key[:n])
        self._dump(layer, "v", value[:n])
        if not self._cuda_write():
            KW.write(key, value, v["kd"], v["ksf"], v["vd"], v["vsf"], slot_mapping, gk, gv, st["signs"], self.search)
        else:
            k3 = key.view(key.shape[0], self.num_kv_heads, D) if key.dim() == 2 else key
            v3 = value.view(value.shape[0], self.num_kv_heads, D) if value.dim() == 2 else value
            SV.write(k3, v3, v["flat"], v["strides"], v["PS"], slot_mapping, gk, gv, self.search,
                     st["signs"] if self.had else None, self.k8)

    def _cuda_write(self):
        """CUDA write kernel (rotation fused when HAD; always for K8V4); else the TileLang kv_write_fp4"""
        return self.k8 or (self.cuda_write and (self.fused_rot or not self.had))

    # ----------------------------------------------------------------------------------------------- forward
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if attn_metadata is None:                 # profiling run
            return output
        assert output_scale is None and output_block_scale is None, "fused output quantisation is not supported"
        AD, R, SV, KW = _K()
        m = attn_metadata
        T = m.num_actual_tokens
        st = self._state(query.device)
        v = self._cache_views(kv_cache)
        gk, gv = self._scales(layer, query.device)
        q = query[:T]
        if not q.is_contiguous():
            q = q.contiguous()
        self._dump(layer, "q", q)
        qd = q
        if self.had and not self.fused_rot:
            q = qd = AD.rotate_q(q, st["signs"])          # old path: one rotation launch per layer
        if m.has_decode and m.num_reqs > 0:
            minc, tw, nst, cps = self._dcfg(m.num_reqs, m.qlmax)
            SV.decode_ragged(qd, v["flat"], v["strides"], v["PS"], m.block_table, m.seq_lens, m.query_start_loc,
                             self.scale, float(layer._v_scale_float), gk, gv, m.qlmax, output, st["ws"],
                             splits=SV.plan_splits(m.num_reqs, self.num_kv_heads, cps), nst=nst,
                             min_chunk=minc, tw=tw, signs=st["signs"] if self._dhad() else None, k8=self.k8)
        if m.prefill_rows:
            self._prefill(layer, q, v, m, output, st)
        return output

    def _prefill(self, layer, q, v, m, output, st):
        rows = m.prefill_rows
        cap = max(self.chunk, self.max_batched)
        if self.batch_prefill:
            plan = getattr(m, "_hz_pf", None)
            if plan is None:
                tag = "mtp" if "mtp" in getattr(layer, "layer_name", "") else "main"
                try:
                    plan = self._plan_batched(rows, cap, q.device, self.scale * float(layer._k_scale_float), tag)
                except Exception:  # noqa: BLE001  (eager mixed step: fall back to the per-request path, never crash)
                    if not _STATE.get("plan_warned"):
                        _STATE["plan_warned"] = True
                        logger.exception("horizon_kvfp4: batched prefill plan failed; per-request prefill this step")
                    plan = {"n_rows": 0, "rest": rows}
                m._hz_pf = plan
            if plan["n_rows"]:
                self._prefill_batched(layer, q, v, m, output, st, plan, cap)
            rows = plan["rest"]
        if rows:
            self._prefill_rows(layer, q, v, m, output, st, rows)

    def _plan_batched(self, rows, cap, device, sm, tag):
        """once per step (shared by every layer): rows whose whole context fits the scratch together -> one packed
        dequant + one ragged FA2 call per layer; the rest go through the per-request chunked path"""
        take, rest, tot = [], [], 0
        for r in rows:
            if r[3] <= cap - tot:
                take.append(r)
                tot += r[3]
            else:
                rest.append(r)
        plan = {"n_rows": len(take), "rest": rest}
        if not take:
            return plan
        qo, kv = [0], [0]
        for _, q0, q1, L in take:
            qo.append(qo[-1] + (q1 - q0))
            kv.append(kv[-1] + L)
        contiguous = all(take[i][2] == take[i + 1][1] for i in range(len(take) - 1))
        plan.update(n_tokens=tot, q_lo=take[0][1], q_hi=take[-1][2], contiguous=contiguous,
                    rows=torch.tensor([r[0] for r in take], dtype=torch.int32).to(device, non_blocking=True),
                    cu=torch.tensor(kv, dtype=torch.int32).to(device, non_blocking=True))
        if not contiguous:
            idx = torch.cat([torch.arange(q0, q1) for _, q0, q1, _ in take])
            plan["qidx"] = idx.to(device, non_blocking=True)
        w = self._ragged(device, tag)
        w.plan(torch.tensor(qo, dtype=torch.int32), torch.tensor(kv, dtype=torch.int32), self.num_heads,
               self.num_kv_heads, D, causal=True, sm_scale=sm, q_data_type=self.pdt, kv_data_type=self.pdt)
        plan["wrapper"] = w
        return plan

    def _prefill_batched(self, layer, q, v, m, output, st, plan, cap):
        AD, R, SV, KW = _K()
        sk, sv = self._scratch(st, cap, q.device)
        N = plan["n_tokens"]
        SV.dequant_multi(v["flat"], v["strides"], v["PS"], m.block_table, plan["rows"], plan["cu"], plan["n_rows"], N,
                         sk, sv, self.k8)
        qq = q[plan["q_lo"]:plan["q_hi"]] if plan["contiguous"] else q.index_select(0, plan["qidx"])
        o = plan["wrapper"].run(self._prefill_q(qq, st), sk[:N], sv[:N])
        vs = float(layer._v_scale_float)
        if vs != 1.0:
            o = o * vs
        if plan["contiguous"]:
            output[plan["q_lo"]:plan["q_hi"]].copy_(o)
        else:
            output.index_copy_(0, plan["qidx"], o.to(output.dtype))

    def _prefill_q(self, qq, st):
        """prefill query in the prefill dtype; with fused HAD rotated straight into it (one launch, like the cast)"""
        if self.had and self.fused_rot:
            AD, R, SV, KW = _K()
            return AD.rotate_q(qq.contiguous(), st["signs"], dtype=self.pdt)
        return qq.to(self.pdt)

    def _scratch(self, st, cap, device):
        sc = st["scratch"]
        if sc is None or sc[0].shape[0] < cap or sc[0].dtype != self.pdt:
            sc = st["scratch"] = (torch.empty(cap, self.num_kv_heads, D, dtype=self.pdt, device=device),
                                  torch.empty(cap, self.num_kv_heads, D, dtype=self.pdt, device=device))
        return sc

    def _prefill_rows(self, layer, q, v, m, output, st, rows):
        from flashinfer import merge_state, single_prefill_with_kv_cache
        AD, R, SV, KW = _K()
        sm = self.scale * float(layer._k_scale_float)
        vs = float(layer._v_scale_float)
        cap = max(self.chunk, self.max_batched)
        sc = self._scratch(st, cap, q.device)
        for row, q0, q1, L in rows:
            ql = q1 - q0
            qi = self._prefill_q(q[q0:q1], st)
            tail = max(0, L - max(self.chunk, ql))
            spans = [(a, min(a + self.chunk, tail)) for a in range(0, tail, self.chunk)] + [(tail, L)]
            o = s = None
            btrow = m.block_table[row]
            for a, b in spans:
                kk, vv = SV.dequant_tokens(v["flat"], v["strides"], v["PS"], btrow, a, b - a, sc[0], sc[1], self.k8)
                oc, lc = single_prefill_with_kv_cache(qi, kk, vv, causal=(b == L), sm_scale=sm, backend="fa2",
                                                      return_lse=True)
                if o is None:
                    o, s = oc, lc
                else:
                    o, s = merge_state(o, s, oc, lc)
            if vs != 1.0:
                o = o * vs
            output[q0:q1].copy_(o)
