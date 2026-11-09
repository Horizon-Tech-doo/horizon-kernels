# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the Qwen3.8-Flash-Next expert-pruning tools (03-10-2026).

Pure Python + optional numpy. No torch, no safetensors package: the pruning tool copies tensor BYTES, so it must not
depend on a library knowing every dtype (NVFP4 weights are U8, their block scales F8_E4M3, MXFP8 scales U8/F8_E8M0).

Contents
  * safetensors header read / streaming write (raw bytes, 8-byte aligned header, no holes);
  * checkpoint tensor-name grammar for Qwen4Exp (`Qwen4ExpForConditionalGeneration`):
        model.language_model.layers.<L>.mlp.experts.<E>.<gate|up|down>_proj.<weight|weight_scale|weight_scale_2|
                                                                             input_scale|weight_scale_inv>
        model.language_model.layers.<L>.mlp.gate.weight            router [num_experts, hidden], BF16
        mtp.layers.<L>.mlp.experts.<E>....                         the MTP layer: same num_experts (vLLM reads the
        mtp.layers.<L>.mlp.gate.weight                              shared text config), so it is pruned too
    (checked against the local-inference-lab and NVIDIA NVFP4 index files, 03-10-2026);
  * mapping vLLM MoE-runner layer names (`language_model.model.layers.5.mlp.experts`) to checkpoint prefixes;
  * numpy dequantizers (NVFP4 E2M1 + E4M3 block-16 scale + FP32 global scale; FP8 128x128 block; BF16) used by the
    offline cross-check and the unit tests.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

# experts stored one tensor per expert (both NVFP4 checkpoints, and the BF16 original)
EXPERT_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<eid>\d+)\.(?P<rest>.+)$")
# experts stored fused along dim 0 (other exporters): <prefix>.experts.<name>  with shape [num_experts, ...]
FUSED_EXPERT_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.(?P<layer>\d+)\.mlp)\.experts\.(?P<rest>[A-Za-z_][\w.]*)$")
# the router: <prefix>.gate.<weight|bias|...>   (not gate_proj, not shared_expert_gate: both lack the '.gate.' form)
GATE_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.(?P<layer>\d+)\.mlp)\.gate\.(?P<rest>.+)$")

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


# ---------------------------------------------------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: list
    start: int          # absolute file offset of the first byte
    nbytes: int
    path: str


def read_header(path: str) -> tuple[dict, int, dict]:
    """-> (tensors {name: TensorInfo}, data_start, metadata)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (short)")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > 512 * 1024 * 1024:
            raise ValueError(f"{path}: implausible header length {n}")
        hdr = json.loads(f.read(n))
    data_start = 8 + n
    meta = hdr.pop("__metadata__", {}) or {}
    out = {}
    for name, d in hdr.items():
        a, b = d["data_offsets"]
        ti = TensorInfo(name, d["dtype"], list(d["shape"]), data_start + a, b - a, path)
        sz = DTYPE_SIZE.get(ti.dtype)
        if sz is not None:
            numel = 1
            for s in ti.shape:
                numel *= s
            if numel * sz != ti.nbytes:
                raise ValueError(f"{path}: {name}: {ti.dtype}{ti.shape} needs {numel * sz} bytes, header says {ti.nbytes}")
        out[name] = ti
    return out, data_start, meta


def read_bytes(ti: TensorInfo, offset: int = 0, length: int | None = None) -> bytes:
    length = ti.nbytes - offset if length is None else length
    with open(ti.path, "rb") as f:
        f.seek(ti.start + offset)
        b = f.read(length)
    if len(b) != length:
        raise IOError(f"{ti.path}: short read for {ti.name}")
    return b


@dataclass
class OutTensor:
    """One tensor of an output shard. `chunks` = list of (path, absolute_offset, nbytes) copied in order."""
    name: str
    dtype: str
    shape: list
    chunks: list = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(c[2] for c in self.chunks)


def write_safetensors(path: str, tensors: list[OutTensor], metadata: dict | None = None, bufsize: int = 64 << 20) -> int:
    """Stream tensors into a new safetensors file (written to path + '.partial', then renamed). Returns bytes written.
    Peak memory is one copy buffer, whatever the tensor sizes."""
    hdr: dict = {}
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    off = 0
    for t in tensors:
        sz = DTYPE_SIZE.get(t.dtype)
        if sz is not None:
            numel = 1
            for s in t.shape:
                numel *= s
            if numel * sz != t.nbytes:
                raise ValueError(f"{t.name}: {t.dtype}{t.shape} needs {numel * sz} bytes, chunks give {t.nbytes}")
        hdr[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
        off += t.nbytes
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path + ".partial"
    written = 0
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        written += 8 + len(hb)
        for t in tensors:
            for (src, start, n) in t.chunks:
                with open(src, "rb") as f:
                    f.seek(start)
                    left = n
                    while left:
                        b = f.read(min(bufsize, left))
                        if not b:
                            raise IOError(f"{src}: short read while copying {t.name}")
                        out.write(b)
                        left -= len(b)
                        written += len(b)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    return written


def load_index(model_dir: str) -> tuple[dict, dict]:
    """-> (weight_map {tensor: shard file name}, index metadata). Single-file checkpoints get a synthetic map."""
    p = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return dict(d["weight_map"]), dict(d.get("metadata", {}))
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        t, _, _ = read_header(single)
        return {k: "model.safetensors" for k in t}, {}
    raise FileNotFoundError(f"{model_dir}: no model.safetensors.index.json or model.safetensors")


# ---------------------------------------------------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------------------------------------------------

def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def expert_layout(names) -> dict:
    """{moe prefix: {"eids": set, "per_expert_rest": set, "fused": set}} for every MoE block in the name list."""
    out: dict = {}
    for n in names:
        m = EXPERT_RE.match(n)
        if m:
            d = out.setdefault(m["prefix"], {"eids": set(), "per_expert_rest": set(), "fused": set()})
            d["eids"].add(int(m["eid"]))
            d["per_expert_rest"].add(m["rest"])
            continue
        m = FUSED_EXPERT_RE.match(n)
        if m:
            d = out.setdefault(m["prefix"], {"eids": set(), "per_expert_rest": set(), "fused": set()})
            d["fused"].add(m["rest"])
    return out


def vllm_layer_to_prefix(layer_name: str, num_hidden_layers: int) -> str:
    """vLLM MoE-runner name -> checkpoint MoE prefix.
    Main model: '...layers.<L>.mlp.experts'           -> 'model.language_model.layers.<L>.mlp'
    MTP drafter: layer index >= num_hidden_layers, or 'mtp' in the name
                 (vLLM numbers MTP layers from num_hidden_layers) -> 'mtp.layers.<L - num_hidden_layers>.mlp'"""
    m = re.search(r"layers\.(\d+)\.", layer_name + ".")
    if not m:
        raise ValueError(f"no layer index in {layer_name!r}")
    L = int(m.group(1))
    if L >= num_hidden_layers or re.search(r"(^|\.)mtp(\.|$)", layer_name):
        return f"mtp.layers.{L - num_hidden_layers if L >= num_hidden_layers else L}.mlp"
    return f"model.language_model.layers.{L}.mlp"


# ---------------------------------------------------------------------------------------------------------------------
# dequantization (numpy; offline cross-check and tests)
# ---------------------------------------------------------------------------------------------------------------------

def _np():
    import numpy as np  # noqa: PLC0415
    return np


def e4m3_table():
    """float32 value of every float8_e4m3fn byte (bias 7, no inf, 0x7F/0xFF = NaN)."""
    np = _np()
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            t[b] = np.nan
        elif e == 0:
            t[b] = s * (m / 8.0) * 2.0 ** (-6)
        else:
            t[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return t


def e2m1_table():
    np = _np()
    t = np.zeros(16, dtype=np.float32)
    for v in range(16):
        t[v] = (-1.0 if v & 0x8 else 1.0) * E2M1_VALUES[v & 0x7]
    return t


def bf16_to_f32(u16):
    np = _np()
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    """Round-to-nearest-even float32 -> bf16 bits (uint16)."""
    np = _np()
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)


def dequant_nvfp4(packed, block_scale_u8, global_scale: float, group: int = 16):
    """ModelOpt NVFP4: packed uint8 [out, in/2] (element 2k in the LOW nibble, 2k+1 in the high nibble, as vLLM's
    break_fp4_bytes), block scales float8_e4m3 [out, in/group] (stored linearly in the checkpoint; vLLM swizzles at
    load), one float32 global scale (`weight_scale_2`). -> float32 [out, in]."""
    np = _np()
    packed = np.asarray(packed, dtype=np.uint8)
    out_dim, half = packed.shape
    lo = packed & 0x0F
    hi = packed >> 4
    codes = np.stack([lo, hi], axis=-1).reshape(out_dim, half * 2)
    vals = e2m1_table()[codes]
    sc = e4m3_table()[np.asarray(block_scale_u8, dtype=np.uint8)]
    if sc.shape != (out_dim, half * 2 // group):
        raise ValueError(f"block scale shape {sc.shape} does not match weight {(out_dim, half * 2)} / {group}")
    return vals * np.repeat(sc, group, axis=1) * np.float32(global_scale)


def dequant_fp8_block(w_u8, scale_inv, block: int = 128):
    """FP8 E4M3 weight [out, in] with float32 block scales [ceil(out/b), ceil(in/b)] (`weight_scale_inv`)."""
    np = _np()
    w = e4m3_table()[np.asarray(w_u8, dtype=np.uint8)]
    s = np.asarray(scale_inv, dtype=np.float32)
    s_full = np.repeat(np.repeat(s, block, axis=0), block, axis=1)[: w.shape[0], : w.shape[1]]
    return w * s_full


def quant_nvfp4_reference(w, group: int = 16):
    """Tiny reference NVFP4 quantizer for the tests only (round to nearest E2M1, E4M3 block scales).
    -> (packed uint8, block scales uint8 (e4m3 bytes), global scale float)."""
    np = _np()
    w = np.asarray(w, dtype=np.float32)
    amax = float(np.abs(w).max()) or 1.0
    gscale = amax / (6.0 * 448.0)
    blocks = w.reshape(w.shape[0], -1, group)
    bmax = np.abs(blocks).max(axis=-1) / 6.0 / gscale
    tab = e4m3_table()
    finite = np.where(np.isfinite(tab) & (tab >= 0))[0]
    sc_bytes = finite[np.abs(tab[finite][None, None, :] - bmax[..., None]).argmin(-1)].astype(np.uint8)
    sc = tab[sc_bytes] * gscale
    sc[sc == 0] = 1.0
    scaled = blocks / sc[..., None]
    e2 = e2m1_table()
    codes = np.abs(scaled[..., None] - e2[None, None, None, :]).argmin(-1).astype(np.uint8)
    codes = codes.reshape(w.shape)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, sc_bytes, gscale
