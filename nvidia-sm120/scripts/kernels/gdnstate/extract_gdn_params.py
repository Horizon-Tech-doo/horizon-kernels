# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Extract the small GDN gating parameters of every linear-attention layer from the Qwen3.8-27B-NVFP4 checkpoint with
# numpy only (no torch, no CUDA: safe on the gpushare host while the GPU is busy). Raw safetensors parsing: 8-byte
# header length, JSON header, then the data. bf16 is widened to fp32 via the upper 16 bits.
# Output npz: per GDN layer A_log (48,), dt_bias (48,), row norms of in_proj_a / in_proj_b (48,), input_layernorm (5120,).
import json, struct, sys
import numpy as np

CK = sys.argv[1] if len(sys.argv) > 1 else "$HK_DATA/models/Qwen3.8-27B-NVFP4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gdn_params.npz"
ix = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_hdr = {}


def header(fn):
    if fn not in _hdr:
        with open(f"{CK}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _hdr[fn] = (json.loads(f.read(n)), 8 + n)
    return _hdr[fn]


def load(name):
    fn = ix[name]
    h, off = header(fn)
    m = h[name]
    a, b = m["data_offsets"]
    with open(f"{CK}/{fn}", "rb") as f:
        f.seek(off + a)
        raw = f.read(b - a)
    dt = m["dtype"]
    if dt == "BF16":
        x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F32":
        x = np.frombuffer(raw, np.float32)
    elif dt == "F16":
        x = np.frombuffer(raw, np.float16).astype(np.float32)
    else:
        raise ValueError(dt)
    return x.reshape(m["shape"]).copy()


P = "model.language_model.layers.%d."
layers = sorted({int(k.split(".")[3]) for k in ix if ".linear_attn.A_log" in k})
out = {"layers": np.array(layers)}
for L in layers:
    p = P % L
    out[f"A_log_{L}"] = load(p + "linear_attn.A_log").astype(np.float32)
    out[f"dt_bias_{L}"] = load(p + "linear_attn.dt_bias").astype(np.float32)
    wa = load(p + "linear_attn.in_proj_a.weight")
    wb = load(p + "linear_attn.in_proj_b.weight")
    ln = load(p + "input_layernorm.weight")
    out[f"ln_{L}"] = ln
    g = 1.0 + ln                                   # Gemma-style RMSNorm (1 + w)
    out[f"a_norm_{L}"] = np.linalg.norm(wa * g[None, :], axis=1)
    out[f"b_norm_{L}"] = np.linalg.norm(wb * g[None, :], axis=1)
np.savez(OUT, **out)
print("layers", len(layers), "->", OUT)
