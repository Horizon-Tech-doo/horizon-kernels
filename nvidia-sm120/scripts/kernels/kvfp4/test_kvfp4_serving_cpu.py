# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for kv_fp4_serving (vLLM serving kernels) and the backend's cache layout. numpy + torch only.

usage: python test_kvfp4_serving_cpu.py [PATH_TO_horizon_ops] [--json OUT.json]
  C1 dequant formulas (e2m1v / e4m3v of kv_fp4_serving, FTZ-safe): exhaustive over all 16 codes x all 256 scale
     bytes, equal to kv_fp4_ref's decode (0x7f read as 480, as the decode kernel's sf117 does)
  C2 slot-interleaved cache layout (vLLM (blocks, 2*KVH, PS, 144) uint8, slot = [128 data | 16 scales], HND and NHD
     physical orders): the backend's [page, pos, head, byte] views address disjoint bytes that cover the slots exactly
  C3 lane-level emulation of decode_ragged (the v2 emulation of test_kvfp4_v2_cpu.py with the ragged row mapping:
     per-request query offsets from query_start_loc, 0 .. QLMAX tokens, rows with more tokens skipped, CUDA-graph
     padding rows with 0 tokens) + the ragged combine, against a per-request fp64 reference over the same dequantised
     cache: relative L2 <= 0.0025 for every decode row; skipped rows untouched
The emulation is derived from the v2 emulation source by explicit, asserted text substitutions (so both share the
same fragment / mma / movmatrix model); the GPU gate test_kvfp4_serving_gpu.py is the real check.
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
JSON_OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if JSON_OUT in ARGS:
    ARGS.remove(JSON_OUT)
PKG = ARGS[0] if ARGS else os.path.expanduser("$HK_SRC/wt-kvfp4/tilelang/contrib/horizon_ops")
sys.argv = [sys.argv[0], PKG]
spec = importlib.util.spec_from_file_location("v2cpu", os.path.join(HERE, "test_kvfp4_v2_cpu.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
R = V.R
U32 = np.uint32
RESULTS = {}


def record(name, ok, **kw):
    RESULTS[name] = dict(ok=bool(ok), **kw)
    print(json.dumps({"test": name, "ok": bool(ok), **kw}), flush=True)
    assert ok, name


def c1():
    import struct

    def u2f(u):
        return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

    def e2m1v(c):
        e, l = (c >> 1) & 3, c & 1
        v = u2f(((e + 126) << 23) | (l << 22)) if e else (0.5 if l else 0.0)
        return -v if c & 8 else v

    def e4m3v(b):
        e, m = (b >> 3) & 15, b & 7
        return u2f(((e + 120) << 23) | (m << 20)) if e else m * 0.001953125
    codes_ok = all(e2m1v(c) == R.e2m1_value(torch.tensor([c], dtype=torch.uint8)).item() for c in range(16))
    sf_ok = all(e4m3v(b) == V.sf_val(b) for b in range(256))
    record("C1_dequant_formulas", codes_ok and sf_ok, e2m1_codes=16, scale_bytes=256)


def views_of(kv, KVH):
    t = kv.transpose(0, 2, 1, 3)                       # (blocks, PS, 2*KVH, 144)
    return t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:]


def c2():
    ok = True
    for layout in ("HND", "NHD"):
        NP, KVH, PS = 3, 4, 16
        base = np.zeros(NP * 2 * KVH * PS * 144, dtype=np.int64)
        if layout == "HND":
            kv = base.reshape(NP, 2 * KVH, PS, 144)
        else:
            kv = base.reshape(NP, PS, 2 * KVH, 144).transpose(0, 2, 1, 3)
        for v in views_of(kv, KVH):
            v += 1                                      # every byte must be hit exactly once
        ok = ok and bool((base == 1).all())
    record("C2_slot_views_partition", ok)


def ragged_emulator():
    import inspect
    src = inspect.getsource(V.emulate)
    subs = [
        ("def emulate(qbits, kd, ksf, vd, vsf, bt, sl, gk, gv, scale_log2, KVH, G, QL, NCW, NST, MINC, PS, S, banks,\n"
         "            mutate=None):",
         "def emulate_ragged(qbits, kd, ksf, vd, vsf, bt, sl, qsl, gk, gv, scale_log2, KVH, G, QL, NCW, NST, MINC, PS,\n"
         "                   S, banks, mutate=None):"),
        ("        L = int(sl[b])\n",
         "        q0 = int(qsl[b]); ql = int(qsl[b + 1]) - q0\n        ql = 0 if (ql < 0 or ql > QL) else ql\n"
         "        L = int(sl[b]) if ql > 0 else 0\n"),
        ("                        okr = row < Rr\n                        tq, hq",
         "                        okr = (row < Rr) & (row // G < ql)\n                        tq, hq"),
        ("qr = qbits[b * QL + tq, h * G + hq]", "qr = qbits[q0 + tq, h * G + hq]"),
        ("ok = (row < Rr) & (pos < en) & (pos < L - QL + row // G + 1)",
         "ok = (row < Rr) & (row // G < ql) & (pos < en) & (pos < L - ql + row // G + 1)"),
    ]
    for a, b in subs:
        assert src.count(a) == 1, ("substitution not unique", a[:60])
        src = src.replace(a, b)
    ns = dict(vars(V))
    exec(src, ns)
    return ns["emulate_ragged"]


def combine_ragged(opart, lse, qsl, NT, KVH, G, QLMAX, v_scale):
    Bn, _, S = lse.shape[:3]
    out = np.full((NT, KVH * G, 256), np.nan)
    for b in range(Bn):
        q0, ql = int(qsl[b]), int(qsl[b + 1]) - int(qsl[b])
        if ql < 0 or ql > QLMAX:
            continue
        for h in range(KVH):
            for r in range(G * min(ql, QLMAX)):
                ls = lse[b, h, :, r].astype(np.float32)
                mx = np.float32(max(ls))
                w = np.where(ls == -np.inf, 0, np.exp2(ls - mx)).astype(np.float32)
                tot = np.float32(w.sum())
                acc = np.zeros(256, np.float32)
                for k in range(S):
                    acc = acc + opart[b, h, k, r] * (w[k] / tot if w[k] else np.float32(0))
                out[q0 + r // G, h * G + r % G] = torch.tensor(acc * np.float32(v_scale)).to(torch.bfloat16).double()
    return out


def c3():
    emu = ragged_emulator()
    worst, recs = 0.0, []
    for i, (KVH, G, lens, qls, S, NCW, layout) in enumerate((
            (4, 6, [300, 17, 5, 0, 900, 64], [4, 2, 1, 0, 9, 3], 3, 2, "HND"),
            (2, 8, [130, 400, 33], [1, 4, 6], 2, 4, "NHD"),
            (4, 6, [1000, 1], [2, 1], 4, 2, "NHD"))):
        gen = torch.Generator().manual_seed(70 + i)
        PS = 64
        B = len(lens)
        pages = [max(1, math.ceil(L / PS)) for L in lens]
        NP = sum(pages) + 2
        perm = torch.randperm(NP - 1, generator=gen) + 1
        bt = torch.zeros(B, max(pages), dtype=torch.int32)
        o = 0
        for j, p in enumerate(pages):
            bt[j, :p] = perm[o:o + p].int()
            o += p
        if layout == "HND":
            kvt = torch.full((NP, 2 * KVH, PS, 144), 0x7F, dtype=torch.uint8)
        else:
            kvt = torch.full((NP, PS, 2 * KVH, 144), 0x7F, dtype=torch.uint8).permute(0, 2, 1, 3)
        t = kvt.permute(0, 2, 1, 3)
        views = (t[:, :, :KVH, :128], t[:, :, :KVH, 128:], t[:, :, KVH:, :128], t[:, :, KVH:, 128:])
        slots = torch.tensor([int(bt[j, p // PS]) * PS + p % PS for j, L in enumerate(lens) for p in range(L)])
        NT = sum(lens)
        key = torch.randn(NT, KVH, 256, generator=gen).to(torch.bfloat16)
        val = torch.randn(NT, KVH, 256, generator=gen).to(torch.bfloat16)
        gk = torch.tensor([1.0, 0.5, 2.0, 1.0][:KVH])
        gv = torch.tensor([1.0, 2.0, 1.0, 0.25][:KVH])
        R.write_reference(key, val, views, slots, gk, gv, None)
        qsl = [0]
        for ql in qls:
            qsl.append(qsl[-1] + ql)
        q = torch.randn(qsl[-1], KVH * G, 256, generator=gen).to(torch.bfloat16)
        qbits = q.view(torch.int16).numpy().view(np.uint16)
        sm = 256 ** -0.5
        kd, ksf, vd, vsf = (v.numpy() for v in views)
        op, ls = emu(qbits, kd, ksf, vd, vsf, bt.numpy(), np.array(lens), np.array(qsl), gk.numpy(), gv.numpy(),
                     sm * 1.4426950408889634, KVH, G, 4, NCW, 3, 64, PS, S, V.Banks())
        out = combine_ragged(op, ls, qsl, qsl[-1], KVH, G, 4, 1.0)
        for j in range(B):
            q0, q1, L, ql = qsl[j], qsl[j + 1], lens[j], qls[j]
            if ql == 0:
                continue
            if ql > 4:
                assert np.isnan(out[q0:q1]).all(), "prefill row written"
                continue
            K, Vv = R.gather(views, bt[j], L)
            ref = np.zeros((ql, KVH * G, 256))
            for tt in range(ql):
                lim = L - ql + tt + 1
                qq = q[q0 + tt].double().view(KVH, G, 256)
                s = torch.einsum("hgd,lhd->hgl", qq, K[:lim]) * gk.double().view(KVH, 1, 1) * sm
                ref[tt] = torch.einsum("hgl,lhd->hgd", torch.softmax(s, -1),
                                       Vv[:lim] * gv.double().view(1, KVH, 1)).reshape(-1, 256).numpy()
            r = float(np.linalg.norm(out[q0:q1] - ref) / np.linalg.norm(ref))
            worst = max(worst, r)
        recs.append({"KVH": KVH, "G": G, "lens": lens, "qls": qls, "layout": layout, "S": S, "NCW": NCW})
    record("C3_ragged_decode_emulation", worst <= 0.0025, worst_rel_l2=worst, gate=0.0025, cases=recs)


if __name__ == "__main__":
    c1()
    c2()
    c3()
    print(json.dumps({"ALL_SERVING_CPU_TESTS": "PASS", "n": len(RESULTS)}), flush=True)
    if JSON_OUT:
        with open(JSON_OUT, "w") as fh:
            json.dump(RESULTS, fh, indent=1)
