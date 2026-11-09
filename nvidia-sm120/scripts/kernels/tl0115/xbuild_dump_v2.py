# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Cross-build bit-exactness dump, v2 (30-09-2026, TileLang v0.1.15 rebase). Same method as tlnext/xbuild_dump.py
# (27-09-2026): run every PRODUCTION TileLang kernel on fixed, seeded inputs and record a SHA-256 of each output's
# bytes. Run once with the production prefix (tl-pkgs-cand-29-09-2026) and once with the new build
# (tl-pkgs-0115-30-09-2026); xbuild_compare.py diffs the two JSON files. Identical hashes = bit-identical outputs.
# v2 covers: regfrag_xl on the 5 main shapes + the NVFP4 MTP fc
# (5120x10240), the native-order FP4 prefill (prefill_rs.linear_native, rsN3) with the torch row scales AND with
# horizon_ops.row_scale installed (HORIZON_PREFILL_RS=1), row_scale.quantize_rs itself, K3 (reference for natpf), K5,
# K1a from the v030 plugin, and the adaptive dispatcher with the production natpf table + HORIZON_NATIVE_PREFILL=1.
#   gpu_lock.sh ... python xbuild_dump_v2.py <out.json> [--quick]
import hashlib, json, os, sys, time
import torch

OUT = sys.argv[1]
QUICK = "--quick" in sys.argv
PLUGIN = "$HK_DATA/horizon-pkgs-v030-29-09-2026"
sys.path.insert(0, PLUGIN)
os.environ["HORIZON_DROP_MARLIN"] = "1"
os.environ["HORIZON_NATIVE_PREFILL"] = "1"
os.environ.setdefault("TILELANG_ADAPTIVE_DIR", "$HK_DATA/horizon-pkgs/adaptive-natpf-07ee3f21")
import tilelang
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_ops import row_scale as rsc
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as an

res = {"_meta": {"tilelang_file": tilelang.__file__, "tilelang_version": getattr(tilelang, "__version__", "?"),
                 "torch": torch.__version__, "start": time.strftime("%d-%m-%Y %H:%M:%S")}}
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
MTP_FC = {"mtp_fc": (5120, 10240)}


def h(*ts):
    m = hashlib.sha256()
    for t in ts:
        t = t.detach().contiguous()
        m.update(str((tuple(t.shape), str(t.dtype))).encode())
        m.update(t.view(torch.uint8).cpu().numpy().tobytes() if t.dtype != torch.bool else t.cpu().numpy().tobytes())
    return m.hexdigest()[:24]


def rec(key, *ts, **extra):
    torch.cuda.synchronize()
    res[key] = {"sha": h(*ts), **extra}


def xin(M, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(M, K, device="cuda", generator=g) * torch.rand(M, 1, device="cuda", generator=g) * 3).to(torch.bfloat16)


t0 = time.time()
# 1. regfrag_xl: every M 1..1024 on the 5 main shapes + the MTP fc (all tuned buckets, rf passes and padded twins)
def xl_ms(N, K):
    """every M 1..130, then every tuned bucket edge (m_max, m_max + 1) and every multiple of 16 +- 1 up to 1024"""
    ms = set(range(1, 131)) | {m for m16 in range(144, 1025, 16) for m in (m16 - 1, m16, m16 + 1)}
    ms |= {m for m_max, _ in xl._TUNED.get((N, K), ()) for m in (m_max, m_max + 1)}
    return sorted(m for m in ms if 1 <= m <= 1024)


for name, (N, K) in {**SHAPES, **MTP_FC}.items():
    XL_MS = xl_ms(N, K) if not QUICK else (1, 4, 16, 17, 32, 33, 64, 65, 128, 512, 1000, 1024)
    packed, scale, alpha = synthetic_layer(N, K, seed=1)
    prep = xl.prepare(packed, scale, alpha)
    for M in XL_MS:
        rec(f"xl:{name}:M{M}", xl.linear(xin(M, K, 100 + M), prep))
    del prep
    torch.cuda.empty_cache()
    print(json.dumps({"stage": f"regfrag_xl {name}", "secs": round(time.time() - t0)}), flush=True)

# 2. prefill: K3 (checkpoint layout) and the native rsN3 path, torch row scales and horizon_ops.row_scale
PF_MS = (1, 17, 128, 512, 513, 1000, 1024, 1025, 2048, 4096, 8192) if not QUICK else (513, 4096)
torch_quantize_rs = prs.quantize_rs
for name, (N, K) in SHAPES.items():
    g = torch.Generator(device="cuda").manual_seed(N + K)
    wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
    gs = torch.tensor([448.0 * 6], device="cuda")
    w0 = nv.prepare_weight(wp, ws, gs)
    rec(f"k3prep:{name}", w0.packed, w0.sfb, w0.alpha)
    wn = prs.prepare_weight_native(wp, ws, gs)
    rec(f"natprep:{name}", wn.w, wn.sw, wn.alpha)
    for M in PF_MS:
        x = xin(M, K, 200 + M)
        rec(f"k3:{name}:M{M}:p2", nv.linear(x, w0, passes=2))
        prs.quantize_rs = torch_quantize_rs
        rec(f"rsq_torch:{name}:M{M}", *torch_quantize_rs(x, 2))
        rec(f"nat:{name}:M{M}:p2:torchrs", prs.linear_native(x, wn, passes=2))
        prs.quantize_rs = rsc.quantize_rs                  # HORIZON_PREFILL_RS=1 (horizon_kernels.ops)
        rec(f"rsq_rowscale:{name}:M{M}", *rsc.quantize_rs(x, 2))
        rec(f"nat:{name}:M{M}:p2:rowscale", prs.linear_native(x, wn, passes=2))
        prs.quantize_rs = torch_quantize_rs
    del w0, wn, wp, ws
    torch.cuda.empty_cache()
print(json.dumps({"stage": "k3+native prefill+row_scale", "secs": round(time.time() - t0)}), flush=True)

# 3. K5: skinny BF16 GEMM (GDN in_proj_ba, 96 x 5120) and add + Gemma RMSNorm (H 5120)
g = torch.Generator(device="cuda").manual_seed(7)
W = (torch.randn(96, 5120, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
for M in range(1, sk.SUPPORTS_M + 1):
    rec(f"k5skinny:M{M}", sk.linear(xin(M, 5120, 300 + M), W))
wn5 = (torch.randn(5120, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
for M in (1, 2, 3, 4, 8, 16, 32, 33, 64, 128, 512, 2048):
    x = xin(M, 5120, 400 + M); r = (xin(M, 5120, 500 + M).float() * 3).to(torch.bfloat16)
    y, ro = an.add_rmsnorm(x, r, wn5, 1e-6)
    rec(f"k5norm:M{M}", y, ro)
    rec(f"k5rms:M{M}", an.rmsnorm(x, wn5, 1e-6))
print(json.dumps({"stage": "k5", "secs": round(time.time() - t0)}), flush=True)

# 4. K1a fused FP8 draft head (production: 49152 x 5120, act fp8, BM 16), M 1..16
from horizon_kernels.draft_head import DraftHeadFP8
Wd = (torch.randn(49152, 5120, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
head = DraftHeadFP8(Wd, act="fp8")
for M in (1, 2, 3, 4, 8, 16):
    ids = head(xin(M, 5120, 600 + M))
    pv, pi = head.kernel(head.Apad, head.W, head.SApad, head.SW)
    rec(f"k1a:M{M}", ids, pv, pi)
del head, Wd
torch.cuda.empty_cache()
print(json.dumps({"stage": "k1a", "secs": round(time.time() - t0)}), flush=True)

# 5. adaptive dispatcher: production natpf no-Marlin table, native prefill; selection per (shape, M) + output
from horizon_kernels.nvfp4_linear import make_op
from horizon_kernels.calibration import make_layer
prs.quantize_rs = rsc.quantize_rs
op = make_op()
for name, (N, K) in SHAPES.items():
    lay = make_layer(N, K, seed=11, with_marlin=False)
    for M in ((1, 4, 32, 64, 128, 256, 512, 513, 1024, 1025, 2048, 4096) if not QUICK else (4, 513, 2048)):
        v = op.select((N, K), M)
        rec(f"adaptive:{name}:M{M}", op((N, K), M, xin(M, K, 700 + M), lay), variant=v.name)
    del lay
    torch.cuda.empty_cache()
res["_meta"]["table"] = str(op.table_path)
res["_meta"]["secs"] = round(time.time() - t0)
json.dump(res, open(OUT, "w"), indent=0)
print(json.dumps({"DUMP_DONE": OUT, "cases": len(res) - 1, "secs": round(time.time() - t0)}), flush=True)
