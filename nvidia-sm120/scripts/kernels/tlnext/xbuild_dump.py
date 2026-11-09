# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Cross-build bit-exactness dump (27-09-2026): run every live horizon_kernels TileLang kernel on fixed, seeded inputs
# and record a SHA-256 of each output's bytes. Run once with the production TileLang ($HK_DATA/tl-pkgs) and once
# with the rebased build ($HK_DATA/tl-pkgs-next); xbuild_compare.py then diffs the two JSON files. Identical hashes
# = bit-identical outputs. Kernels: regfrag_xl (all W4A16 decode), K3 (sm120_nvfp4.quantize + linear, passes 1 and 2),
# prefill_rf (one-weight-copy prefill), K5 (skinny BF16 GEMM, add+RMSNorm), K1a (fused FP8 draft head) and the
# adaptive dispatcher (horizon_kernels.nvfp4_linear.make_op with the production no-Marlin table: selection + output).
#   gpu_lock.sh xbuild-dump python xbuild_dump.py <out.json> [--quick]
import hashlib, json, os, sys, time
import torch

OUT = sys.argv[1]
QUICK = "--quick" in sys.argv
import tilelang
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rfm
from tilelang.contrib.sm120_nvfp4 import prefill_rf as prf
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as an

res = {"_meta": {"tilelang_file": tilelang.__file__, "tilelang_version": getattr(tilelang, "__version__", "?"),
                 "torch": torch.__version__, "start": time.strftime("%d-%m-%Y %H:%M:%S")}}
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}


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
# 1. regfrag_xl: every bucket edge used by the plugin warm-up range (1..1024), 5 shapes
XL_MS = (1, 2, 3, 4, 7, 8, 12, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 49, 56, 63, 64, 65, 80, 96, 112, 128, 144, 160,
         176, 192, 200, 208, 224, 240, 256, 288, 320, 352, 384, 448, 512, 576, 640, 700, 768, 896, 1000, 1024)
if QUICK:
    XL_MS = (1, 4, 32, 33, 128, 512, 1024)
for name, (N, K) in SHAPES.items():
    packed, scale, alpha = synthetic_layer(N, K, seed=1)
    prep = xl.prepare(packed, scale, alpha)
    for M in XL_MS:
        rec(f"xl:{name}:M{M}", xl.linear(xin(M, K, 100 + M), prep))
    del prep
print(json.dumps({"stage": "regfrag_xl", "secs": round(time.time() - t0)}), flush=True)

# 2. K3 (quantizer + FP4 GEMM, passes 2 and 1) and prefill_rf on the same random layer
K3_MS = (128, 512, 1000, 1024, 4096, 8192) if not QUICK else (512, 4096)
for name, (N, K) in SHAPES.items():
    g = torch.Generator(device="cuda").manual_seed(N + K)
    wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
    gs = torch.tensor([448.0 * 6], device="cuda")
    w0 = nv.prepare_weight(wp, ws, gs)
    rec(f"k3prep:{name}", w0.packed, w0.sfb, w0.alpha)
    prep, w1 = prf.prepare_layer(wp, ws, gs)
    rec(f"rfprep:{name}", prep.Wg, prep.Sg, w1.sfb, w1.alpha)
    for M in K3_MS:
        x = xin(M, K, 200 + M)
        A, SFA, rsinv = nv.quantize(x, 2)
        rec(f"k3quant:{name}:M{M}", A, SFA, rsinv)
        for p in (2, 1):
            rec(f"k3:{name}:M{M}:p{p}", nv.linear(x, w0, passes=p))
            rec(f"rf:{name}:M{M}:p{p}", prf.linear_rf(x, w1, passes=p))
    del w0, w1, prep, wp, ws
    torch.cuda.empty_cache()
print(json.dumps({"stage": "k3+prefill_rf", "secs": round(time.time() - t0)}), flush=True)

# 3. K5: skinny BF16 GEMM (GDN in_proj_ba, 96 x 5120) and add + Gemma RMSNorm (H 5120)
g = torch.Generator(device="cuda").manual_seed(7)
W = (torch.randn(96, 5120, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
for M in (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48):
    rec(f"k5skinny:M{M}", sk.linear(xin(M, 5120, 300 + M), W))
wn = (torch.randn(5120, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
for M in (1, 4, 8, 16, 32, 64, 128, 512):
    x = xin(M, 5120, 400 + M); r = (xin(M, 5120, 500 + M).float() * 3).to(torch.bfloat16)
    y, ro = an.add_rmsnorm(x, r, wn, 1e-6)
    rec(f"k5norm:M{M}", y, ro)
    rec(f"k5rms:M{M}", an.rmsnorm(x, wn, 1e-6))
print(json.dumps({"stage": "k5", "secs": round(time.time() - t0)}), flush=True)

# 4. K1a fused FP8 draft head (48k draft vocabulary x 5120, fp8 act), M 1..8
sys.path.insert(0, "$HK_DATA/horizon-pkgs")
from horizon_kernels.draft_head import DraftHeadFP8
Wd = (torch.randn(49152, 5120, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
head = DraftHeadFP8(Wd, act="fp8")
for M in (1, 2, 4, 8, 16):
    hx = xin(M, 5120, 600 + M)
    ids = head(hx)
    a, sa = torch.zeros(16, 5120, dtype=torch.float8_e4m3fn, device="cuda"), torch.ones(16, device="cuda")
    q, s = head.W, head.SW
    pv, pi = head.kernel(head.Apad, q, head.SApad, s)          # raw kernel outputs of the last call's padded input
    rec(f"k1a:M{M}", ids, pv, pi)
del head, Wd
torch.cuda.empty_cache()
print(json.dumps({"stage": "k1a", "secs": round(time.time() - t0)}), flush=True)

# 5. adaptive dispatcher: production no-Marlin table, selection per (shape, M) and the dispatched output
os.environ["HORIZON_DROP_MARLIN"] = "1"
from horizon_kernels.nvfp4_linear import make_op
from horizon_kernels.calibration import make_layer
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
