# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# S7 (formats track, 27-09-2026): "a new model gets the fastest kernel per shape automatically".
#
#   calibrate_model.py MODEL_DIR [--list] [--table-dir DIR] [--max-seqs 32] [--drafts 3] [--prefill 8192]
#
# 1. Enumerate (CPU only, no weights loaded): every linear weight of the checkpoint from the safetensors headers, its
#    format (NVFP4 compressed-tensors / NVFP4 ModelOpt / FP8 per-tensor / per-channel / block / BF16), and its role
#    (linear, lm_head, mtp draft, moe expert), fused the way vLLM fuses them (q/k/v -> qkv, gate/up -> gate_up,
#    in_proj_qkv + in_proj_z -> in_proj_qkvz, in_proj_a + in_proj_b -> in_proj_ba). --list stops here.
# 2. Calibrate (GPU; run under $HK_WORK/gpu_lock.sh with vLLM stopped): every (N, K) per format that has our kernels
#    races them against vLLM's stock kernel per batch bucket (horizon_kernels.calibration / fp8_linear.calibrate_shape:
#    CUDA-graph replay, > 512 MB rotated weight copies, correctness vs the reference variant, 2% exact / 10% lossy
#    margins) and writes the winners into --table-dir (default: a NEW directory, never production's).
#    Buckets per role: linear 1..16384 (prefill), lm_head up to max-seqs x (1 + drafts), mtp up to max-seqs.
# 3. Report: per shape the winner per bucket, and the layer-weighted time per decode step at 1 / 8 / 32 users.
# The vLLM plugin does step 2 by itself for any shape missing from its table at load (HORIZON_AUTO_CALIBRATE=1), so
# this script is for a fresh GPU / driver / kernel version, or to calibrate before the first start.
import argparse
import collections
import json
import os
import re
import sys

from safetensors import safe_open

FUSE = [(r"\.(q|k|v)_proj$", ".qkv_proj"), (r"\.(gate|up)_proj$", ".gate_up_proj"),
        (r"\.in_proj_(qkv|z)$", ".in_proj_qkvz"), (r"\.in_proj_(a|b)$", ".in_proj_ba")]


def headers(root):
    idx = os.path.join(root, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    out = {}
    for fn in files:
        with safe_open(os.path.join(root, fn), "pt", device="cpu") as f:
            for k in f.keys():
                s = f.get_slice(k)
                out[k] = (tuple(s.get_shape()), s.get_dtype())
    return out


def role_of(name):
    if name.startswith("lm_head") or name.endswith(".lm_head"):
        return "lm_head"
    if re.search(r"(^|\.)mtp\.", name):
        return "mtp"
    if ".experts." in name:
        return "expert"
    return "linear"


def enumerate_linears(root):
    h = headers(root)
    layers = {}
    for k, (shape, dt) in h.items():
        base = None
        if k.endswith(".weight_packed"):
            base, fmt = k[:-len(".weight_packed")], "nvfp4_ct"
            N, K = shape[0], shape[1] * 2
        elif k.endswith(".weight") and len(shape) == 2:
            base = k[:-len(".weight")]
            if "embed" in base or base.endswith("norm") or "conv1d" in base or "visual" in base:
                continue
            N, K = shape
            if dt == "U8" and base + ".weight_scale_2" in h:
                fmt, K = "nvfp4_modelopt", K * 2
            elif dt == "F8_E4M3":
                ss = h.get(base + ".weight_scale", ((), ""))[0]
                n = 1
                for d in ss:
                    n *= d
                fmt = "fp8_block" if len(ss) == 2 and ss[1] > 1 else ("fp8_channel" if n == N else "fp8_tensor")
            elif dt in ("BF16", "F16", "F32"):
                fmt = "bf16"
            else:
                fmt = "other:" + dt
        if base is None:
            continue
        layers[base] = (N, K, fmt)
    fused = collections.OrderedDict()
    for base, (N, K, fmt) in layers.items():
        name = base
        for pat, rep in FUSE:
            if re.search(pat, base):
                name = re.sub(pat, rep, base)
                break
        e = fused.setdefault(name, [0, K, fmt, []])
        e[0] += N
        e[3].append(base)
        if e[2] != fmt:
            e[2] = e[2] + "|" + fmt
    shapes = collections.Counter()
    for name, (N, K, fmt, parts) in fused.items():
        generic = re.sub(r"\.\d+\.", ".*.", name)
        shapes[(N, K, fmt, role_of(name), generic)] += 1
    return shapes


# ---------------------------------------------------------------------------------------------------------------
# Variant registry: per weight format, the implementations the plugin can pick, their numerics class and tolerance
# (relative error vs the reference variant), and which one is stock vLLM. Roles restrict classes: an LM head (logits)
# and anything verified token-by-token only takes "exact" variants unless its switch says otherwise.
REGISTRY = {
    "nvfp4": {"op": "qwen3_8_nvfp4_linear_nomarlin", "bytes_per_param": 0.5625,
              "variants": {"regfrag": ("exact", 0.0, "reference + default (drop-Marlin)"),
                           "fp4x2": ("near-exact", 0.02, "residual two-pass FP4 tensor cores, prefill"),
                           "fp4x1": ("lossy", 0.15, "one-pass W4A4, off unless HORIZON_ALLOW_LOSSY=1")},
              "stock": "marlin (not built with HORIZON_DROP_MARLIN=1)",
              "modules": ["tilelang/contrib/sm120_nvfp4", "tilelang/adaptive"]},
    "fp8": {"op": "horizon_fp8_linear", "bytes_per_param": 1.0,
            "variants": {"w8a16": ("exact", 0.0, "reference: BF16 activations x e4m3, no activation quant"),
                         "cutlass": ("near-exact", 0.10, "stock vLLM W8A8 (static activation quant) = default")},
            "stock": "cutlass", "modules": ["tilelang/contrib/sm120_fp8", "tilelang/adaptive"]},
    "bf16": {"op": None, "bytes_per_param": 2.0,
             "variants": {"cublas": ("exact", 0.0, "stock"),
                          "skinny": ("exact", 0.0, "K5a, N <= 512 and M <= 48 only (fixed rule, no table)")},
             "stock": "cublas", "modules": ["tilelang/contrib/horizon_fused"]},
}
ROLE_POLICY = {"lm_head": "exact only (logits); FP8 / NVFP4 head copies via HORIZON_LMHEAD_QUANT after their gate",
               "mtp": "drafts only (output cannot change): FP8 via HORIZON_MTP_QUANT, gate = acceptance",
               "expert": "MoE track (grouped kernels)", "linear": "table per bucket"}


def fmt_family(fmt):
    return "nvfp4" if fmt.startswith("nvfp4") else ("fp8" if fmt.startswith("fp8") else ("bf16" if fmt == "bf16" else fmt))


def kernel_source_hash():
    """sha256 (12 hex) over every kernel module the tables depend on (fork sources + the plugin's variant modules),
    read from disk without importing them: a kernel change gives a new table directory, so it is re-calibrated."""
    import hashlib
    import importlib.util
    h = hashlib.sha256()
    spec = importlib.util.find_spec("tilelang")
    root = os.path.dirname(spec.origin) if spec and spec.origin else None
    mods = sorted({m for r in REGISTRY.values() for m in r["modules"]})
    for m in mods:
        d = os.path.join(os.path.dirname(root), m) if root else None
        if d and os.path.isdir(d):
            for fn in sorted(f for f in os.listdir(d) if f.endswith(".py")):
                h.update(f"{m}/{fn}".encode()); h.update(open(os.path.join(d, fn), "rb").read())
    hs = importlib.util.find_spec("horizon_kernels")
    if hs and hs.origin:
        for fn in ("nvfp4_linear.py", "fp8_linear.py", "calibration.py"):
            p = os.path.join(os.path.dirname(hs.origin), fn)
            if os.path.exists(p):
                h.update(fn.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:12]


def report(by, table_dir, max_seqs, drafts):
    """Winners per bucket, % of the 1.8 TB/s roofline at M=1, and layer-weighted GEMM time per decode step."""
    import glob
    tables = {}
    for f in glob.glob(os.path.join(table_dir, "*.json")):
        d = json.load(open(f))
        tables[d.get("op")] = d
    ms = {"1 user": 1 + drafts, "8 users": 8 * (1 + drafts), f"{max_seqs} users": max_seqs * (1 + drafts)}
    step = {k: [0.0, 0.0, True] for k in ms}     # ours (best), stock, complete
    rows = []
    for (N, K, fmt, role), c in sorted(by.items()):
        fam = fmt_family(fmt)
        reg = REGISTRY.get(fam)
        r = {"N": N, "K": K, "format": fmt, "role": role, "count": c, "policy": ROLE_POLICY.get(role)}
        t = tables.get(reg["op"]) if reg and reg["op"] else None
        key = str((N, K))
        if t is None or key not in t.get("winners", {}):
            r["kernel"] = "stock / fixed rule (no table entry)" if reg else "no horizon variant"
            for k in step:
                step[k][2] = False
            rows.append(r); continue
        r["winners"] = t["winners"][key]
        tim = t.get("timings_us", {}).get(key, {})
        if "1" in tim:
            best = min(v for n, v in tim["1"].items() if n in reg["variants"] and reg["variants"][n][0] != "lossy")
            r["pct_roofline_M1"] = round(100 * N * K * reg["bytes_per_param"] / (best * 1e-6) / 1.8e12, 1)
        for k, m in ms.items():
            probe = min((int(p) for p in tim if int(p) >= m), default=None)
            if probe is None:
                step[k][2] = False; continue
            cand = {n: v for n, v in tim[str(probe)].items() if n in reg["variants"] and reg["variants"][n][0] != "lossy"}
            if role == "lm_head":
                cand = {n: v for n, v in cand.items() if reg["variants"][n][0] == "exact"}
            if cand:
                step[k][0] += c * min(cand.values())
            st = tim[str(probe)].get(reg["stock"].split()[0])
            if st is not None:
                step[k][1] += c * st
        rows.append(r)
    for r in rows:
        print(json.dumps(r))
    for k, (ours, stock, complete) in step.items():
        print(json.dumps({"decode_step": k, "rows_per_gemm": ms[k], "ours_gemm_us": round(ours, 1),
                          "stock_gemm_us_where_measured": round(stock, 1) if stock else None,
                          "all_shapes_in_table": complete}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--list", action="store_true", help="enumerate shapes and applicable variants (CPU only)")
    ap.add_argument("--report", action="store_true", help="report from existing tables in --table-dir (CPU only)")
    ap.add_argument("--table-dir", default=None, help="parent directory; tables go to <dir>/k-<kernel source hash>")
    ap.add_argument("--max-seqs", type=int, default=32)
    ap.add_argument("--drafts", type=int, default=3)
    a = ap.parse_args()
    shapes = enumerate_linears(a.model)
    by = collections.defaultdict(int)
    print(f"{'count':>5} {'N':>7} {'K':>6}  {'format':<16} {'role':<8} {'variants':<28} name")
    for (N, K, fmt, role, gen), c in sorted(shapes.items(), key=lambda x: (x[0][3], x[0][2], -x[1])):
        reg = REGISTRY.get(fmt_family(fmt))
        var = ",".join(reg["variants"]) if reg else "-"
        print(f"{c:>5} {N:>7} {K:>6}  {fmt:<16} {role:<8} {var:<28} {gen}")
        by[(N, K, fmt.split("|")[0], role)] += c
    kh = kernel_source_hash()
    print(f"kernel source hash: {kh}")
    if a.list:
        return
    if not a.table_dir:
        sys.exit("--table-dir is required (never production's table directory)")
    tdir = os.path.join(a.table_dir, f"k-{kh}")
    os.makedirs(tdir, exist_ok=True)
    if a.report:
        report(by, tdir, a.max_seqs, a.drafts)
        return
    os.environ["TILELANG_ADAPTIVE_DIR"] = tdir
    import torch  # noqa: F401
    lm_max = a.max_seqs * (1 + a.drafts)
    for (N, K, fmt, role), c in sorted(by.items()):
        fam = fmt_family(fmt)
        if role == "expert" or fam not in ("nvfp4", "fp8"):
            continue                                   # experts: MoE track; BF16: fixed rules (skinny) / stock
        if fam == "nvfp4":
            from horizon_kernels.calibration import calibrate_shape
            from horizon_kernels.nvfp4_linear import make_op
            op = make_op(allow_lossy=False)
            limit = lm_max if role == "lm_head" else (a.max_seqs if role == "mtp" else 1 << 30)
            calibrate_shape(op, N, K, buckets=[b for b in op.buckets if b <= limit])
        else:
            from horizon_kernels import fp8_linear
            fp8_linear.calibrate_shape(fp8_linear.make_op(), N, K)
    json.dump({"model": a.model, "kernel_source_hash": kh, "shapes": [list(k) + [c] for k, c in by.items()]},
              open(os.path.join(tdir, "manifest.json"), "w"), indent=1)
    report(by, tdir, a.max_seqs, a.drafts)
    print(f"TILELANG_ADAPTIVE_DIR={tdir}")


if __name__ == "__main__":
    main()
