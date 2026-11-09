# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# API-surface check for the rebase: every upstream TileLang name our fork files use (T.<name>, PassConfigKey.<name>,
# and the few direct imports) must still exist; its signature is recorded so the two builds can be diffed.
#   python api_check.py <out.json>        (run once per build, then diff the JSON files)
import inspect, json, re, sys
from pathlib import Path
import tilelang
import tilelang.language as T

root = Path(tilelang.__file__).parent
files = [p for d in ("contrib/sm120_nvfp4", "contrib/horizon_fused", "adaptive") for p in (root / d).glob("*.py")
         if not p.name.startswith("._")]          # production tl-pkgs has macOS AppleDouble "._*.py" files
src = "\n".join(p.read_text() for p in files)
names = sorted(set(re.findall(r"\bT\.([A-Za-z_][A-Za-z0-9_]*)", src)))
pcks = sorted(set(re.findall(r"PassConfigKey\.([A-Z_0-9]+)", src)))
out = {"tilelang": str(root), "T": {}, "PassConfigKey": {}, "imports": {}, "modules": {}}
for n in names:
    obj = getattr(T, n, None)
    if obj is None:
        out["T"][n] = "MISSING"
        continue
    try:
        out["T"][n] = str(inspect.signature(obj))
    except (TypeError, ValueError):
        out["T"][n] = type(obj).__name__
for n in pcks:
    out["PassConfigKey"][n] = str(getattr(tilelang.PassConfigKey, n, "MISSING"))
for mod, attr in (("tilelang.cuda.intrinsics.macro.mma_macro_generator", "TensorCoreIntrinEmitter"),
                  ("tilelang.cuda.intrinsics.layout.mma_layout", "make_mma_swizzle_layout"),
                  ("tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator", "TensorCoreIntrinEmitterSM120"),
                  ("tilelang.cuda.op.gemm.gemm_mma_sm120", "GemmMMASm120BlockScaled"),
                  ("tvm.tirx", "BufferRegion"), ("tvm.ir", "Range")):
    try:
        m = __import__(mod, fromlist=[attr])
        o = getattr(m, attr)
        try:
            out["imports"][f"{mod}.{attr}"] = str(inspect.signature(o))
        except (TypeError, ValueError):
            out["imports"][f"{mod}.{attr}"] = "present"
    except Exception as ex:  # noqa: BLE001
        out["imports"][f"{mod}.{attr}"] = f"MISSING {type(ex).__name__}: {ex}"
for m in ("tilelang.contrib.sm120_nvfp4", "tilelang.contrib.sm120_nvfp4.decode_regfrag", "tilelang.contrib.sm120_nvfp4.decode_regfrag_xl",
          "tilelang.contrib.sm120_nvfp4.prefill_rf", "tilelang.contrib.sm120_nvfp4.decode_gemv", "tilelang.contrib.sm120_nvfp4.decode_fp4x3",
          "tilelang.contrib.sm120_nvfp4.decode_shuffled", "tilelang.contrib.sm120_nvfp4.decode_twiddle",
          "tilelang.contrib.sm120_nvfp4.decode_v2smem", "tilelang.contrib.sm120_nvfp4.decode",
          "tilelang.contrib.horizon_fused.skinny_bf16", "tilelang.contrib.horizon_fused.add_rmsnorm", "tilelang.adaptive"):
    try:
        __import__(m); out["modules"][m] = "ok"
    except Exception as ex:  # noqa: BLE001
        out["modules"][m] = f"IMPORT FAIL {type(ex).__name__}: {ex}"
json.dump(out, open(sys.argv[1], "w"), indent=1)
miss = [k for k, v in out["T"].items() if v == "MISSING"] + [k for k, v in out["imports"].items() if v.startswith("MISSING")] \
    + [k for k, v in out["modules"].items() if v != "ok"] + [k for k, v in out["PassConfigKey"].items() if "MISSING" in v]
print(json.dumps({"names": len(names), "missing": miss}))
