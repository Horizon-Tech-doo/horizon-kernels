"""Drive upstream examples/amd/example_amd_flash_attn_fwd.py config-by-config (no autotuner),
so compile/runtime errors are visible. Correctness vs the example's torch reference, then timing.

Usage: python run_fa.py <tilelang_src> --dim 128 --seq_len 4096 --heads 8 [--groups 1] [--causal]
"""
import argparse, json, os, re, runpy, sys, traceback
from functools import partial

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--batch", type=int, default=1); ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--seq_len", type=int, default=4096); ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--groups", type=int, default=1); ap.add_argument("--causal", action="store_true")
a = ap.parse_args()

code = open(os.path.join(a.src, "examples/amd/example_amd_flash_attn_fwd.py")).read()
code, n = re.subn(r"^@tilelang\.autotune\(.*\)\n", "", code, flags=re.M)
assert n == 1, "autotune decorator not found; upstream example changed"
dst = os.path.abspath("fa_fwd_noautotune.py")
open(dst, "w").write(code)
mod = runpy.run_path(dst, run_name="fa_example")
tilelang = mod["tilelang"]

CONFIGS = [dict(block_M=bm, block_N=bn, num_split_q=128, threads=256, num_stages=st, enable_rasterization=True,
                k_pack=2, panel_size=7, qk_coalesced_width=8, v_coalesced_width=4)
           for bm, bn, st in [(64, 64, 0), (64, 32, 0), (128, 32, 0), (128, 64, 0)]]
CONFIGS += [dict(c, k_pack=1) for c in CONFIGS[:2]]
flops = 4.0 * a.batch * a.heads * a.seq_len * a.seq_len * a.dim * (0.5 if a.causal else 1.0)
ref = partial(mod["ref_program"], is_causal=a.causal, groups=a.groups)
ref_ms = None
for cfg in CONFIGS:
    r = {"dim": a.dim, "heads": a.heads, "groups": a.groups, "seq": a.seq_len, "causal": a.causal, "cfg": cfg}
    try:
        k = mod["fast_flashattn"](a.batch, a.heads, a.seq_len, a.dim, a.causal, a.groups, **cfg)
        prof = k.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
        prof.assert_allclose(ref, rtol=0.01, atol=0.01)
        r["ok"] = True
        if ref_ms is None:
            ref_ms = prof.do_bench(ref, warmup=20)
        ms = prof.do_bench(warmup=50)
        r.update(tl_ms=ms, tl_tflops=flops / ms * 1e-9, torch_ref_ms=ref_ms, torch_ref_tflops=flops / ref_ms * 1e-9)
    except Exception as ex:  # noqa
        r["ok"] = False
        r["error"] = (type(ex).__name__ + ": " + str(ex))[-800:]
        traceback.print_exc(limit=3, file=sys.stdout)
    print(json.dumps(r), flush=True)
