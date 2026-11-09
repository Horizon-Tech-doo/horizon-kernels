# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Probe for the SM120 register-A lowering change (27-09-2026): compile upstream's own test kernel builder for a few
# register-A / shared-A configs, dump the CUDA source (for SASS inspection) and check exactness vs the fp32 reference.
#   gpu_lock.sh probe python probe_up.py <tag>   (cwd = $HK_WORK/rega/src, PYTHONPATH incl. src + testpkgs)
import os, sys, json
import torch
import tilelang
import tilelang.language as T
sys.path.insert(0, "testing/python/language")
import test_tilelang_language_nvf4_mma_block_scale as t

tag = sys.argv[1]
out = f"$HK_WORK/rega/probe_{tag}"
os.makedirs(out, exist_ok=True)
CASES = [
    # name, K, fragment_a, fragment_scales, policy, block_col_warps, warp_row_tiles
    ("rs_k256_fullrow", 256, True, False, T.GemmWarpPolicy.FullRow, 4, 64),
    ("rs_k256_square", 256, True, False, T.GemmWarpPolicy.Square, 4, 64),
    ("rs_k128_fullcol", 128, True, False, T.GemmWarpPolicy.FullCol, 4, 64),
    ("rs_k192_fragsf", 192, True, True, T.GemmWarpPolicy.FullRow, 4, 64),
    ("ss_k192_rowmajor", 192, False, False, T.GemmWarpPolicy.Square, 4, 64),
    ("ss_k128_rowmajor", 128, False, False, T.GemmWarpPolicy.Square, 2, 32),
]
for name, K, fa, fs, pol, bcw, wrt in CASES:
    torch.manual_seed(42)
    A, B = t._make_packed_fp4_inputs(128, 128, K, "random")
    SFA, sfa_bytes = t._make_varying_power_of_two_scale_words(128, K)
    SFB, sfb_bytes = t._make_varying_power_of_two_scale_words(128, K)
    ref = t._reference_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, 128, 128, K)
    kw = dict(num_stages=0, block_col_warps=bcw, warp_row_tiles=wrt, fragment_a=fa, fragment_scales=fs,
              sf_layout="rowmajor", policy=pol)
    if bcw == 2:
        kw.update(block_row_warps=2, warp_col_tiles=32, warp_row_tiles=32)
    try:
        kern = tilelang.compile(t._make_nvf4_matmul_codegen_kernel(128 if bcw == 4 else 64, 128 if bcw == 4 else 64, K, **kw),
                                target={"kind": "cuda", "arch": "sm_120a"}, out_idx=[4])
    except Exception as ex:  # noqa: BLE001
        print(json.dumps({"case": name, "compile_error": f"{type(ex).__name__}: {str(ex)[:300]}"}), flush=True)
        continue
    src = kern.get_kernel_source()
    open(f"{out}/{name}.cu", "w").write(src)
    mm = 128 if bcw == 4 else 64
    if mm != 128:
        A, B = A[:mm], B[:mm]
        SFA, SFB = SFA[:mm], SFB[:mm]
        ref = t._reference_blockscaled_gemm(A, B, sfa_bytes[:mm], sfb_bytes[:mm], mm, mm, K)
    C = kern(A, B, SFA, SFB)
    exact = bool(torch.equal(C, ref))
    torch.save(C.cpu(), f"{out}/{name}.pt")
    print(json.dumps({"case": name, "exact_vs_ref": exact, "mma_calls_in_src": src.count("sm120_mma_sync_blockscaled<"),
                      "max_abs_err": float((C - ref).abs().max())}), flush=True)
print("PROBE_DONE", tag)
