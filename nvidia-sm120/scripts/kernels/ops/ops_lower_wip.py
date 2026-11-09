# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Offline lowering of the WIP copy (tilelang.contrib.horizon_ops_wip) of the ops kernels, incl. G2 verify / commit.
import faulthandler, os, subprocess, sys, time
faulthandler.dump_traceback_later(1500, exit=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.contrib.horizon_ops_wip import gdn_decode as gd, gdn_recover as gr, attn_decode as ad, qk_norm_rope as qk, row_scale as rsc, conv1d_update as cvu, attn_prefill as apf, gdn_prefill as gpf, kv_write as kvw
out = sys.argv[1]; sel = sys.argv[2] if len(sys.argv) > 2 else ""
os.makedirs(out, exist_ok=True)
TG = Target({"kind": "cuda", "arch": "sm_120a"})
SS = 48 * 128 * 128 + 1024
CASES = [("gdn_vr32", gd.gdn_decode_kernel, (48, 16, 32, 16384, 96, 96, 16384, SS, "bfloat16", "bfloat16", False)),
         ("verify_vr32", gr.gdn_verify_kernel, (48, 16, 32, 16384, 96, 96, 16384, SS, "bfloat16", "bfloat16", False)),
         ("verify_vr128", gr.gdn_verify_kernel, (48, 16, 128, 16384, 96, 96, 16384, SS, "bfloat16", "bfloat16", False)),
         ("commit_vr32", gr.gdn_commit_kernel, (48, 16, 32, SS)),
         ("attn_split_ql4", ad.attn_split_kernel, (4, 6, 4, 256, 1600, 64, 32, 128, 256, 4 * 1600 * 512, 512, 1600 * 512)),
         ("attn_split_pipe32", ad.attn_split_kernel, (4, 6, 4, 256, 1600, 32, 32, 128, 256, 4 * 1600 * 512, 512, 1600 * 512, 1)),
         ("attn_split_ql1", ad.attn_split_kernel, (4, 6, 1, 256, 1600, 64, 32, 128, 256, 4 * 1600 * 512, 512, 1600 * 512)),
         ("attn_combine_ql4", ad.attn_combine_kernel, (4, 6, 4, 256)),
         ("qk_norm_rope", qk.qk_norm_rope_kernel, (24, 4, 256, 64, 14336, 14336)),
         ("row_scale_5120", rsc.row_scale_kernel, (5120,)), ("row_scale_17408", rsc.row_scale_kernel, (17408,)),
         ("quant_nopad_5120", rsc.quant_rs_nopad_kernel, (5120, 2)),
         ("conv_update", cvu.conv_update_kernel, (10240, 16384, 10240 * 6 + 64, 6, 1, 16384)),
         ("attn_prefill", apf.attn_prefill_kernel, (4, 6, 256, 1600, 16, 32, 0, 4 * 1600 * 512, 512, 1600 * 512)),
         ("attn_prefill_s64", apf.attn_prefill_kernel, (4, 6, 256, 1600, 16, 64, 1, 4 * 1600 * 512, 512, 1600 * 512)),
         ("attn_prefill_v2", apf.attn_prefill_kernel_v2, (4, 6, 256, 1600, 8, 32, 4 * 1600 * 512, 512, 1600 * 512)),
         ("attn_prefill_v3", apf.attn_prefill_kernel_v3, (4, 6, 256, 1600, 16, 32, 4 * 1600 * 512, 512, 1600 * 512)),
         ("gdnpf_wy", gpf.wy_kernel, (16, 48)), ("gdnpf_pass", gpf.pass_kernel, (16, 48)),
         ("kv_write", kvw.kv_write_kernel, (4, 256, 1600, 14336, 256, 14336, 256, 4 * 1600 * 512, 512, 1600 * 512))]
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
for name, jf, args in CASES:
    if sel and sel not in name:
        continue
    t0 = time.time()
    try:
        tir = jf.get_tir(*args)
        with TG:
            with tilelang.transform.PassContext(config=jf.pass_configs or {}):
                art = tilelang.lower(tir, target=TG)
        src = art.kernel_source if hasattr(art, "kernel_source") else str(art)
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"{name} LOWER-FAIL {type(e).__name__}: {str(e)[:1500]}", flush=True); continue
    cu = f"{out}/{name}.cu"; open(cu, "w").write(src)
    r = subprocess.run([get_nvcc_compiler(), "--cubin", "-O3", "-arch=sm_120a", "-std=c++20", "-w",
                        "-I" + TILELANG_TEMPLATE_PATH, "-I" + CUTLASS_INCLUDE_DIR, "-Xptxas", "-v", "-o", cu[:-3] + ".cubin", cu],
                       capture_output=True, text=True)
    regs = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "registers" in l or "spill" in l or "error" in l]
    print(f"{name}: {time.time() - t0:.1f} s rc={r.returncode}", *regs[:4], sep="\n   ", flush=True)
print("OPS_LOWER_WIP_DONE", flush=True)
