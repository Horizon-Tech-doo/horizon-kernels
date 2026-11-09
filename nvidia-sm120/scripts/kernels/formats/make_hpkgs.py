# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Build $HK_WORK/formats/hpkgs: the ModelOpt-adapter plugin copy ($HK_WORK/nvours/hpkgs) + the formats-track modules
# (fp8_linear.py, lmhead.py, mtp_quant.py) with their register() hooks. Production's plugin is not touched.
#   python3 make_hpkgs.py SRC_DIR_WITH_NEW_MODULES
import os, shutil, sys
SRC = os.path.expanduser("$HK_WORK/nvours/hpkgs")
DST = os.path.expanduser("$HK_WORK/formats/hpkgs")
NEW = sys.argv[1]
if os.path.exists(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__"))
for m in ("fp8_linear.py", "lmhead.py", "mtp_quant.py", "ckpt_mix.py", "gdn_h_fix.py"):
    shutil.copy(os.path.join(NEW, m), os.path.join(DST, "horizon_kernels", m))
p = os.path.join(DST, "horizon_kernels", "vllm_plugin.py")
s = open(p).read()
a = '''        from horizon_kernels import k5
        k5.install()
'''
assert s.count(a) == 1
s = s.replace(a, a + '''    if os.environ.get("HORIZON_FP8", "0") == "1" or os.environ.get("HORIZON_FP8_PERCOL", "0") == "1":  # fp8_linear.py
        from horizon_kernels import fp8_linear
        fp8_linear.install()
    if os.environ.get("HORIZON_LMHEAD_QUANT") or os.environ.get("HORIZON_DUMP_LMHEAD_H"):  # horizon_kernels/lmhead.py
        from horizon_kernels import lmhead
        lmhead.install()
    if os.environ.get("HORIZON_MTP_QUANT", ""):   # quantized MTP draft layer (horizon_kernels/mtp_quant.py)
        from horizon_kernels import mtp_quant
        mtp_quant.install()
    if os.environ.get("HORIZON_GDN_H_FIX", "0") == "1":   # fp8fix track: FLA fwd-h Blackwell config fix (gdn_h_fix.py)
        from horizon_kernels import gdn_h_fix
        gdn_h_fix.install()
    if os.environ.get("HORIZON_MIX_MLP_FROM", "") or os.environ.get("HORIZON_MIX_NVFP4_FROM_CT", ""):  # ckpt_mix.py
        from horizon_kernels import ckpt_mix
        ckpt_mix.install()
''')
b = "        mods = [ad, nv]\n"
assert s.count(b) == 1
s = s.replace(b, b + '''        if os.environ.get("HORIZON_FP8", "0") == "1" or os.environ.get("HORIZON_MTP_QUANT") \\
                or os.environ.get("HORIZON_LMHEAD_QUANT") == "fp8":
            import tilelang.contrib.sm120_fp8 as hf8
            mods.append(hf8)
''')
open(p, "w").write(s)
print("built", DST)
