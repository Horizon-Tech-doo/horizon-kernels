# Python-level C1 path on the unmodified wheel: rewrite the emitted device source before hipcc.
# Uses the existing tilelang_callback_hip_postproc hook (src/rocm/codegen/rt_mod_hip.cc:78).
import os, re, sys, json, glob
sys.path.insert(0, '/scratch/project_XXXXXXXXX/tilelang/exp/E1')
import torch, tilelang, tvm_ffi
from gemm_prefetch import gemm
def wpe(blocks, T):  # waves per EU for `blocks` blocks per CU
    return (blocks * ((T + 63) // 64) + 3) // 4
BLOCKS = int(os.environ.get('C1_BLOCKS', '2'))
def postproc(code, target):
    return re.sub(r'__launch_bounds__\((\d+)\)', lambda m: f'__launch_bounds__({m.group(1)}, {wpe(BLOCKS, int(m.group(1)))})', code)
tvm_ffi.register_global_func("tilelang_callback_hip_postproc", postproc, override=True)
k = gemm(8192, 8192, 8192, 256, 128, 32, 256, 2, 'base')
s = k.get_kernel_source()
print([l.strip()[:80] for l in s.splitlines() if 'launch_bounds' in l])
d = os.environ['TILELANG_CACHE_DIR']
for f in sorted(glob.glob(d + '/0.1.15/*/kernels/*/device_kernel.cu'), key=os.path.getmtime)[-1:]:
    print(f[-60:], open(f).read().count('__launch_bounds__(256, 2)'))
    print(json.load(open(os.path.dirname(f) + '/resource_usage.json')))
