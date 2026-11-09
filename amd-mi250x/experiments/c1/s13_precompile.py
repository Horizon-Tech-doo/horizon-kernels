# Compile-only, no GPU: fills the cache for Experiments B and C.
import os, sys, runpy
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
from s13_kernel import build
SH = {'4k': (4096, 4096, 4096), '17k': (4096, 17408, 5120), '32k': (32768, 1024, 2048), '8k': (8192, 8192, 8192), '16k': (16384, 16384, 16384)}
for sn, names in eval(sys.argv[1]).items():
    for n in names:
        k = build(n, *SH[sn]); print(sn, n, k.get_kernel_source().count('a_r_0'), k.resource_usage, flush=True)
