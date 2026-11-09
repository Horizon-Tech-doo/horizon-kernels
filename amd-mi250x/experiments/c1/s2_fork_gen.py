# Compile-only, fork build: annotated kernels through the real JIT. Reads the JIT's own cached compiler metadata (resource_usage.json).
import os, re, json, glob, time
import torch, tilelang
from s3_kernel import gemm
SH = {'4k': (4096, 4096, 4096), '8k': (8192, 8192, 8192), '32k': (32768, 1024, 2048), '17k': (4096, 17408, 5120), '16k': (16384, 16384, 16384)}
d = os.environ['TILELANG_CACHE_DIR']
os.makedirs('s2co', exist_ok=True)
def latest():
    return max(glob.glob(d + '/*/*/kernels/*/device_kernel.cu'), key=os.path.getmtime)
print('shape arm T blocks bounds V A scratch Occ')
for sn, (M, N, K) in SH.items():
    for nm, bM, bN, bK, th, bl in [('b25632', 256, 128, 32, 256, 0), ('b25632_a2', 256, 128, 32, 256, 2), ('b25632_a3', 256, 128, 32, 256, 3), ('b12864_a1', 128, 128, 64, 256, 1), ('b12832T128_a2', 128, 128, 32, 128, 2), ('b25632T512_a1', 256, 128, 32, 512, 1)]:
        if sn != '8k' and nm in ('b12832T128_a2', 'b25632T512_a1', 'b25632_a3', 'b12864_a1'):
            continue
        k = gemm(M, N, K, bM, bN, bK, th, 2, bl)
        s = k.get_kernel_source()
        f = latest()
        c = open(f).read()
        assert c == s, 'latest cache entry is not this kernel'
        r = json.load(open(os.path.dirname(f) + '/resource_usage.json'))['base_kernel']['extra']
        lb = re.search(r'__launch_bounds__\([^)]*\)', s).group(0)
        print(sn, nm, th, bl, lb.replace(' ', ''), r['VGPRs'], r['AGPRs'], r['ScratchSize [bytes/lane]'], r['Occupancy [waves/SIMD]'], flush=True)
        if nm == 'b25632_a2':
            open(f's2co/{sn}.cc', 'w').write(s)
