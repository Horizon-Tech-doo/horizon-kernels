# Generate device sources through the real JIT (v0.1.15 wheel) at several thread counts. No fork.
import os, sys
sys.path.insert(0, '/scratch/project_XXXXXXXXX/tilelang/exp/E1')
import torch, tilelang
from gemm_prefetch import gemm
os.makedirs('s1src', exist_ok=True)
for T, (bM, bN, bK) in {64: (64,64,32), 128: (128,128,32), 256: (256,128,32), 512: (256,128,32), 768: (256,128,32)}.items():
    k = gemm(8192, 8192, 8192, bM, bN, bK, T, 2, 'base')
    s = k.get_kernel_source()
    open(f's1src/t{T}.cu', 'w').write(s)
    print(T, len(s), [l.strip()[:90] for l in s.splitlines() if 'launch_bounds' in l], flush=True)
