# GEMM through TileLang's JIT (python call path). argv: mode(wheel|hook|fork) round
# Arms: the same `base` kernel as E1 gemm_prefetch.py, varying only tile + occupancy route.
import json, os, re, sys, time
import torch, tilelang
import tilelang.rocm.language as T
mode, rnd = sys.argv[1], int(sys.argv[2])
if mode == 'hook':
    import tvm_ffi
    tvm_ffi.register_global_func("tilelang_callback_hip_postproc",
        lambda code, target: re.sub(r'__launch_bounds__\(256\)', '__launch_bounds__(256, 2)', code), override=True)

from s3_kernel import gemm

SH = {'4k': (4096, 4096, 4096, 40), '8k': (8192, 8192, 8192, 40), '32k': (32768, 1024, 2048, 40), '17k': (4096, 17408, 5120, 40), '16k': (16384, 16384, 16384, 15)}
ARMS = {'wheel': [('b25632', 256, 128, 32, 0), ('b12864', 128, 128, 64, 0)],
        'hook':  [('b25632_hook2', 256, 128, 32, 0)],
        'fork':  [('b25632_ann2', 256, 128, 32, 2), ('b25632_ann3', 256, 128, 32, 3), ('b12864_ann1', 128, 128, 64, 1)]}[mode]
torch.manual_seed(1234)
def mk(n, m):
    r = torch.randint(0, 65536, (n, m), dtype=torch.int32, device='cuda')
    return ((r & 0x83ff) | (14 << 10)).to(torch.int16).view(torch.float16)
os.makedirs('s3src', exist_ok=True)
for sn, (M, N, K, it) in SH.items():
    A, B = mk(M, K), mk(N, K)
    C = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    fl = 2.0 * M * N * K
    ref = None
    if sn in ('4k',):
        ref = A[:1024].float() @ B.float().T
    if mode == 'wheel':
        for w in range(5): _ = A @ B.T
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(it): _ = A @ B.T
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3; del _
        print(json.dumps(dict(mode=mode, round=rnd, arm='hipblaslt_torch', shape=sn, ms=round(ms, 4), TF=round(fl / ms * 1e-9, 2))), flush=True)
    for name, bM, bN, bK, blocks in ARMS:
        t0 = time.perf_counter()
        k = gemm(M, N, K, bM, bN, bK, 256, 2, blocks)
        comp = time.perf_counter() - t0
        src = k.get_kernel_source()
        if rnd == 0:
            open(f's3src/{mode}_{name}_{sn}.cu', 'w').write(src)
        lb = re.search(r'__launch_bounds__\([^)]*\)', src).group(0)
        for w in range(5): k(A, B, C)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(it): k(A, B, C)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3
        rec = dict(mode=mode, round=rnd, arm=name, shape=sn, bounds=lb, compile_s=round(comp, 1), ms=round(ms, 4), TF=round(fl / ms * 1e-9, 2))
        if ref is not None:
            rec['max_rel_err'] = float((C[:1024].float() - ref).abs().max() / ref.abs().max())
        print(json.dumps(rec), flush=True)
    del A, B, C
    torch.cuda.empty_cache()
