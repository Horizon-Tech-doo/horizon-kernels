# Fork JIT: repeat of the s5 winners on all 5 shapes, plus (256,4) and (256,5) on 128x128x64. Interleaved rounds with rotated arm order.
import json, sys, time, re
import torch, tilelang
from s3_kernel import gemm
SH = {'4k': (4096, 4096, 4096, 40), '8k': (8192, 8192, 8192, 40), '32k': (32768, 1024, 2048, 40), '17k': (4096, 17408, 5120, 40), '16k': (16384, 16384, 16384, 10)}
ARMS = [(128,128,64,256,0),(128,128,64,256,2),(128,128,64,256,3),(128,128,64,256,4),(128,128,64,256,5),(256,128,32,256,2),(256,128,32,256,0),(256,128,32,512,0),(256,128,32,512,3)]
ROUNDS = int(sys.argv[1])
torch.manual_seed(1234)
def mk(n, m):
    r = torch.randint(0, 65536, (n, m), dtype=torch.int32, device='cuda')
    return ((r & 0x83ff) | (14 << 10)).to(torch.int16).view(torch.float16)
for sn, (M, N, K, it) in SH.items():
    A, B = mk(M, K), mk(N, K)
    C = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    fl = 2.0 * M * N * K
    kern = {}
    for a in ARMS:
        bM, bN, bK, th, bl = a
        try:
            kern[a] = gemm(M, N, K, bM, bN, bK, th, 2, bl)
        except Exception as e:
            print(json.dumps(dict(arm=a, shape=sn, error=str(e)[:300])), flush=True)
    good = list(kern)
    for r in range(ROUNDS):
        rot = good[(r * 7) % len(good):] + good[:(r * 7) % len(good)]
        # hipBLASLt through torch, once per round, first
        for w in range(3): _ = A @ B.T
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(it): _ = A @ B.T
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3; del _
        print(json.dumps(dict(round=r, arm='hipblaslt_torch', shape=sn, ms=round(ms, 4), TF=round(fl / ms * 1e-9, 2))), flush=True)
        for a in rot:
            k = kern[a]
            for w in range(5): k(A, B, C)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(it): k(A, B, C)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3
            u = list(k.resource_usage.values())[0]
            print(json.dumps(dict(round=r, arm=a, shape=sn, bounds=re.search(r'__launch_bounds__\([^)]*\)', k.get_kernel_source()).group(0), V=u.n_regs, A=u.extra.get('AGPRs'), scr=u.scratch_bytes, occ=u.extra.get('Occupancy [waves/SIMD]'), ms=round(ms, 4), TF=round(fl / ms * 1e-9, 2))), flush=True)
    del A, B, C; torch.cuda.empty_cache()
