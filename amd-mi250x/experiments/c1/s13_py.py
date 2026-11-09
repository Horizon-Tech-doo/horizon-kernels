# Experiments B and C through the real JIT python call: same process, same data, hipBLASLt via torch. fp32 check on a poisoned C at every shape, every arm.
import json, re, sys, time
import torch
from s13_kernel import build
rounds = int(sys.argv[1]); seed = int(sys.argv[2])
SH = {'4k': (4096, 4096, 4096, 40), '17k': (4096, 17408, 5120, 20), '32k': (32768, 1024, 2048, 40), '8k': (8192, 8192, 8192, 20), '16k': (16384, 16384, 16384, 5)}
PLAN = {'4k': ['b12864', 'b12864cl1', 'b12864cl2', 'rr25632w2', 'rr25632w2cl1', 'rr12864w3', 'b25632a2'],
        '17k': ['b12864', 'b12864cl1', 'rr25632w2', 'rr25632w2cl1', 'rr12864w3', 'b25632a2'],
        '32k': ['b12864', 'b12864cl1', 'b12864cl2', 'rr12864w3', 'rr25632w2'],
        '8k': ['b12864', 'b12864cl1', 'rr25632w2', 'b25632a2'],
        '16k': ['b12864cl1', 'rr25632w2', 'b25632a2']}
if len(sys.argv) > 3:
    PLAN = json.loads(sys.argv[3])
torch.manual_seed(1234)
def mk(n, m):
    r = torch.randint(0, 65536, (n, m), dtype=torch.int32, device='cuda')
    return ((r & 0x83ff) | (14 << 10)).to(torch.int16).view(torch.float16)
for sn, names in PLAN.items():
    M, N, K, it = SH[sn]
    A, B = mk(M, K), mk(N, K); C = torch.zeros(M, N, dtype=torch.float16, device='cuda')
    fl = 2.0 * M * N * K
    ref = A.float() @ B.float().T; refmax = ref.abs().max()
    print(json.dumps(dict(shape=sn, A=hex(A.data_ptr()), B=hex(B.data_ptr()), C=hex(C.data_ptr()))), flush=True)
    K_ = {}
    for n in names:
        t0 = time.perf_counter(); k = build(n, M, N, K); comp = time.perf_counter() - t0
        src = k.get_kernel_source()
        K_[n] = k
        print(json.dumps(dict(shape=sn, arm=n, compile_s=round(comp, 1), bounds=re.search(r'__launch_bounds__\([^)]*\)', src).group(0), rr='a_r_0' in src, resource={x: k.resource_usage.get(x) for x in k.resource_usage} if isinstance(k.resource_usage, dict) else str(k.resource_usage)), default=str), flush=True)
    arms = ['hipblaslt_torch'] + names
    g = torch.Generator().manual_seed(seed)
    for r in range(rounds):
        order = [arms[i] for i in torch.randperm(len(arms), generator=g).tolist()]
        for pos, a in enumerate(order):
            if a == 'hipblaslt_torch':
                for w in range(5): _ = A @ B.T
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for i in range(it): _ = A @ B.T
                torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3
                res = _; del _
            else:
                k = K_[a]
                for w in range(5): k(A, B, C)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for i in range(it): k(A, B, C)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / it * 1e3
                C.fill_(float('nan')); k(A, B, C); torch.cuda.synchronize(); res = C
            rec = dict(round=r, pos=pos, shape=sn, arm=a, ms=round(ms, 4), TF=round(fl / ms * 1e-9, 2))
            if r == 0 or r == rounds - 1:
                rec['fp32_rel'] = float((res.float() - ref).abs().max() / refmax)
                rec['nan'] = int(res.isnan().sum())
            print(json.dumps(rec), flush=True)
            del res
    del A, B, C, ref, K_
    torch.cuda.empty_cache()
print("DONE", flush=True)
