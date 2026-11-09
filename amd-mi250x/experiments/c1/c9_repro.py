# Reproduce C9 failures with the installed TileLang (wheel or fork via PYTHONPATH). Compile only, CPU.
import sys, traceback, os
import torch, tilelang
import tilelang.rocm.language as T
print("tilelang", tilelang.__file__, tilelang.__version__)

@tilelang.jit
def u8copy(N, D, TH):
    # dynamically indexed uint8 global->shared copy: safe-memory legalisation adds a zero else-branch
    @T.prim_func
    def main(A: T.Tensor((N, D), T.uint8), Ix: T.Tensor((1,), T.int32), O: T.Tensor((N, D), T.uint8)):
        with T.Kernel(1, threads=TH) as bx:
            S = T.alloc_shared((16, D), T.uint8)
            n = Ix[0]
            for k in T.serial(1):
                T.copy(A[n * 16, 0], S)
            for i, j in T.Parallel(16, D):
                O[i, j] = S[i, j]
    return main

@tilelang.jit
def chain(M16, D, P, TH, policy, wrap=0):
    # q(fragment) -> GEMM with K(shared) -> scores(fragment) -> softmax-like -> GEMM with V(shared) -> out(fragment)
    @T.prim_func
    def main(Q: T.Tensor((M16, D), T.float16), K: T.Tensor((P, D), T.float16), V: T.Tensor((P, D), T.float16),
             O: T.Tensor((M16, D), T.float16)):
        with T.Kernel(1, threads=TH) as bx:
            K_s = T.alloc_shared((P, D), T.float16)
            V_s = T.alloc_shared((P, D), T.float16)
            q = T.alloc_fragment((M16, D), T.float16)
            s = T.alloc_fragment((M16, P), T.float32)
            p = T.alloc_fragment((M16, P), T.float16)
            o = T.alloc_fragment((M16, D), T.float32)
            T.copy(Q[0, 0], q)
            T.copy(K[0, 0], K_s)
            T.copy(V[0, 0], V_s)
            T.clear(s)
            T.gemm(q, K_s, s, transpose_B=True, policy=policy)
            for i, j in T.Parallel(M16, P):
                p[i, j] = s[i, j]
            T.clear(o)
            T.gemm(p, V_s, o, policy=policy)
            T.copy(o, O[0, 0])
    return main

@tilelang.jit
def single(M, N, K, TH, policy):
    # one shared-shared GEMM on a tile with few rows
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((N, K), T.float16), C: T.Tensor((M, N), T.float16)):
        with T.Kernel(1, threads=TH) as bx:
            A_s = T.alloc_shared((M, K), T.float16)
            B_s = T.alloc_shared((N, K), T.float16)
            C_l = T.alloc_fragment((M, N), T.float32)
            T.clear(C_l)
            T.copy(A[0, 0], A_s)
            T.copy(B[0, 0], B_s)
            T.gemm(A_s, B_s, C_l, transpose_B=True, policy=policy)
            T.copy(C_l, C[0, 0])
    return main

def attempt(name, fn):
    try:
        s = fn().get_kernel_source()
        print(f"RESULT {name}: OK", flush=True)
        return s
    except BaseException as e:
        msg = str(e)
        os.makedirs(os.environ.get('C9_OUT','c9_err'), exist_ok=True)
        open(os.path.join(os.environ.get('C9_OUT','c9_err'), name.replace(' ','_').replace('=','')+'.txt'),'w').write(msg)
        hits = [l.strip()[:300] for l in msg.splitlines() if any(x in l for x in ('error:', 'conflict', 'Divide', 'divide', 'must be', 'Check failed'))]
        print(f"RESULT {name}: FAIL {type(e).__name__}: " + (" | ".join(hits[:4]) or msg[:400]), flush=True)

which = sys.argv[1] if len(sys.argv) > 1 else 'all'
if which in ('all', 'u8'):
    attempt("u8copy D=64 256t", lambda: u8copy(256, 64, 256))
    attempt("u8copy D=128 128t", lambda: u8copy(256, 128, 128))
if which in ('all', 'chain'):
    G = T.GemmWarpPolicy
    for pname in ('FullRow', 'FullCol', 'Square'):
        pol = getattr(G, pname)
        for th in (64, 128, 256):
            attempt(f"chain 16x128x64 {pname} {th}t", lambda: chain(16, 128, 64, th, pol))
            attempt(f"chain 32x128x64 {pname} {th}t", lambda: chain(32, 128, 64, th, pol))
        for th in (128, 256):
            attempt(f"single 16x32x32 {pname} {th}t", lambda: single(16, 32, 32, th, pol))
            attempt(f"single 16x128x32 {pname} {th}t", lambda: single(16, 128, 32, th, pol))
