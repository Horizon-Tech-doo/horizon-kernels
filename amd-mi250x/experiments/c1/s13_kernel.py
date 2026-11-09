# JIT kernels for Experiments B and C: E1 gemm_prefetch.py's `base` and `regstage`, with a tag per arm so each arm has its own cache key.
import re, os
import tvm_ffi
import tilelang
import tilelang.rocm.language as T

STATE = {'wpe': 0, 'rr': False}

def rr(s):
    # E1 gen_e1.py rr(): the register arrays become scalars, so the K loop loads straight into registers
    m = re.search(r'half_t a_r\[(\d+)\];\s*\n\s*half_t b_r\[(\d+)\];', s)
    na, nb = int(m.group(1)) // 8, int(m.group(2)) // 8
    decl = " ".join("uint4 a_r_%d;" % i for i in range(na)) + " " + " ".join("uint4 b_r_%d;" % i for i in range(nb))
    s = s.replace(m.group(0), decl)
    pat = re.compile(r'for \(int (v(?:_\d+)?) = 0; \1 < (\d+); \+\+\1\) \{\n(.*?)\n\s*\}', re.S)
    def exp(mm):
        v, n, b = mm.group(1), int(mm.group(2)), mm.group(3)
        out = []
        for i in range(n):
            t = re.sub(r'\*\(uint4\*\)\(([ab])_r \+ \(%s \* 8\)\)' % v, lambda q: "%s_r_%d" % (q.group(1), i), b)
            t = re.sub(r'\b%s\b' % v, "(%d)" % i, t)
            out.append(t)
        return "\n".join(out)
    s = pat.sub(exp, s)
    assert "a_r +" not in s and "b_r +" not in s
    return s

def postproc(code, target):
    if STATE['rr']:
        code = rr(code)
    if STATE['wpe']:
        code = re.sub(r'__launch_bounds__\((\d+)\)', lambda m: f"__launch_bounds__({m.group(1)}, {STATE['wpe']})", code)
    return code

tvm_ffi.register_global_func("tilelang_callback_hip_postproc", postproc, override=True)

@tilelang.jit
def gemm_t(M, N, K, bM, bN, bK, threads, kp, variant):
    dtype, acc = T.float16, T.float32
    nk = K // bK
    VA = max(1, bM * bK // 8 // threads)
    VB = max(1, bN * bK // 8 // threads)
    @T.prim_func
    def base(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            T.clear(C_l)
            for k in T.serial(nk):
                T.copy(A[by * bM, k * bK], A_s)
                T.copy(B[bx * bN, k * bK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])
    @T.prim_func
    def regstage(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=threads) as (bx, by):
            A_s = T.alloc_shared((bM, bK), dtype)
            B_s = T.alloc_shared((bN, bK), dtype)
            a_r = T.alloc_local((VA * 8,), dtype)
            b_r = T.alloc_local((VB * 8,), dtype)
            C_l = T.alloc_fragment((bM, bN), acc)
            tx = T.get_thread_binding()
            T.clear(C_l)
            for v in T.serial(VA):
                for j in T.vectorized(8):
                    a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for v in T.serial(VB):
                for j in T.vectorized(8):
                    b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j]
            for k in T.serial(nk):
                for v in T.serial(VA):
                    for j in T.vectorized(8):
                        A_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = a_r[v * 8 + j]
                for v in T.serial(VB):
                    for j in T.vectorized(8):
                        B_s[((v * threads + tx) * 8) // bK, ((v * threads + tx) * 8) % bK + j] = b_r[v * 8 + j]
                if k + 1 < nk:
                    for v in T.serial(VA):
                        for j in T.vectorized(8):
                            a_r[v * 8 + j] = A[by * bM + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                    for v in T.serial(VB):
                        for j in T.vectorized(8):
                            b_r[v * 8 + j] = B[bx * bN + ((v * threads + tx) * 8) // bK, (k + 1) * bK + ((v * threads + tx) * 8) % bK + j]
                T.gemm(A_s, B_s, C_l, transpose_B=True, k_pack=kp)
            T.copy(C_l, C[by * bM, bx * bN])
    return {"base": base, "regstage": regstage}[variant]

ARMS = {  # variant, bM, bK, waves per EU, rr transform, memory clause limit
    'b12864':       ('base', 128, 64, 0, False, None),
    'b12864cl1':    ('base', 128, 64, 0, False, 1),
    'b12864cl2':    ('base', 128, 64, 0, False, 2),
    'b25632a2':     ('base', 256, 32, 2, False, None),
    'rr25632w2':    ('regstage', 256, 32, 2, True, None),
    'rr25632w2cl1': ('regstage', 256, 32, 2, True, 1),
    'rr12864w3':    ('regstage', 128, 64, 3, True, None),
}

def build(name, M, N, K):
    variant, bM, bK, wpe, isrr, cl = ARMS[name]
    STATE.update(wpe=wpe, rr=isrr)
    old = os.environ.pop('HIPCC_COMPILE_FLAGS_APPEND', None)
    if cl is not None:
        os.environ['HIPCC_COMPILE_FLAGS_APPEND'] = f'-mllvm -amdgpu-max-memory-clause={cl}'
    try:
        # The cache key does not see this flag or the hook. compile_flags is in the key and is unused on the HIP path, so it separates the arms.
        return tilelang.compile(gemm_t.get_tir(M, N, K, bM, 128, bK, 256, 2, variant), compile_flags=[f'-DC1ARM_{name}'])
    finally:
        os.environ.pop('HIPCC_COMPILE_FLAGS_APPEND', None)
        if old is not None:
            os.environ['HIPCC_COMPILE_FLAGS_APPEND'] = old
        STATE.update(wpe=0, rr=False)
