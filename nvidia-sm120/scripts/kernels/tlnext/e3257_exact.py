# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# #3257 feasibility for the one-weight-copy prefill (27-09-2026). Two bit-exactness questions, answered on the GPU with
# upstream's own T.gemm_blockscaled lowering (rebased build):
#  E1  swap-AB: is C = X @ W^T computed with A = X (activations, as K3 does) bit-identical to (W @ X^T)^T computed with
#      A = W (weights: the operand #3257 lets us feed from registers)?  Also: fragment A (#3257 "rs" path) vs shared A.
#  E2  k permutation inside a 16-element scale group, applied to both operands: bit-identical?  (lets each lane load
#      regfrag words without cross-lane shuffles; the activation quantizer can emit any in-group order for free)
#  E3  K3's own kernel (chunk-kmajor full-tile path) vs the rowmajor rs path with A = W, same data: bit-identical?
import json, sys, torch
import tilelang, tilelang.language as T
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4.kernels import nvfp4_gemm_kernel

torch.manual_seed(0)


@tilelang.jit
def gemm_bs(M, N, K, fragment_a: bool, policy_name: str = "Square", brw: int = 2, bcw: int = 2, wrt: int = 32, wct: int = 32):
    """C (M, N) fp32 = dq(A) @ dq(B)^T, rowmajor scales (M|N, K/64) uint32, one K block (chunk = K). Upstream test kernel."""
    fp4 = T.float4_e2m1fn
    block_M, block_N, block_K = brw * wrt, bcw * wct, K
    policy = getattr(T.GemmWarpPolicy, policy_name)

    @T.prim_func
    def main(A: T.Tensor((M, K), fp4), B: T.Tensor((N, K), fp4), SFA: T.Tensor((M, K // 64), T.uint32),
             SFB: T.Tensor((N, K // 64), T.uint32), C: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=32 * brw * bcw) as (bx, by):
            A_s = T.alloc_fragment((block_M, block_K), fp4) if fragment_a else T.alloc_shared((block_M, block_K), fp4)
            B_s = T.alloc_shared((block_N, block_K), fp4)
            SFA_s = T.alloc_shared((block_M, block_K // 64), T.uint32)
            SFB_s = T.alloc_shared((block_N, block_K // 64), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)
            for i, k in T.Parallel(block_M, block_K):
                A_s[i, k] = A[by * block_M + i, k]
            for j, k in T.Parallel(block_N, block_K):
                B_s[j, k] = B[bx * block_N + j, k]
            for i, k in T.Parallel(block_M, block_K // 64):
                SFA_s[i, k] = SFA[by * block_M + i, k]
            for j, k in T.Parallel(block_N, block_K // 64):
                SFB_s[j, k] = SFB[bx * block_N + j, k]
            T.gemm_blockscaled(A_s, B_s, C_l, SFA_s, SFB_s, transpose_B=True, policy=policy, clear_accum=True, k_start=0,
                               sf_a_granularity_k=16, sf_b_granularity_k=16, sf_layout="rowmajor")
            T.copy(C_l, C[by * block_M, bx * block_N])
    return main


def rand_fp4(rows, K):
    return torch.randint(-128, 128, (rows, K // 2), device="cuda", dtype=torch.int8)


def rand_scales(rows, K, lo=0x28, hi=0x48):
    b = torch.randint(lo, hi, (rows, K // 16), device="cuda", dtype=torch.int64)
    b = b.reshape(rows, K // 64, 4)
    return (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)).to(torch.uint32).contiguous(), b.reshape(rows, K // 16)


def unpack(p):
    u = p.view(torch.uint8)
    c = torch.empty(u.shape[0], u.shape[1] * 2, dtype=torch.uint8, device=u.device)
    c[:, 0::2] = u & 0xF; c[:, 1::2] = u >> 4
    return c


def pack(c):
    return (c[:, 0::2] | (c[:, 1::2] << 4)).view(torch.int8).contiguous()


def same(a, b):
    return bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


res = {}
M, N, K = 128, 256, 512
X, W = rand_fp4(M, K), rand_fp4(N, K)
SX, _ = rand_scales(M, K); SW, _ = rand_scales(N, K)
k_xw_ss = gemm_bs(M, N, K, False)
k_wx_ss = gemm_bs(N, M, K, False)
k_wx_rs = gemm_bs(N, M, K, True)
c1 = torch.empty(M, N, device="cuda"); c2 = torch.empty(N, M, device="cuda"); c3 = torch.empty(N, M, device="cuda")
k_xw_ss(X, W, SX, SW, c1); k_wx_ss(W, X, SW, SX, c2); k_wx_rs(W, X, SW, SX, c3)
torch.cuda.synchronize()
res["E1_swapAB_ss_bitexact"] = same(c1, c2.T)
res["E1_rs_vs_ss_bitexact"] = same(c2, c3)
res["E1_swapAB_rs_bitexact"] = same(c1, c3.T)
res["E1_nonzero"] = float(c1.abs().mean())
# FullRow policy (all warps along A rows: the layout a one-weight-copy kernel wants, each weight row in one warp)
k_wx_rs_fr = gemm_bs(N, M, K, True, "FullRow", 4, 1, 32, 128)
c4 = torch.empty(N, M, device="cuda"); k_wx_rs_fr(W, X, SW, SX, c4); torch.cuda.synchronize()
res["E1_rs_fullrow_bitexact"] = same(c1, c4.T)

# E2: a within-16-group permutation of k (nibble level), same for both operands
perm = torch.randperm(16, device="cuda")
idx = (torch.arange(K, device="cuda") // 16) * 16 + perm[torch.arange(K, device="cuda") % 16]
Xp, Wp = pack(unpack(X)[:, idx]), pack(unpack(W)[:, idx])
c5 = torch.empty(M, N, device="cuda"); k_xw_ss(Xp, Wp, SX, SW, c5); torch.cuda.synchronize()
res["E2_perm_in_group_bitexact"] = same(c1, c5)
# control: a permutation ACROSS groups must change results (the test can see differences)
idx2 = torch.randperm(K, device="cuda")
Xq, Wq = pack(unpack(X)[:, idx2]), pack(unpack(W)[:, idx2])
c6 = torch.empty(M, N, device="cuda"); k_xw_ss(Xq, Wq, SX, SW, c6); torch.cuda.synchronize()
res["E2_control_cross_group_differs"] = not same(c1, c6)
# more E2 trials (different permutations, wide scale range incl. tiny/huge exponents)
ok = True
for t in range(8):
    perm = torch.randperm(16, device="cuda")
    idx = (torch.arange(K, device="cuda") // 16) * 16 + perm[torch.arange(K, device="cuda") % 16]
    X2, W2 = rand_fp4(M, K), rand_fp4(N, K)
    S2x, _ = rand_scales(M, K, 0x01, 0x7E); S2w, _ = rand_scales(N, K, 0x01, 0x7E)
    a = torch.empty(M, N, device="cuda"); b = torch.empty(M, N, device="cuda")
    k_xw_ss(X2, W2, S2x, S2w, a); k_xw_ss(pack(unpack(X2)[:, idx]), pack(unpack(W2)[:, idx]), S2x, S2w, b)
    d = torch.empty(N, M, device="cuda"); k_wx_rs(W2, X2, S2w, S2x, d)
    torch.cuda.synchronize()
    ok &= same(a, b) and same(a, d.T)
res["E2_E1_8_trials_wide_scales"] = ok

# E3: K3's kernel (A = activations, chunk-kmajor full tile, 2 passes interleaved) vs rs path with A = W, same values.
# K3 pads M to 128; use its quantizer on random bf16 data and its weight preparation, then rebuild rowmajor words.
Mx, Nw, Kx = 128, 256, 512
x = torch.randn(Mx, Kx, device="cuda", dtype=torch.bfloat16)
wp = torch.randint(0, 256, (Nw, Kx // 2), dtype=torch.uint8, device="cuda")
ws = (torch.rand(Nw, Kx // 16, device="cuda") * 2 + 0.5).to(torch.float8_e4m3fn)
w0 = nv.prepare_weight(wp, ws, torch.tensor([1.0], device="cuda"))
A, SFA, rsinv = nv.quantize(x, 1)
ones = torch.ones(Mx, device="cuda")
c_k3 = nvfp4_gemm_kernel(A, w0.packed, SFA.reshape(1, Mx * (Kx // 256), 4), w0.sfb, torch.ones(Nw, device="cuda"), ones,
                         passes=1, out_dtype=T.float32)
# unswizzle SFA (chunk-kmajor, 128-row chunks) back to rowmajor words (rows, K/64)
def unswz(words_sw, rows, cols):
    src = words_sw.reshape(rows // 128, cols, 32, 4)            # inverse of permute(0,3,2,1) in swizzle_scale_words
    return src.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)
SXr = unswz(SFA.reshape(-1)[: Mx * Kx // 64], Mx, Kx // 64)
SWr = nv._pack_scale_bytes(ws.view(torch.uint8))
res["E3_unswizzle_roundtrip"] = bool(torch.equal(nv.swizzle_scale_words(SXr).reshape(-1), SFA.reshape(-1)[: Mx * Kx // 64]))
c_rs = torch.empty(Nw, Mx, device="cuda")
gemm_bs(Nw, Mx, Kx, True)(w0.packed, A, SWr, SXr, c_rs); torch.cuda.synchronize()
res["E3_k3_vs_rs_swapAB_bitexact"] = same(c_k3, c_rs.T)
res["E3_maxabs"] = float((c_k3 - c_rs.T).abs().max())
print(json.dumps(res))
