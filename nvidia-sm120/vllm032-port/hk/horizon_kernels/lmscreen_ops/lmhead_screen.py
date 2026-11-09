# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exact screening for a wide BF16 output layer (lm_head, e.g. 248,320 x 5,120) on SM120.

Greedy decoding, greedy speculative verification and top-k sampling read only the few largest logits of a row, but a
BF16 lm_head GEMM streams the whole BF16 weight (2.54 GB for Qwen3.8's head) per call. This module keeps the result
exact while streaming mostly FP8:

  1. screen      s~ = decode_w8a16(x, FP8-per-row copy of W)                (1 byte per weight; existing kernel)
  2. tile bound  TM[r, t] = max over tile t of the lower bound  s~ - B       (tile_lower_max_kernel)
                 B[r, c]  = xn[r] * P[c] + REL * |s~[r, c]| + ABS            (see the plugin's lmhead_screen.py for the
                 proof; P[c] = ||w_c - w^_c|| + gamma_K (||w_c|| + ||w^_c||), xn[r] = ||x_r|| (1 + 1e-3))
     threshold   thr[r] = the k-th largest TM[r, :] (host glue: torch amax / topk over (M, V / TILE) values)
  3. compact     every token c with  up(s~ + B) >= dn(thr)  is appended to cand[r, :]   (compact_kernel)
                 up(v) = v + 2^-7 |v| + ABS, dn(v) = v - 2^-7 |v| - ABS (one BF16 ulp is <= 2^-7 |v|)
  4. recheck     out[r, c] = bf16(x_r . w_c) for every selected c: BF16 weights, fp32 tensor-core accumulation in
                 k order, one BF16 rounding (recheck_kernel); rows whose selection overflowed KMAX are recomputed
                 in full by full_rows_kernel with the same arithmetic (device-side decision, no host sync)
  5. out is -inf everywhere else. Argmax / top-k masks / samplers applied to `out` see exactly what they see on
     the full BF16 logits, provided the recheck arithmetic equals the reference GEMM's (see HZ_H1 in the design doc).

Every kernel is M-dynamic (one compile per shape), deterministic (integer atomics only reserve output slots: the SET
of candidates and every value are order-independent) and CUDA-graph capturable (no host sync, fixed buffer shapes).
"""

import tilelang
import tilelang.language as T

REL = 2.0 ** -8 * (1.0 + 2.0 ** -6)     # screen BF16 output rounding (relative to the rounded value)
ULP_REL = 2.0 ** -7                     # one BF16 ulp relative to |v|
ABS = 1e-30
SUPPORTS_M = 1024
_PASS = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
         tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True}

# Block-wide exclusive scan of one int per thread (blockDim.x a multiple of 32, <= 1024). ws: shared int[33].
# Returns the exclusive prefix of v; ws[32] holds the block total afterwards. Two __syncthreads inside; the caller
# must __syncthreads before reusing ws.
_SCAN_SRC = r"""
__device__ __forceinline__ int hz_block_excl_scan(int v, int *ws) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
  if (lane == 31) ws[warp] = x;
  __syncthreads();
  if (warp == 0) {
    const int w = lane < nw ? ws[lane] : 0;
    int t = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { const int y = __shfl_up_sync(0xffffffffu, t, o); if (lane >= o) t += y; }
    if (lane < nw) ws[lane] = t - w;
    if (lane == 31) ws[32] = t;
  }
  __syncthreads();
  return ws[warp] + x - v;
}
"""


def _lower(s, xn, p):
    """Lower bound on the pre-rounding fp32 logit from the screen value s (fp32), ||x|| and P."""
    return s - (xn * p + REL * T.abs(s) + ABS)


@tilelang.jit(pass_configs=_PASS)
def tile_lower_max_kernel(V: int, TILE: int = 128, TPC: int = 8, threads: int = 128):
    """TM[r, t] = max_{c in tile t} (s~[r, c] - B[r, c]); tiles of TILE tokens, TPC tiles per CTA. -inf past V."""
    assert (TPC * TILE) % threads == 0
    nt = (V + TILE - 1) // TILE
    M = T.dynamic("M")
    bf16, f32 = T.bfloat16, T.float32

    @T.prim_func
    def main(S: T.Tensor((M, V), bf16), xn: T.Tensor((M,), f32), P: T.Tensor((V,), f32),
             TM: T.Tensor((M, nt), f32)):
        with T.Kernel(T.ceildiv(nt, TPC), M, threads=threads) as (bt, r):
            T.assume(M >= 1)
            L = T.alloc_fragment((TPC, TILE), f32)
            R = T.alloc_fragment((TPC,), f32)
            x_n = xn[r]
            for i, j in T.Parallel(TPC, TILE):
                c = (bt * TPC + i) * TILE + j
                cc = T.min(c, V - 1)
                s = T.cast(S[r, cc], f32)
                L[i, j] = T.if_then_else(c < V, _lower(s, x_n, P[cc]), -T.infinity(f32))
            T.reduce_max(L, R, dim=1)
            for i in T.Parallel(TPC):
                if bt * TPC + i < nt:
                    TM[r, bt * TPC + i] = R[i]

    return main


@tilelang.jit(pass_configs=_PASS)
def compact_kernel(V: int, KMAX: int, CT: int = 2048, threads: int = 256):
    """Append every token c of the row with up(s~ + B) >= dn(thr[r]) to cand[r, :KMAX]; cnt[r] counts ALL selected
    tokens (cnt > KMAX = overflow: the full-row kernel recomputes that row). cnt must be zero on entry."""
    assert CT % threads == 0
    per = CT // threads
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    @T.prim_func
    def main(S: T.Tensor((M, V), bf16), xn: T.Tensor((M,), f32), P: T.Tensor((V,), f32), thr: T.Tensor((M,), f32),
             cand: T.Tensor((M, KMAX), i32), cnt: T.Tensor((M,), i32)):
        with T.Kernel(T.ceildiv(V, CT), M, threads=threads, prelude=_SCAN_SRC) as (bt, r):
            T.assume(M >= 1)
            ws = T.alloc_shared((33,), i32)
            base_s = T.alloc_shared((1,), i32)
            keep = T.alloc_local((per,), i32)
            n_loc = T.alloc_var(i32)
            pos = T.alloc_var(i32)
            tid = T.get_thread_binding()
            x_n = xn[r]
            t = thr[r]
            t_dn = t - ULP_REL * T.abs(t) - ABS
            n_loc = 0
            for j in T.unroll(per):
                c = bt * CT + j * threads + tid
                cc = T.min(c, V - 1)
                s = T.cast(S[r, cc], f32)
                u = s + (x_n * P[cc] + REL * T.abs(s) + ABS)
                u_up = u + ULP_REL * T.abs(u) + ABS
                keep[j] = T.if_then_else(T.And(c < V, u_up >= t_dn), 1, 0)
                n_loc = n_loc + keep[j]
            pos = T.call_extern(i32, "hz_block_excl_scan", n_loc, T.access_ptr(ws, "rw"))
            if tid == 0:
                if ws[32] > 0:
                    base_s[0] = T.atomic_add(cnt[r], ws[32], return_prev=True)
            T.sync_threads()
            pos = pos + base_s[0]
            for j in T.unroll(per):
                if keep[j] == 1:
                    if pos < KMAX:
                        cand[r, pos] = bt * CT + j * threads + tid
                    pos = pos + 1

    return main


@tilelang.jit(pass_configs=_PASS)
def recheck_kernel(V: int, K: int, KMAX: int, BN: int = 64, BK: int = 128, num_stages: int = 3, threads: int = 128):
    """out[r, cand[r, i]] = bf16(x_r . W[cand[r, i]]) for i < cnt[r] <= KMAX. Grid (KMAX / BN, M): chunk bj of row r
    is idle when bj * BN >= cnt[r] or the row overflowed (cnt[r] > KMAX: full_rows_kernel owns it).
    Arithmetic: the A tile is x_r in row 0 and zeros in rows 1..15; T.gemm (mma m16n8k16 bf16, fp32 accumulate) over
    k tiles in order, one BF16 rounding: the per-element sequence of a non-split-K tensor-core GEMM."""
    assert KMAX % BN == 0 and K % BK == 0
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    @T.prim_func
    def main(x: T.Tensor((M, K), bf16), W: T.Tensor((V, K), bf16), cand: T.Tensor((M, KMAX), i32),
             cnt: T.Tensor((M,), i32), out: T.Tensor((M, V), bf16)):
        with T.Kernel(KMAX // BN, M, threads=threads) as (bj, r):
            T.assume(M >= 1)
            idx_s = T.alloc_shared((BN,), i32)
            A_s = T.alloc_shared((16, BK), bf16)
            B_s = T.alloc_shared((BN, BK), bf16)
            C_s = T.alloc_shared((16, BN), f32)
            C = T.alloc_fragment((16, BN), f32)
            n = cnt[r]
            if T.And(n <= KMAX, bj * BN < n):
                for i in T.Parallel(BN):
                    idx_s[i] = T.if_then_else(bj * BN + i < n, cand[r, bj * BN + i], 0)
                T.clear(C)
                for k in T.Pipelined(K // BK, num_stages=num_stages):
                    for i, kk in T.Parallel(16, BK):
                        A_s[i, kk] = T.if_then_else(i == 0, x[r, k * BK + kk], T.cast(0, bf16))
                    for i, kk in T.Parallel(BN, BK):
                        B_s[i, kk] = W[idx_s[i], k * BK + kk]
                    T.gemm(A_s, B_s, C, transpose_B=True)
                T.copy(C, C_s)
                for j in T.Parallel(BN):
                    if bj * BN + j < n:
                        out[r, idx_s[j]] = T.cast(C_s[0, j], bf16)

    return main


@tilelang.jit(pass_configs=_PASS)
def full_rows_kernel(V: int, K: int, BN: int = 128, BK: int = 64, MB: int = 16, num_stages: int = 3,
                     threads: int = 128):
    """out[r, :] = bf16(x_r @ W^T) for every row with cnt[r] > limit[0] (limit = KMAX: the overflowed rows;
    limit = -1: every row, the "own head" mode). Grid V / BN; each CTA streams its BN x K weight tile once per group
    of MB selected rows and does nothing when no row is selected. Same per-element arithmetic as recheck_kernel."""
    assert V % BN == 0 and K % BK == 0
    M = T.dynamic("M")
    bf16, f32, i32 = T.bfloat16, T.float32, T.int32

    @T.prim_func
    def main(x: T.Tensor((M, K), bf16), W: T.Tensor((V, K), bf16), cnt: T.Tensor((M,), i32),
             limit: T.Tensor((1,), i32), out: T.Tensor((M, V), bf16)):
        with T.Kernel(V // BN, threads=threads) as bx:
            T.assume(M >= 1)
            rows_s = T.alloc_shared((SUPPORTS_M,), i32)
            nsel_s = T.alloc_shared((1,), i32)
            A_s = T.alloc_shared((MB, BK), bf16)
            B_s = T.alloc_shared((BN, BK), bf16)
            C_s = T.alloc_shared((MB, BN), f32)
            C = T.alloc_fragment((MB, BN), f32)
            nsel = T.alloc_var(i32)
            tid = T.get_thread_binding()
            if tid == 0:
                nsel = 0
                lim = limit[0]
                for r in T.serial(M):
                    if cnt[r] > lim:
                        rows_s[nsel] = r
                        nsel = nsel + 1
                nsel_s[0] = nsel
            T.sync_threads()
            ns = T.alloc_var(i32)
            ns = nsel_s[0]
            for g in T.serial(T.ceildiv(ns, MB)):
                T.clear(C)
                for k in T.Pipelined(K // BK, num_stages=num_stages):
                    for i, kk in T.Parallel(MB, BK):
                        A_s[i, kk] = x[rows_s[T.min(g * MB + i, ns - 1)], k * BK + kk]
                    for i, kk in T.Parallel(BN, BK):
                        B_s[i, kk] = W[bx * BN + i, k * BK + kk]
                    T.gemm(A_s, B_s, C, transpose_B=True)
                T.copy(C, C_s)
                for i, j in T.Parallel(MB, BN):
                    if g * MB + i < ns:
                        out[rows_s[g * MB + i], bx * BN + j] = T.cast(C_s[i, j], bf16)
                T.sync_threads()

    return main


# ---------------------------------------------------------------------------------------------------------------
# Host side.
class Screen:
    """Exact screening head. w: (V, K) bf16 (the served lm_head weight, not copied); q: (V, K) float8_e4m3fn and
    scale (V,) fp32 (w^ = scale * q); P: (V,) fp32 per-token bound terms (plugin: lmhead_screen.token_terms).
    kmax: per-row candidate capacity for greedy rows; topk_kmax: for top-k rows (k > 1 selects far more tokens).
    tile / topk_tile: token tile of the threshold kernel (greedy: any tile gives the exact max; top-k: smaller tiles
    give a tighter k-th value)."""

    def __init__(self, w, q, scale, P, kmax=4096, topk_kmax=16384, tile=128, topk_tile=16):
        import torch
        from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
        self.V, self.K = w.shape
        assert q.shape == w.shape and P.shape == (self.V,) and w.dtype == torch.bfloat16
        assert self.V % 128 == 0 and self.K % 128 == 0
        self.w, self.P = w, P.float().contiguous()
        self.prep = w8.prepare(q, scale)
        self.kmax, self.topk_kmax = kmax, topk_kmax
        self.tiles = {1: tile, 2: topk_tile}
        self._tlm = {}
        self._sel = {}
        self.k_full = full_rows_kernel(self.V, self.K)
        dev = w.device
        self.lim_all = torch.tensor([-1], dtype=torch.int32, device=dev)
        self.last_cnt, self.last_kmax = None, None
        for km in {kmax, topk_kmax}:
            self._kernels(km)
        for t in set(self.tiles.values()):
            self._tile_kernel(t)

    def _tile_kernel(self, tile):
        k = self._tlm.get(tile)
        if k is None:
            k = self._tlm[tile] = tile_lower_max_kernel(self.V, tile, max(1, 1024 // tile))
        return k

    def _kernels(self, km):
        k = self._sel.get(km)
        if k is None:
            import torch
            lim = torch.tensor([km], dtype=torch.int32, device=self.w.device)
            k = self._sel[km] = (compact_kernel(self.V, km), recheck_kernel(self.V, self.K, km), lim)
        return k

    def screen(self, x):
        from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
        return w8.linear(x, self.prep)

    def threshold(self, s, xn, k=None, k_host_max=None):
        """k None: greedy, the max of the per-tile maxima = the exact max of the lower bounds (capturable).
        k: (M,) int64 per-row k with its host-known maximum k_host_max: the k-th largest per-tile maximum, a lower
        bound of the k-th largest lower bound, so the selection is a superset of the exact one (eager path)."""
        import torch
        M = s.shape[0]
        tile = self.tiles[1 if k is None else 2]
        nt = (self.V + tile - 1) // tile
        TM = torch.empty(M, nt, dtype=torch.float32, device=s.device)
        self._tile_kernel(tile)(s, xn, self.P, TM)
        if k is None:
            return TM.amax(-1)
        vals = TM.topk(min(int(k_host_max), nt), dim=-1).values
        return vals.gather(1, (k.clamp(min=1, max=vals.shape[1]) - 1).view(-1, 1)).view(-1)

    def logits(self, x, k=None, k_host_max=None, s=None):
        """Sparse exact logits (M, V) bf16 for x (M, K) bf16: exact values at every token that can be in the row's
        top-k set (k per row; greedy when None), -inf elsewhere; overflowed rows complete."""
        import torch
        x = x.contiguous()
        M = x.shape[0]
        assert M <= SUPPORTS_M, M
        km = self.kmax if k is None else self.topk_kmax
        cmp_k, rc_k, lim = self._kernels(km)
        xn = x.float().norm(dim=-1) * (1.0 + 1e-3)
        if s is None:
            s = self.screen(x)
        thr = self.threshold(s, xn, k, k_host_max)
        cand = torch.empty(M, km, dtype=torch.int32, device=x.device)
        cnt = torch.zeros(M, dtype=torch.int32, device=x.device)
        cmp_k(s, xn, self.P, thr, cand, cnt)
        out = torch.full((M, self.V), float("-inf"), dtype=torch.bfloat16, device=x.device)
        rc_k(x, self.w, cand, cnt, out)
        self.k_full(x, self.w, cnt, lim, out)
        self.last_cnt, self.last_kmax = cnt, km
        return out

    def full(self, x):
        """Every row in full with the recheck arithmetic (the "own head" mode)."""
        import torch
        x = x.contiguous()
        M = x.shape[0]
        assert M <= SUPPORTS_M, M
        out = torch.empty(M, self.V, dtype=torch.bfloat16, device=x.device)
        cnt = torch.zeros(M, dtype=torch.int32, device=x.device)
        self.k_full(x, self.w, cnt, self.lim_all, out)
        return out
