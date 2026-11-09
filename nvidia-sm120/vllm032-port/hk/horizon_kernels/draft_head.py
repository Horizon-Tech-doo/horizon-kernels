# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K1a: fused draft head for MTP drafts = argmax over vocab of (h @ W^T), W stored FP8 e4m3 with per-row scales.
# Never materialises the (M, 248320) logits: each CTA reduces its vocab slice to (max, argmax) per row, then a
# tiny torch reduction picks the winner across CTAs. Drafts only propose tokens (main model verifies with the
# full BF16 lm_head), so FP8 here can only change acceptance, never output.
import torch, tilelang, tilelang.language as T

FP8 = "float8_e4m3"


@tilelang.jit(out_idx=[-2, -1])
def draft_head_fp8(N, K, BM=16, BN=128, BK=128, stages=4, threads=128):
    nblk = (N + BN - 1) // BN

    @T.prim_func
    def main(A: T.Tensor((BM, K), FP8),          # draft hidden, quantised per row, padded to BM rows
             W: T.Tensor((N, K), FP8),           # draft lm_head, FP8 per-row
             SA: T.Tensor((BM,), "float32"),     # per-row activation scale
             SW: T.Tensor((N,), "float32"),      # per-vocab-row weight scale
             PV: T.Tensor((nblk, BM), "float32"),  # partial max value per CTA
             PI: T.Tensor((nblk, BM), "int32")):   # partial argmax per CTA
        with T.Kernel(nblk, threads=threads) as bx:
            As = T.alloc_shared((BM, BK), FP8)
            Ws = T.alloc_shared((BN, BK), FP8)
            C = T.alloc_fragment((BM, BN), "float32")
            Cs = T.alloc_shared((BM, BN), "float32")
            T.clear(C)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[0, k * BK], As)
                T.copy(W[bx * BN, k * BK], Ws)
                T.gemm(As, Ws, C, transpose_B=True)
            T.copy(C, Cs)
            for m in T.Parallel(BM):
                best = T.alloc_var("float32")
                bi = T.alloc_var("int32")
                best = -T.infinity("float32")
                bi = 0
                for j in T.serial(BN):
                    n = bx * BN + j
                    v = T.if_then_else(n < N, Cs[m, j] * SW[T.min(n, N - 1)] * SA[m], -T.infinity("float32"))
                    if v > best:
                        best = v
                        bi = n
                PV[bx, m] = best
                PI[bx, m] = bi

    return main


# K1a v2 (W8A16): the hidden state stays BF16. LLM hidden states have a few outlier channels, and per-row FP8
# scaling of h squashes every other channel into a handful of FP8 steps, which cost draft acceptance in v1.
# Only the weights (99.9% of the bytes) are FP8; each tile is widened to BF16 in shared memory before the MMA.
@tilelang.jit(out_idx=[-2, -1])
def draft_head_w8a16(N, K, BM=16, BN=128, BK=64, stages=4, threads=128):
    nblk = (N + BN - 1) // BN

    @T.prim_func
    def main(A: T.Tensor((BM, K), "bfloat16"),     # draft hidden, padded to BM rows
             W: T.Tensor((N, K), FP8),             # draft lm_head, FP8 per-row
             SW: T.Tensor((N,), "float32"),        # per-vocab-row weight scale
             PV: T.Tensor((nblk, BM), "float32"),
             PI: T.Tensor((nblk, BM), "int32")):
        with T.Kernel(nblk, threads=threads) as bx:
            As = T.alloc_shared((BM, BK), "bfloat16")
            Ws = T.alloc_shared((BN, BK), FP8)
            Wb = T.alloc_shared((BN, BK), "bfloat16")
            C = T.alloc_fragment((BM, BN), "float32")
            Cs = T.alloc_shared((BM, BN), "float32")
            T.clear(C)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=stages):
                T.copy(A[0, k * BK], As)
                T.copy(W[bx * BN, k * BK], Ws)
                for i, j in T.Parallel(BN, BK):
                    Wb[i, j] = T.cast(Ws[i, j], "bfloat16")
                T.gemm(As, Wb, C, transpose_B=True)
            T.copy(C, Cs)
            for m in T.Parallel(BM):
                best = T.alloc_var("float32")
                bi = T.alloc_var("int32")
                best = -T.infinity("float32")
                bi = 0
                for j in T.serial(BN):
                    n = bx * BN + j
                    v = T.if_then_else(n < N, Cs[m, j] * SW[T.min(n, N - 1)], -T.infinity("float32"))
                    if v > best:
                        best = v
                        bi = n
                PV[bx, m] = best
                PI[bx, m] = bi

    return main


def quant_rows_fp8(x):
    s = x.float().abs().amax(-1).clamp(min=1e-8) / 448.0
    return (x.float() / s[:, None]).to(torch.float8_e4m3fn), s.float()


class DraftHeadFP8:
    """Fused draft-head argmax. act="fp8": K1a v1 (W8A8, deployed default). act="bf16": K1a v2 (W8A16).

    vocab_ids (optional, 1-D int64): the draft vocabulary. The head then only scores those rows of weight_bf16
    (a (V, K) matrix: either lm_head[vocab_ids] or a separately trained draft head) and maps the winning row back
    to the full token id. With V ~ 32k instead of 248k, the head reads ~8x fewer bytes per draft step. Drafts
    outside the vocabulary can never be proposed; the main model still verifies every token, so output is unchanged.
    """

    def __init__(self, weight_bf16, BM=16, act="fp8", vocab_ids=None, **kw):
        if act not in ("fp8", "bf16"):
            raise ValueError(f"act must be 'fp8' or 'bf16', got {act!r}")
        self.vocab_ids = None
        if vocab_ids is not None:
            self.vocab_ids = vocab_ids.to(device=weight_bf16.device, dtype=torch.long).contiguous()
            if weight_bf16.shape[0] != self.vocab_ids.numel():      # given the full lm_head: take the vocab rows
                weight_bf16 = weight_bf16[self.vocab_ids]
        self.N, self.K = weight_bf16.shape
        self.BM, self.act = BM, act
        dev = weight_bf16.device
        self.W, self.SW = quant_rows_fp8(weight_bf16)
        if act == "bf16":
            self.kernel = draft_head_w8a16(self.N, self.K, BM=BM, **kw)
            self.Apad = torch.zeros(BM, self.K, dtype=torch.bfloat16, device=dev)
        else:
            self.kernel = draft_head_fp8(self.N, self.K, BM=BM, **kw)
            self.Apad = torch.zeros(BM, self.K, dtype=torch.float8_e4m3fn, device=dev)
            self.SApad = torch.ones(BM, dtype=torch.float32, device=dev)

    def __call__(self, h):  # h: (M<=BM, K) bf16 -> (M,) int64 token ids
        M = h.shape[0]
        if self.act == "bf16":
            self.Apad[:M].copy_(h)
            pv, pi = self.kernel(self.Apad, self.W, self.SW)
        else:
            a, sa = quant_rows_fp8(h)
            self.Apad[:M].copy_(a); self.SApad[:M].copy_(sa)
            pv, pi = self.kernel(self.Apad, self.W, self.SApad, self.SW)
        best = pv[:, :M].argmax(0)
        local = pi[:, :M].gather(0, best[None]).squeeze(0).long()
        return local if self.vocab_ids is None else self.vocab_ids[local]
