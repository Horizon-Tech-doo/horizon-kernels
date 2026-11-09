# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# chunk_report2.py [PREFIX] (fp8fix, 27-09-2026): chunk-check arms, split at each chunked arm's first prefill boundary.
# Boundary from the arm's own journal lines (in $HK_WORK/fp8fix/chunk-arm-<A>.log): align mode (prefix caching on)
# -> floor(max_num_batched_tokens / attention block size) * block size; otherwise max_num_batched_tokens.
# Same-tokenizer pairs: dNLL per token before / from the boundary. Cross-model pairs: nats per character over the
# common character range, split at the character where the first arm's boundary token starts.
import json, math, os, re, sys
D = os.path.expanduser("~/logprob_eval"); F = os.path.expanduser("$HK_WORK/fp8fix")
P = sys.argv[1] if len(sys.argv) > 1 else "cc2"


def load(a):
    f = f"{D}/{P}-{a}.json"
    return json.load(open(f)) if os.path.exists(f) else None


def boundary(a):
    try:
        j = open(f"{F}/chunk-arm-{a}.log", errors="ignore").read()
    except OSError:
        return None
    mn = [int(x) for x in re.findall(r"max_num_batched_tokens=(\d+)", j)]
    bs = [int(x) for x in re.findall(r"attention block size to (\d+)", j)]
    if not mn:
        return None
    b = mn[-1]
    if "'align'" in j and bs:
        b = (b // bs[-1]) * bs[-1]
    return b


def tok_pair(a, b, what):
    A, B = load(a), load(b)
    if not A or not B:
        return
    bd = boundary(a) or 8000
    print(f"== {a} -> {b}: {what}  (boundary of {a}: token {bd})")
    for doc in sorted(k for k in set(A) & set(B) if not k.startswith("_")):
        x, y = A[doc]["lp"], B[doc]["lp"]
        n = min(len(x), len(y)); c = min(bd - 1, n)
        d1 = (sum(x[:c]) - sum(y[:c])) / c
        d2 = (sum(x[c:n]) - sum(y[c:n])) / (n - c) if n > c else float("nan")
        m2 = sum(abs(p - q) for p, q in zip(x[c:n], y[c:n])) / (n - c) if n > c else float("nan")
        print(f"   {doc:12s} n={n:6d} before: dNLL {d1:+.4f} | from boundary: dNLL {d2:+.4f} "
              f"(PPL x{math.exp(d2) if d2 == d2 else float('nan'):.3f}) mean|d| {m2:.4f}")


def char_pair(a, b, what):
    A, B = load(a), load(b)
    if not A or not B:
        return
    bd = boundary(a) or 8000
    print(f"== {a} -> {b}: {what}  (nats per character; split at the start of {a}'s token {bd})")
    for doc in sorted(k for k in set(A) & set(B) if not k.startswith("_")):
        if "span" not in A[doc] or "span" not in B[doc]:
            continue
        end = min(A[doc]["span"][-1][1], B[doc]["span"][-1][1])
        split = A[doc]["span"][min(bd - 1, len(A[doc]["span"]) - 1)][0]
        r = []
        for X in (A[doc], B[doc]):
            pre = sum(-l for l, s in zip(X["lp"], X["span"]) if s[1] <= split)
            post = sum(-l for l, s in zip(X["lp"], X["span"]) if split < s[1] <= end)
            r.append((pre / split, post / max(1, end - split)))
        print(f"   {doc:12s} chars {end:7d} split {split:6d}: {a} {r[0][0]:.4f} / {r[0][1]:.4f}   "
              f"{b} {r[1][0]:.4f} / {r[1][1]:.4f}   (before / after)")


tok_pair("Pc", "Nc", "CHUNKED production -> NVIDIA (the known failure)")
tok_pair("Nc", "Nf", "NVIDIA chunked: stock -> HORIZON_GDN_H_FIX=1  (the fix)")
tok_pair("Nc", "Nu", "NVIDIA: chunked -> UNCHUNKED")
tok_pair("Nu", "Nf", "NVIDIA: unchunked -> chunked with the fix (should be ~0 both sides)")
tok_pair("Pc", "Pf", "production chunked: stock -> fix  (does the bug touch production?)")
tok_pair("Pc", "Pu", "production: chunked -> UNCHUNKED")
tok_pair("Pu", "Nu", "UNCHUNKED production -> NVIDIA")
tok_pair("Pf", "Nf", "FIXED production -> NVIDIA")
tok_pair("Mc", "Mu", "MoE: chunked -> UNCHUNKED")
tok_pair("Mc", "Mf", "MoE chunked: stock -> fix")
tok_pair("Nc", "Np", "NVIDIA chunked: stock -> HORIZON_FP8_PERCOL=1 (per-column FP8 shard scales)")
char_pair("Pc", "Mc", "chunked production -> MoE")
char_pair("Pu", "Mu", "unchunked production -> MoE")
char_pair("Pf", "Mf", "fixed production -> fixed MoE")
for a in ("Pc", "Pf", "Pu", "Nc", "Nf", "Nu", "Np", "Mc", "Mf", "Mu"):
    X = load(a)
    if X and "prose_16384" in X:
        lp = X["prose_16384"]["lp"]
        print(f"-- {a} (boundary {boundary(a)}) prose_16384 NLL per 2000 tokens:",
              " ".join(f"{-sum(lp[s:s + 2000]) / len(lp[s:s + 2000]):.2f}" for s in range(0, len(lp), 2000)))
