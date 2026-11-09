# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Programmatic Dependent Launch (PDL) helpers for the Horizon kernels (K4 foundations, 27-09-2026).

PDL lets kernel N+1 start while kernel N is still running. Two PTX instructions (upstream TileLang: T.pdl_trigger /
T.pdl_sync; the launcher adds cudaLaunchAttributeProgrammaticStreamSerialization to every kernel that calls T.pdl_sync):

  griddepcontrol.launch_dependents  (T.pdl_trigger)  in kernel N: "the next kernel may be scheduled now". The next grid
      is launched once EVERY CTA of kernel N has executed it (or exited). It says nothing about memory: it only lets the
      next grid's CTAs become resident early.
  griddepcontrol.wait               (T.pdl_sync)     in kernel N+1: blocks the calling thread until kernel N has
      COMPLETED and all its memory writes are visible. Everything after it may read N's outputs.

Rules our kernels follow (they make PDL safe inside CUDA graphs and with PyTorch's caching allocator):
  1. Before T.pdl_sync a kernel writes NO global memory (outputs, split-K partials, ticket counters) and reads only
     constant data (weights, scales, alpha, norm weights). Output / workspace memory may still be in use by kernel N
     (the allocator assumes stream order), and activations are kernel N's output.
  2. Every thread that later touches dependent data executes T.pdl_sync (it is per-thread, not per-CTA). Our kernels
     call it once, uniformly, before the main loop.
  3. The chain is transitive: kernel N passed its own T.pdl_sync before it could complete, so after N+1's T.pdl_sync
     everything before N is complete too, even when N triggered before its own sync.

Flags (one int `pdl` per kernel builder; 0 = off = the unchanged kernel, identical generated source):
  PDL_ON   = 1  T.pdl_trigger at kernel entry + T.pdl_sync before the first read of dependent data.
  PDL_PF   = 2  (with ON) before the sync, prefetch the CTA's first weight tiles (packed FP4 + scales, or BF16 rows)
                into L2 (prefetch.global.L2: no registers, no shared memory, no completion to wait for), so the main loop
                starts on L2 hits while kernel N drains.
  PDL_LATE = 4  (with ON) trigger after the main loop instead of at entry: the next grid becomes resident only while this
                one runs its epilogue / split-K reduction (less cascading of waiting grids; see nvidia-sm120/docs/k4-design.md).

__restrict__ and PDL: upstream's CUDA codegen drops __restrict__ from EVERY kernel parameter of a kernel that calls
T.pdl_sync (codegen_cuda.cc, PrintFunctionSignature). The reason is sound for data written by the previous kernel: a
load through a `const T* __restrict__` pointer may be turned into a read-only (ld.global.nc) load that the compiler can
hoist above griddepcontrol.wait, reading stale data. But it also drops it from parameters that are constant for the
whole model (weights, scales, alpha), and in our GEMMs that doubles the register count (regfrag_xl qkvz M=4: 72 -> 138,
gate_up M=32: 102 -> 220 registers; measured offline 27-09-2026 with ptxas): without __restrict__ on alpha the epilogue's
alpha loads cannot be kept apart from the Y stores. restrict_marker() below puts a comment into the kernel source
naming the parameters that are provably constant; a CUDA post-processing callback (TileLang's documented
tilelang_callback_cuda_postproc hook, the same one IKET uses) re-adds __restrict__ to exactly those parameters, so
the pdl kernels keep the pdl=0 register counts. Parameters carrying the previous kernel's outputs never get it.
"""

import re

import tilelang.language as T

PDL_ON = 1
PDL_PF = 2
PDL_LATE = 4

# Host-side default for the kernels' linear() wrappers when their `pdl` argument is None (0: production behaviour).
DEFAULT = 0
# Optional per-batch-size policy: [(max_rows or None, flags), ...], first match wins (set_policy). The 27-09-2026
# chain A/B found early trigger + prefetch (3) best at M <= 8 and a late trigger (5) best at M = 16-32; above the
# measured range the kernels run without PDL.
POLICY = None


def set_default(pdl: int) -> None:
    """Set the pdl flags every Horizon linear() uses when called without an explicit `pdl` (tests / A-B only)."""
    global DEFAULT, POLICY
    DEFAULT = check(pdl)
    POLICY = None


def parse_policy(spec: str):
    """"3@8,5@64,0" -> [(8, 3), (64, 5), (None, 0)]: flags 3 for M <= 8, 5 for M <= 64, 0 above. A bare "3" means
    flags 3 for every M. Empty / "0" -> no PDL."""
    out = []
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "@" in part:
            f, m = part.split("@")
            out.append((int(m), check(int(f))))
        else:
            out.append((None, check(int(part))))
    if not out or out[-1][0] is not None:
        out.append((None, 0))
    return out


def set_policy(spec) -> None:
    """Per-batch-size PDL flags for every wrapper called without an explicit `pdl` (vLLM plugin: HORIZON_PDL)."""
    global DEFAULT, POLICY
    POLICY = parse_policy(spec)
    DEFAULT = 0


def policy_str() -> str:
    if POLICY is None:
        return str(DEFAULT)
    return ",".join(f"{f}@{m}" if m is not None else str(f) for m, f in POLICY)


def warm_ms(max_m: int):
    """Row counts that together hit every flag value the policy can pick for 1 <= M <= max_m (for warm-ups of
    kernels whose binary depends only on the flags, e.g. dynamic-M norms): 1, max_m and both sides of each bucket
    edge."""
    ms = {1, max_m}
    for m, _ in (POLICY or []):
        if m is not None:
            ms.update(x for x in (m, m + 1) if 1 <= x <= max_m)
    return sorted(ms)


def resolve(pdl, M=None) -> int:
    """Explicit flags win; otherwise the per-M policy (when M is known), else DEFAULT."""
    if pdl is not None:
        return check(pdl)
    if POLICY is not None:
        if M is None:
            return DEFAULT
        for m, f in POLICY:
            if m is None or M <= m:
                return f
    return DEFAULT

PDL_SRC = r"""
#ifndef HZ_PDL_SRC_DEFINED
#define HZ_PDL_SRC_DEFINED
// L2 prefetch of one 128-byte line (fire and forget; weights are read-only for the whole decode step)
__device__ __forceinline__ void hz_pf_l2(const void *p) { asm volatile("prefetch.global.L2 [%0];" :: "l"(p)); }
#endif
"""


_MARK_RE = re.compile(r"// HZ_PDL_RESTRICT\(([A-Za-z0-9_,]*)\)")
_CB = "tilelang_callback_cuda_postproc"
_installed = False
_prev_cb = None


def add_restrict(code: str, names) -> str:
    """Add __restrict__ to the named pointer parameters in every `extern "C" __global__` signature line of code."""
    out = []
    for line in code.splitlines(True):
        if line.startswith('extern "C" __global__'):
            for n in names:
                line = re.sub(r"\*\s+" + re.escape(n) + r"\b(?=\s*[,)])", "* __restrict__ " + n, line)
        out.append(line)
    return "".join(out)


def _postproc(code, target):
    if _prev_cb is not None:
        code = _prev_cb(code, target)
    m = _MARK_RE.search(code)
    if m:
        code = add_restrict(code, [n for n in m.group(1).split(",") if n])
    return code


def install_restrict_callback() -> None:
    """Register the post-processing callback once per process (chains to a callback registered before it)."""
    global _installed, _prev_cb
    if _installed:
        return
    import tvm_ffi
    prev = tvm_ffi.get_global_func(_CB, allow_missing=True)
    _prev_cb = prev
    tvm_ffi.register_global_func(_CB, f=_postproc, override=True)
    _installed = True


def restrict_marker(*names) -> str:
    """Prelude line naming the kernel parameters that are constant for the whole model (weights, scales, alpha): they
    keep __restrict__ under PDL. Installs the callback that applies it."""
    install_restrict_callback()
    return "\n// HZ_PDL_RESTRICT(" + ",".join(names) + ")\n"


def check(pdl: int) -> int:
    pdl = int(pdl)
    assert pdl in (0, 1, 3, 5, 7), f"pdl flags must be 0 or ON(1) [+PF(2)] [+LATE(4)], got {pdl}"
    return pdl


def trigger_early(pdl: int) -> bool:
    return bool(pdl & PDL_ON) and not (pdl & PDL_LATE)


def trigger_late(pdl: int) -> bool:
    return bool(pdl & PDL_ON) and bool(pdl & PDL_LATE)


def prefetch(pdl: int) -> bool:
    return bool(pdl & PDL_ON) and bool(pdl & PDL_PF)


def l2_prefetch_blocks(nr: int, nc: int, E: int, threads: int):
    """Macro: prefetch into L2 the rows r0 .. r0+nr-1, k chunks c0 .. c0+min(nc, ncl)-1 of an int32 tensor laid out
    (R, C, E) (regfrag's Wg: E = 128, 512 B per chunk; Sg: E = 16, 64 B per chunk). Each row's chunk range is
    contiguous (nc * E * 4 bytes), cut into 128-byte lines spread over the CTA's threads."""
    lines_per_row = (nc * E + 31) // 32
    total = nr * lines_per_row
    iters = (total + threads - 1) // threads

    @T.macro
    def pf(buf, r0, c0, ncl, tid):
        for it in T.serial(iters):
            idx = it * threads + tid
            if idx < total:
                if (idx % lines_per_row) * 32 // E < ncl:
                    T.call_extern("handle", "hz_pf_l2",
                                  T.access_ptr(buf[r0 + idx // lines_per_row, c0 + (idx % lines_per_row) * 32 // E,
                                                   (idx % lines_per_row) * 32 % E], "r"))

    return pf


def l2_prefetch_rows_bf16(nr: int, K: int, threads: int):
    """Macro: prefetch into L2 rows r0 .. r0+nr-1 of a (N, K) bf16 tensor (K * 2 bytes per row, 64 values per line)."""
    lines_per_row = (K + 63) // 64
    total = nr * lines_per_row
    iters = (total + threads - 1) // threads

    @T.macro
    def pf(buf, r0, tid):
        for it in T.serial(iters):
            idx = it * threads + tid
            if idx < total:
                T.call_extern("handle", "hz_pf_l2",
                              T.access_ptr(buf[r0 + idx // lines_per_row, (idx % lines_per_row) * 64], "r"))

    return pf
