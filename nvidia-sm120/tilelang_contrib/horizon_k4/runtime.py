# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""K4 persistent-kernel runtime (phase 2a, 27-09-2026): task descriptors, a static scheduler, cross-CTA counters.

A K4 kernel is launched with exactly one CTA per SM (NCTA = 188 on the RTX PRO 6000) and every CTA walks its own list
of tasks ("one tile of one GEMM", "one row of one norm", ...). CTAs hand results to each other through int32 counters in
global memory instead of kernel boundaries:

  producer task:  all threads store their outputs -> CTA barrier -> thread 0: red.release.gpu.global.add [counter], 1
  consumer task:  thread 0 spins on ld.acquire.gpu [counter] >= target (pure spin by default; a %globaltimer timeout
                  raises an error flag instead of hanging) -> CTA barrier -> every thread reads.
  Hand-over latency (rt_stress.py chain, 27-09-2026): 1.07 us per hop with a 32 -> 256 ns nanosleep back-off and an
  extra fence; the pure-spin number is in our run notes.
  Dependent data (activations written by other CTAs in the same launch) must be read through L2: ld.global.cg /
  cp.async.cg, never the L1-cached or read-only (.nc) path: an L1 line fetched earlier by this SM can hold stale bytes
  of a neighbouring slot written later by another SM (selftest.py provokes exactly that with plain loads).

Deadlock freedom (nvidia-sm120/docs/k4-design.md, 4.2): every CTA's list is a subsequence of ONE global topological order and all
NCTA CTAs are co-resident (grid = NCTA, one CTA per SM fits). A K4 kernel that uses PDL waits in griddepcontrol.wait
before its first counter spin and triggers its dependents only after its last task, so a dependent grid can never take
SM space from CTAs of this grid that are not resident yet.

Counters reset themselves: the last CTA to finish (a `done` ticket) zeroes every counter and the ticket, so a CUDA
graph replays without a memset (the split-K ticket trick of regfrag).

Descriptor (int32 x 8 per task): [op, a0, a1, a2, wait0, target0, wait1, target1]; wait = -1: none. The task signals
the counter `sig[op](a0, a1, a2)` computed in the kernel (ops define their own counter layout), or the schedule stores
it: we keep it explicit in a second table so the kernel does not need per-op arithmetic: sigtab[cta, t] = counter id
or -1.
"""

import tilelang.language as T

NCTA_DEFAULT = 188
D_OP, D_A0, D_A1, D_A2, D_W0, D_T0, D_W1, D_T1 = range(8)
DESC = 8
ERR_TIMEOUT = 1

_RT_TMPL = r"""
#ifndef HZ_HKT_SRC_DEFINED
#define HZ_HKT_SRC_DEFINED
__device__ __forceinline__ long long hz_gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return (long long)t;
}
__device__ __forceinline__ long long hz_smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return (long long)r;
}
#endif
#ifndef HZ_RT_SRC_DEFINED
#define HZ_RT_SRC_DEFINED
__device__ __forceinline__ int hz_ld_acq(const int *p) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int hz_ld_cg(const int *p) {
  int v; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
// producer side, thread 0 after a CTA barrier: release-add (cumulative over the CTA's stores ordered by the barrier;
// the same pattern as CUTLASS's GenericBarrier::arrive_inc)
__device__ __forceinline__ void hz_signal(int *p) {
  asm volatile(HZ_SIGNAL_ASM :: "l"(p) : "memory");
}
// consumer side, thread 0: spin until *cnt >= target. Returns 0, or 1 on timeout / when another CTA raised the error
// flag (then the caller skips its work; the host sees err != 0 and must not trust the outputs). The error flag and
// the %globaltimer timeout are checked every 64 polls only (they cost an extra load each). On a timeout the flag is
// also written to errh, a word of page-locked HOST memory mapped into the device (runtime.HostFlag): the host sees it
// without any CUDA call (a watchdog thread polls it; no stream sync, no graph break).
__device__ __forceinline__ void hz_raise(int *err, int *errh) {
  atomicExch(err, 1);
  *(volatile int *)errh = 1;
  __threadfence_system();
}
__device__ __forceinline__ int hz_wait_ge(const int *cnt, int target, int *err, int *errh, long long timeout_ns) {
  long long t0 = 0;
  unsigned n = 0;
  while (true) {
    if (hz_ld_acq(cnt) >= target) return 0;
    if ((++n & 63u) == 0u) {
      if (*(volatile int *)err) return 1;
      long long now = hz_gtime();
      if (t0 == 0) t0 = now;
      else if (now - t0 > timeout_ns) { hz_raise(err, errh); return 1; }
    }
    HZ_SPIN_PAUSE
  }
}
// end of a CTA's list: the last CTA to arrive zeroes the counters and the ticket (graph replay without a memset)
__device__ __forceinline__ int hz_done_ticket(int *done) {
  int v; asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;" : "=r"(v) : "l"(done) : "memory");
  return v;
}
#endif
"""


def rt_src(sleep_ns: int = 0, fence: bool = False) -> str:
    """Device helpers. sleep_ns: __nanosleep between polls (0 = pure spin: lowest hand-over latency; the spinning
    thread is one of 128, the other warps wait in the CTA barrier and issue nothing). fence: an extra fence.acq_rel.gpu
    before the release-add (not needed by the PTX memory model; kept for A/B)."""
    sig = '"fence.acq_rel.gpu;\\n\\tred.release.gpu.global.add.s32 [%0], 1;"' if fence else \
        '"red.release.gpu.global.add.s32 [%0], 1;"'
    pause = f"__nanosleep({sleep_ns});" if sleep_ns else ""
    return (_RT_TMPL.replace("HZ_SIGNAL_ASM", sig).replace("HZ_SPIN_PAUSE", pause)
            .replace("HZ_RT_SRC_DEFINED", f"HZ_RT_SRC_DEFINED_{sleep_ns}_{int(fence)}"))


RT_SRC = rt_src()


# ---------------------------------------------------------------------------------------------------------------
# Device-side macros (use inside `with T.Kernel(NCTA, threads=...)`). The task's descriptor is first copied into a
# local int32[8] `dl` (load_desc); ok_s / tk_s are 1-int shared buffers.
@T.macro
def load_desc(desc, cta, t, dl):
    for i in T.unroll(DESC):
        dl[i] = desc[cta, t, i]


def macros(timeout_ns: int = 2_000_000_000):
    """Returns (task_wait, task_signal, grid_end). All threads of the CTA call them (they contain CTA barriers)."""

    @T.macro
    def task_wait(dl, counters, err, ok_s, tid, errh):
        """Waits for the task's (up to two) input counters. Leaves ok_s[0] = 1 when the task may run.
        errh: host-mapped error mirror (HostFlag.dev; pass err itself when there is none)."""
        if tid == 0:
            ok_s[0] = 1
            if dl[D_W0] >= 0:
                if T.call_extern(T.int32, "hz_wait_ge", T.access_ptr(counters[dl[D_W0]], "r"), dl[D_T0],
                                 T.access_ptr(err[0], "rw"), T.access_ptr(errh[0], "rw"), T.int64(timeout_ns)) != 0:
                    ok_s[0] = 0
            if dl[D_W1] >= 0:
                if T.call_extern(T.int32, "hz_wait_ge", T.access_ptr(counters[dl[D_W1]], "r"), dl[D_T1],
                                 T.access_ptr(err[0], "rw"), T.access_ptr(errh[0], "rw"), T.int64(timeout_ns)) != 0:
                    ok_s[0] = 0
        T.sync_threads()

    @T.macro
    def task_signal(sig, counters, tid, go):
        """After the task's stores: CTA barrier, then thread 0 release-adds 1 to counter `sig` (if sig >= 0 and the
        uniform condition `go` holds: e.g. only the split-K task that completed the tile)."""
        T.sync_threads()
        if tid == 0:
            if sig >= 0:
                if go:
                    T.call_extern("handle", "hz_signal", T.access_ptr(counters[sig], "rw"))

    @T.macro
    def grid_end(counters, n_counters, done, ncta, tid, tk_s, threads):
        """Last CTA to finish zeroes every counter and the ticket."""
        T.sync_threads()
        if tid == 0:
            tk_s[0] = T.call_extern(T.int32, "hz_done_ticket", T.access_ptr(done[0], "rw"))
        T.sync_threads()
        if tk_s[0] == ncta - 1:
            for i in T.serial(T.ceildiv(n_counters, threads)):
                if i * threads + tid < n_counters:
                    counters[i * threads + tid] = 0
            T.sync_threads()
            if tid == 0:
                T.call_extern("handle", "__threadfence")
                done[0] = 0

    return task_wait, task_signal, grid_end


# ---------------------------------------------------------------------------------------------------------------
# Host side: the static scheduler
class Schedule:
    """Builds per-CTA task lists for one K4 launch.

    add(op, tasks) appends an op: `tasks` is a list of (a0, a1, a2, waits, signal) with waits a list of up to two
    (counter, target) pairs and signal a counter id or -1. Ops must be added in a topological order (every wait refers
    to counters signalled by ops added earlier); tasks of one op keep their order. Task i of an op goes to CTA
    (base + i) % ncta, where base continues where the previous op stopped (load spreads evenly); `cta_of` overrides it.
    Each CTA's list is its tasks in the global (op, index) order: a subsequence of one topological order."""

    def __init__(self, ncta: int = NCTA_DEFAULT):
        self.ncta = ncta
        self.lists = [[] for _ in range(ncta)]
        self.base = 0
        self.n_counters = 0
        self.targets = {}          # counter -> number of signals per launch (sanity)
        self.ops = []

    def counters(self, n: int) -> int:
        """Reserve n consecutive counters, return the first id."""
        c = self.n_counters
        self.n_counters += n
        return c

    def add(self, op: int, tasks, cta_of=None, signals_per_task: float = 1.0):
        """signals_per_task < 1: only some tasks signal (split-K: the task completing a tile), for check()."""
        self.ops.append((op, len(tasks)))
        for i, (a0, a1, a2, waits, sig) in enumerate(tasks):
            assert len(waits) <= 2, waits
            for c, tgt in waits:
                assert 0 <= c < self.n_counters and tgt >= 1, (c, tgt)
            cta = cta_of(i) if cta_of is not None else (self.base + i) % self.ncta
            w = list(waits) + [(-1, 0)] * (2 - len(waits))
            self.lists[cta].append((op, a0, a1, a2, w[0][0], w[0][1], w[1][0], w[1][1], sig))
            if sig >= 0:
                self.targets[sig] = self.targets.get(sig, 0) + signals_per_task
        if cta_of is None:
            self.base = (self.base + len(tasks)) % self.ncta

    def check(self):
        """Every waited counter must receive at least `target` signals per launch."""
        for lst in self.lists:
            for d in lst:
                for c, tgt in ((d[4], d[5]), (d[6], d[7])):
                    if c >= 0:
                        got = round(self.targets.get(c, 0), 6)
                        assert got >= tgt, f"counter {c}: waits for {tgt}, signalled {got}"

    def tensors(self, device="cuda"):
        """(desc int32 (ncta, maxt, 8), sigtab int32 (ncta, maxt), ntask int32 (ncta,), counters int32 zeros,
        done int32 (1,), err int32 (1,))."""
        import torch
        self.check()
        maxt = max(1, max(len(l) for l in self.lists))
        desc = torch.full((self.ncta, maxt, DESC), -1, dtype=torch.int32)
        sig = torch.full((self.ncta, maxt), -1, dtype=torch.int32)
        nt = torch.zeros(self.ncta, dtype=torch.int32)
        for c, lst in enumerate(self.lists):
            nt[c] = len(lst)
            for t, d in enumerate(lst):
                desc[c, t] = torch.tensor(d[:8], dtype=torch.int32)
                sig[c, t] = d[8]
        dev = lambda x: x.to(device)  # noqa: E731
        return dict(desc=dev(desc), sigtab=dev(sig), ntask=dev(nt),
                    counters=torch.zeros(max(1, self.n_counters), dtype=torch.int32, device=device),
                    done=torch.zeros(1, dtype=torch.int32, device=device),
                    err=torch.zeros(1, dtype=torch.int32, device=device), maxt=maxt)

    def stats(self):
        n = [len(l) for l in self.lists]
        return {"tasks": sum(n), "max_per_cta": max(n), "min_per_cta": min(n), "counters": self.n_counters}


# ---------------------------------------------------------------------------------------------------------------
# Host side: the timeout guard. A K4 launch that times out has wrong outputs; the device can not stop the step, so the
# host must notice. HostFlag is one int32 in page-locked host memory registered as MAPPED: the kernel writes it through
# its device address, the host reads it with a plain memory load (no CUDA call, no sync, works while CUDA graphs
# replay). Watchdog polls a set of flags from a daemon thread and calls on_trip once.
class HostFlag:
    mapped = True

    def __init__(self):
        import ctypes, mmap
        import torch
        self._buf = mmap.mmap(-1, 4096)                     # page-aligned, zeroed
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        rt = _cudart()
        rt.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        cudaHostRegisterMapped, cudaHostRegisterPortable = 2, 1
        rc = rt.cudaHostRegister(self.addr, 4096, cudaHostRegisterMapped | cudaHostRegisterPortable)
        if rc != 0:
            raise RuntimeError(f"cudaHostRegister failed ({rc})")
        dptr = ctypes.c_void_p()
        rc = rt.cudaHostGetDevicePointer(ctypes.byref(dptr), self.addr, 0)
        if rc != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer failed ({rc})")
        self.dptr = dptr.value
        self._word = ctypes.c_int.from_address(self.addr)
        cai = {"shape": (1,), "typestr": "<i4", "data": (self.dptr, False), "version": 3, "strides": None}
        self._cai = type("_CAI", (), {"__cuda_array_interface__": cai})()
        self.dev = torch.as_tensor(self._cai, device="cuda")   # int32 (1,) CUDA tensor aliasing the host word

    def get(self) -> int:
        return int(self._word.value)

    def clear(self):
        self._word.value = 0


def _cudart():
    import ctypes, glob, os
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    import torch
    dirs = [os.path.dirname(os.path.dirname(torch.__file__))] + os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "nvidia", "**", "libcudart.so*"), recursive=True)
                        + glob.glob(os.path.join(d, "libcudart.so*"))):
            try:
                return ctypes.CDLL(f)
            except OSError:
                pass
    raise OSError("libcudart not found")


class Watchdog:
    """Daemon thread: every `period_s` read each registered HostFlag; on the first non-zero value call on_trip(name)."""

    def __init__(self, on_trip, period_s: float = 0.05):
        import threading
        self.flags = {}
        self.on_trip = on_trip
        self.period = period_s
        self._lock = threading.Lock()
        self._t = threading.Thread(target=self._run, name="horizon-k4-watchdog", daemon=True)
        self._t.start()

    def add(self, name, flag):
        if getattr(flag, "mapped", False):
            with self._lock:
                self.flags[name] = flag

    def _run(self):
        import time
        while True:
            time.sleep(self.period)
            with self._lock:
                items = list(self.flags.items())
            for name, f in items:
                if f.get():
                    self.on_trip(name)
                    return
