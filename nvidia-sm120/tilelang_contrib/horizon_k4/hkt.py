# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""HKT: Horizon in-kernel timing (our profiler; K4 foundations, 27-09-2026).

Why our own: Nsight Compute segfaults processes with driver 570 + CUDA forward-compat libs, and upstream TileLang's
IKET (tilelang.tools.cuda.iket) only EMITS events: its TL_IKET_EVENT macro stores a %globaltimer_lo word into shared
memory offset 0x20 and raises `pmevent`, and an external IKET collector (a separate CUDA tool, not on PyPI, not on this
box) is what reserves that shared word and dumps the records. Without the collector an IKET-instrumented kernel would
overwrite its own shared memory. HKT keeps IKET's idea (timestamped per-CTA events, compiled in only on request) but
writes plain int64 records into a global tensor the caller passes, so it needs nothing but the kernel itself.

Record layout: TS[cta, slot] int64, SLOTS = 8 per CTA (thread 0 of the CTA writes them):
  ENTRY  %globaltimer (ns) at kernel entry
  SYNC   after T.pdl_sync (= ENTRY + a few ns without PDL)
  LOOP   after the main (k) loop
  EXIT   after the epilogue / split-K reduction, behind a CTA barrier (the whole CTA is done)
  SMID   the SM the CTA ran on
  AUX    kernel specific (regfrag: 1 if this CTA did the split-K reduction)
%globaltimer is one clock shared by all SMs (so CTAs of different kernels line up on one timeline); its update
granularity is implementation-defined (our measurement is in our run notes),
so measure it on the target GPU before trusting short intervals.
Cost: a handful of instructions per CTA plus one extra barrier before EXIT; kernels built with stamps=0 are unchanged.
"""

import tilelang.language as T

SLOTS = 8
ENTRY, SYNC, LOOP, EXIT, SMID, AUX = 0, 1, 2, 3, 4, 5

HKT_SRC = r"""
#ifndef HZ_HKT_SRC_DEFINED
#define HZ_HKT_SRC_DEFINED
__device__ __forceinline__ long long hz_gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return (long long)t;
}
__device__ __forceinline__ long long hz_smid() {
  unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return (long long)r;
}
#endif
"""


@T.macro
def stamp(TS, cta, slot, tid):
    """Thread 0 writes %globaltimer into TS[cta, slot]."""
    if tid == 0:
        TS[cta, slot] = T.call_extern(T.int64, "hz_gtime")


@T.macro
def stamp_smid(TS, cta, tid):
    if tid == 0:
        TS[cta, SMID] = T.call_extern(T.int64, "hz_smid")


@T.macro
def stamp_exit(TS, cta, tid):
    """EXIT stamp after a CTA barrier, so it marks the end of the whole CTA's work."""
    T.sync_threads()
    if tid == 0:
        TS[cta, EXIT] = T.call_extern(T.int64, "hz_gtime")


# ---------------------------------------------------------------------------------------------------------------
# Host side: buffers and summaries.
def buffer(n_ctas, device="cuda"):
    import torch
    return torch.zeros(n_ctas, SLOTS, dtype=torch.int64, device=device)


def kernel_summary(ts, n_sms=188, name=""):
    """Per-launch summary of one kernel's records (rows with ENTRY == 0 are unused and dropped).

    span      first ENTRY -> last EXIT (ns)
    wait      per CTA SYNC - ENTRY (time parked in griddepcontrol.wait)
    loop      per CTA LOOP - SYNC (main loop), epi: EXIT - LOOP (epilogue + split-K reduction)
    busy      sum over CTAs of (EXIT - SYNC) / (n_sms * span): the fraction of SM-time this kernel did work
              (> 1 is possible when several CTAs share an SM)
    tail      last EXIT - median EXIT (the stragglers everybody waits for at kernel end)
    sm_idle   per SM, the time inside the span with no CTA of this kernel active (mean over SMs)"""
    import torch
    t = ts[ts[:, ENTRY] != 0].to("cpu")
    if t.numel() == 0:
        return {"name": name, "ctas": 0}
    e, s, lp, x = (t[:, i].double() for i in (ENTRY, SYNC, LOOP, EXIT))
    t0, t1 = e.min().item(), x.max().item()
    span = t1 - t0
    busy = (x - s).clamp_min(0).sum().item() / (n_sms * span) if span > 0 else float("nan")
    # SM idle: union of [SYNC, EXIT) intervals per SM, subtracted from the span
    sm = t[:, SMID]
    idle = []
    for sid in range(n_sms):
        m = sm == sid
        if not bool(m.any()):
            idle.append(span)
            continue
        iv = sorted(zip(s[m].tolist(), x[m].tolist()))
        cov, cur_a, cur_b = 0.0, None, None
        for a, b in iv:
            if cur_b is None or a > cur_b:
                if cur_b is not None:
                    cov += cur_b - cur_a
                cur_a, cur_b = a, b
            else:
                cur_b = max(cur_b, b)
        cov += cur_b - cur_a
        idle.append(span - cov)

    def q(v, p):
        return float(torch.quantile(v, p).item())

    return {
        "name": name, "ctas": int(t.shape[0]), "t0": t0, "t1": t1, "span_us": span / 1e3,
        "wait_us_med": q(s - e, 0.5) / 1e3, "wait_us_max": float((s - e).max()) / 1e3,
        "loop_us_med": q(lp - s, 0.5) / 1e3, "loop_us_max": float((lp - s).max()) / 1e3,
        "epi_us_med": q(x - lp, 0.5) / 1e3, "epi_us_max": float((x - lp).max()) / 1e3,
        "start_skew_us": float((e - t0).max()) / 1e3, "tail_us": (t1 - q(x, 0.5)) / 1e3,
        "busy": busy, "sm_idle_us_mean": sum(idle) / len(idle) / 1e3,
        "reducers": int((t[:, AUX] == 1).sum()),
        "sms_used": int(torch.unique(sm).numel()),
    }


def activity_curve(ts, bins=20, n_sms=188):
    """Fraction of SMs with at least one CTA of this kernel between SYNC and EXIT, in `bins` equal slices of the
    kernel's span (first ENTRY .. last EXIT): the utilisation curve a timeline plot would show."""
    t = ts[ts[:, ENTRY] != 0].to("cpu").double()
    if t.numel() == 0:
        return []
    t0, t1 = t[:, ENTRY].min().item(), t[:, EXIT].max().item()
    w = (t1 - t0) / bins
    out = []
    for b in range(bins):
        a, z = t0 + b * w, t0 + (b + 1) * w
        act = (t[:, SYNC] < z) & (t[:, EXIT] > a)
        out.append(round(float(torch_unique_count(t[act][:, SMID])) / n_sms, 3))
    return out


def torch_unique_count(v):
    import torch
    return int(torch.unique(v).numel()) if v.numel() else 0


def chain_summary(named_ts, n_sms=188):
    """[(name, ts), ...] in launch order -> per-kernel summaries plus the gap to the previous kernel:
    gap_us   = this kernel's first ENTRY - previous kernel's last EXIT (negative: the CTAs were resident early (PDL))
    ready_us = this kernel's median SYNC - previous kernel's last EXIT (the real hand-over: > 0 is idle GPU time)"""
    import torch
    out, prev_t1 = [], None
    for name, ts in named_ts:
        k = kernel_summary(ts, n_sms, name)
        if k.get("ctas"):
            t = ts[ts[:, ENTRY] != 0].to("cpu")
            sync_med = float(torch.quantile(t[:, SYNC].double(), 0.5).item())
            sync_first = float(t[:, SYNC].min().item())
            if prev_t1 is not None:
                k["gap_us"] = (k["t0"] - prev_t1) / 1e3
                k["ready_us"] = (sync_med - prev_t1) / 1e3
                k["first_sync_us"] = (sync_first - prev_t1) / 1e3
            prev_t1 = k["t1"]
        out.append(k)
    return out
