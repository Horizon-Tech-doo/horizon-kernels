# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""tilelang.adaptive: choose the fastest implementation of an operation per shape and batch size, measured on
the GPU it runs on.

No single kernel wins everywhere. A memory-bound decode (a few tokens) and a compute-bound prefill (thousands of
tokens) want different kernels, and the crossover depends on the layer shape, the GPU and even the driver.
``AdaptiveOp`` holds several interchangeable implementations ("variants"), calibrates them per
(shape key, batch bucket) and persists the winners in a JSON table keyed by a device fingerprint. At run time
selection is a dict lookup, so it is safe inside CUDA-graph capture (the choice is fixed per captured shape).

    op = AdaptiveOp("nvfp4_linear", [
        Variant("marlin", marlin_fn),                                   # exact reference, the default
        Variant("fp4x2", fp4x2_fn, rel_tol=0.02),                       # must stay within 2% of the reference
        Variant("fp4x1", fp4x1_fn, tags={"lossy"}, rel_tol=0.15),
    ], default="marlin", reference="marlin", exclude_tags={"lossy"})
    op.calibrate(key=(N, K), make_args=lambda m: [args_for(m, copy) for copy in range(4)])
    y = op(key=(N, K), m=x.shape[0], *args)

Calibration method (chosen so the table reflects how kernels really run, not host overhead or noise):
- timing: CUDA-graph replay of ``iters * len(arg_sets)`` calls (eager fallback if a variant cannot be captured);
  several argument sets (e.g. weight copies) so memory-bound timings are not flattered by L2; median over
  ``repeats`` rounds with the variant order rotated every round.
- both bucket edges are measured (previous bucket + 1 and the bucket itself); a variant wins a bucket only if it
  wins at every probe point.
- a non-default variant must beat the default by ``margin`` (default 10%) at every probe; near-ties go to the
  default, which is expected to be the exact implementation. Variants tagged "exact" (same numerics class as the
  default) only need ``exact_margin`` (default 2%, enough to reject timing noise).
- correctness gate: every variant's output on the first argument set is compared with the ``reference`` variant;
  a variant whose relative error exceeds its ``rel_tol`` is disqualified for that key.
Variants tagged with any of ``exclude_tags`` are never selected; this is how an accuracy policy is expressed.
"""

from __future__ import annotations

import functools
import json
import os
import statistics
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Sequence

__all__ = ["Variant", "AdaptiveOp", "DEFAULT_BUCKETS", "device_fingerprint"]

DEFAULT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
TABLE_VERSION = 2


@dataclass
class Variant:
    name: str
    fn: Callable[..., Any]
    supports: Callable[[Hashable, int], bool] = lambda key, m: True
    tags: set = field(default_factory=set)
    rel_tol: float | None = None   # max relative error vs the reference variant; None = not checked


def _cuda_driver_api_version() -> str:
    """Version reported by the libcuda actually loaded in this process (reflects cuda-compat forward-compat libs)."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return f"{v.value // 1000}.{(v.value % 1000) // 10}"
    except OSError:
        pass
    return "unknown"


def _kernel_driver_version() -> str:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetDriverVersion()
        return v.decode() if isinstance(v, bytes) else str(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return out[0] if out else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@functools.lru_cache(maxsize=1)
def device_fingerprint() -> str:
    """GPU name + compute capability + kernel driver + CUDA driver API (libcuda) + CUDA runtime."""
    import torch
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (f"{props.name}|sm{props.major}{props.minor}|drv{_kernel_driver_version()}"
            f"|libcuda{_cuda_driver_api_version()}|cuda{torch.version.cuda}")


def _default_table_dir() -> Path:
    return Path(os.environ.get("TILELANG_ADAPTIVE_DIR", Path.home() / ".cache" / "tilelang" / "adaptive"))


def _rel_err(out, ref) -> float:
    o, r = out.float(), ref.float()
    return ((o - r).norm() / r.norm().clamp_min(1e-30)).item()


class AdaptiveOp:
    def __init__(self, name: str, variants: Sequence[Variant], buckets: Sequence[int] = DEFAULT_BUCKETS,
                 exclude_tags: Iterable[str] = (), table_dir: str | os.PathLike | None = None,
                 default: str | None = None, reference: str | None = None, margin: float = 0.10,
                 exact_margin: float = 0.02):
        assert variants, "need at least one variant"
        self.name = name
        self.variants = {v.name: v for v in variants}
        self.buckets = tuple(sorted(set(int(b) for b in buckets)))
        assert self.buckets and self.buckets[0] >= 1
        self.exclude_tags = set(exclude_tags)
        self.default = default or variants[0].name
        self.reference = reference
        self.margin = margin
        self.exact_margin = exact_margin
        assert self.default in self.variants and (reference is None or reference in self.variants)
        self.table_dir = Path(table_dir) if table_dir else _default_table_dir()
        self._lock = threading.Lock()
        self._table: dict[str, dict[str, str]] | None = None      # key -> {bucket: variant}
        self._timings: dict[str, dict[str, dict[str, float]]] = {}  # key -> {m: {variant: median_us}}
        self._errors: dict[str, dict[str, dict[str, float]]] = {}   # key -> {m: {variant: rel_err}}

    # ---------------------------------------------------------------- table
    @property
    def table_path(self) -> Path:
        fp = device_fingerprint()
        for ch in "/ |":
            fp = fp.replace(ch, "_")
        return self.table_dir / f"{self.name}__{fp}.json"

    def table_exists(self) -> bool:
        return self.table_path.exists()

    def has_key(self, key: Hashable) -> bool:
        return str(key) in self._load()

    def _load(self):
        if self._table is None:
            with self._lock:
                if self._table is None:
                    try:
                        data = json.loads(self.table_path.read_text())
                        if data.get("version") != TABLE_VERSION:
                            raise ValueError("stale table version")
                        self._table = data.get("winners", {})
                        self._timings = data.get("timings_us", {})
                        self._errors = data.get("rel_err", {})
                    except (FileNotFoundError, json.JSONDecodeError, ValueError):
                        self._table, self._timings, self._errors = {}, {}, {}
        return self._table

    def save(self):
        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.table_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": TABLE_VERSION, "op": self.name, "device": device_fingerprint(),
                                   "method": {"margin": self.margin, "exact_margin": self.exact_margin, "default": self.default,
                                              "reference": self.reference, "exclude_tags": sorted(self.exclude_tags)},
                                   "winners": self._table, "timings_us": self._timings, "rel_err": self._errors},
                                  indent=1, sort_keys=True))
        tmp.replace(self.table_path)

    # ------------------------------------------------------------ selection
    def bucket(self, m: int) -> int:
        for b in self.buckets:
            if m <= b:
                return b
        return self.buckets[-1]

    def _allowed(self, v: Variant, key, m) -> bool:
        return not (v.tags & self.exclude_tags) and v.supports(key, m)

    def select(self, key: Hashable, m: int) -> Variant:
        name = self._load().get(str(key), {}).get(str(self.bucket(m)))
        v = self.variants.get(name) if name else None
        if v is not None and self._allowed(v, key, m):
            return v
        d = self.variants[self.default]
        if self._allowed(d, key, m):
            return d
        return next(v for v in self.variants.values() if self._allowed(v, key, m))

    def __call__(self, key: Hashable, m: int, *args, **kwargs):
        return self.select(key, m).fn(*args, **kwargs)

    # ---------------------------------------------------------- calibration
    @staticmethod
    def _time_us(fn, arg_sets, iters, warmup, graph):
        import torch
        for _ in range(warmup):
            for a in arg_sets:
                fn(*a)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        n = iters * len(arg_sets)
        if graph:
            try:
                g = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(g, stream=stream):
                        for _ in range(iters):
                            for a in arg_sets:
                                fn(*a)
                torch.cuda.synchronize()
                g.replay()
                torch.cuda.synchronize()
                s.record(); g.replay(); e.record(); torch.cuda.synchronize()
                return s.elapsed_time(e) * 1e3 / n, "graph"
            except Exception:  # noqa: BLE001  not capturable: fall back to eager timing
                torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            for a in arg_sets:
                fn(*a)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3 / n, "eager"

    def calibrate(self, key: Hashable, make_args: Callable[[int], list[tuple]], ms: Sequence[int] | None = None,
                  iters: int = 5, repeats: int = 5, warmup: int = 2, graph: bool = True, save: bool = True,
                  log: Callable[[str], None] | None = print):
        """Measure every variant supporting (key, m) at both edges of each bucket and record the winners.

        ``make_args(m)`` returns a list of argument tuples (e.g. several weight copies so memory-bound timings are not
        flattered by L2); timing cycles through them. ``ms`` restricts calibration to the buckets containing them.
        """
        self._load()
        k = str(key)
        ktab = self._table.setdefault(k, {})
        ttab = self._timings.setdefault(k, {})
        etab = self._errors.setdefault(k, {})
        targets = sorted({self.bucket(m) for m in ms}) if ms else list(self.buckets)
        for b in targets:
            i = self.buckets.index(b)
            probes = sorted({(self.buckets[i - 1] + 1) if i > 0 else b, b})
            per_probe = {}
            disq = set()
            for m in probes:
                arg_sets = make_args(m)
                cands = [v for v in self.variants.values() if v.supports(key, m)]
                # correctness gate against the reference variant
                if self.reference is not None and any(v.name == self.reference for v in cands):
                    ref = self.variants[self.reference].fn(*arg_sets[0])
                    errs = {}
                    for v in cands:
                        try:
                            errs[v.name] = 0.0 if v.name == self.reference else _rel_err(v.fn(*arg_sets[0]), ref)
                        except Exception as ex:  # noqa: BLE001
                            errs[v.name] = float("inf")
                            if log:
                                log(f"[adaptive:{self.name}] {v.name} failed key={key} m={m}: {type(ex).__name__}: {ex}")
                        if v.rel_tol is not None and not errs[v.name] <= v.rel_tol:
                            disq.add(v.name)
                    etab[str(m)] = {n: round(x, 6) for n, x in errs.items()}
                # interleaved repeated timing, median per variant
                samples = {v.name: [] for v in cands if v.name not in disq}
                mode = {}
                names = list(samples)
                for r in range(repeats):
                    for j in range(len(names)):
                        n = names[(j + r) % len(names)]
                        try:
                            t, mode[n] = self._time_us(self.variants[n].fn, arg_sets, iters, warmup if r == 0 else 0, graph)
                            samples[n].append(t)
                        except Exception as ex:  # noqa: BLE001  a variant that fails for a shape simply loses
                            if log and r == 0:
                                log(f"[adaptive:{self.name}] {n} failed key={key} m={m}: {type(ex).__name__}: {ex}")
                med = {n: statistics.median(ts) for n, ts in samples.items() if ts}
                per_probe[m] = med
                ttab[str(m)] = {n: round(t, 2) for n, t in med.items()}
                if log:
                    log(f"[adaptive:{self.name}] key={key} m={m}: " + ", ".join(
                        f"{n}={t:.1f}us({mode.get(n, '?')})" for n, t in sorted(med.items(), key=lambda x: x[1]))
                        + (f"  disqualified={sorted(disq)}" if disq else ""))
                del arg_sets
            # winner: fastest allowed variant at every probe, beating the default by the margin; else the default
            allowed = [n for n in self.variants if n not in disq and not (self.variants[n].tags & self.exclude_tags)
                       and all(n in per_probe[m] for m in probes)]
            winner = self.default if self.default in allowed else None
            if allowed:
                fastest = {m: min(allowed, key=lambda n: per_probe[m][n]) for m in probes}
                cand = fastest[probes[-1]]
                same = all(f == cand for f in fastest.values())
                mg = self.exact_margin if "exact" in self.variants[cand].tags else self.margin
                beats = cand == self.default or winner is None or all(
                    per_probe[m][cand] * (1 + mg) <= per_probe[m][self.default] for m in probes)
                if same and beats:
                    winner = cand
                elif winner is None:
                    winner = cand
            if winner:
                ktab[str(b)] = winner
            if log:
                log(f"[adaptive:{self.name}] key={key} bucket<={b} (probes {probes}) -> {winner}")
        if save:
            self.save()
        return ktab
