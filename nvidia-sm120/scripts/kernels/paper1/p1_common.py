# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the paper-1 NVIDIA re-measurement races.

Method (the standard of the AMD part of this repository):
  * every arm of a cell runs in ONE process on the same input buffers: our kernel ("new"), the vendor kernel(s),
    and "ctl" = the same kernel objects as "new" captured into a second CUDA graph (identical code);
  * >= 9 cycles per cell; each cycle runs every arm once, arm order rotated by one per cycle; --reverse reverses the
    arm list for the second job; every per-cycle value is written, nothing is reduced to a minimum;
  * timing = device events around `reps` CUDA-graph replays after `warm` replays (capture_error_mode="thread_local"
    with retry); an arm that cannot be captured is not timed;
  * correctness per cell before timing: fp64 reference or bit-exact comparison, non-finite count, repeat equality,
    graph replay after poisoning the outputs == eager;
  * energy pass, separate from the timing pass: per arm a tight replay loop for a fixed wall time with a background
    power sampler; no sampler thread exists while the timing pass runs;
  * records are JSONL; every string is passed through mask() so no path, host name, address or device id is kept.

This module imports only the standard library at import time, so --dry-run works on a machine without torch.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

SCHEMA = 1
RESERVE_MIB = 1536          # real GPU memory that must stay free next to the co-resident process
BLOCK_SECONDS = 720         # hard limit of one block (enforced by the caller, not here)

# ------------------------------------------------------------------------------------------------- sanitising
_HOST = socket.gethostname()
_HOME = os.path.expanduser("~")
_USER = os.path.basename(_HOME)
_RX = [
    (re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"']+"), "<url>"),
    (re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"), "<uuid>"),
    (re.compile(r"\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]\b"), "<pci>"),
    (re.compile(r"(?<![\w.<>/~-])~?/(?:[\w.+@%~-]+/)+([\w.+@%~-]*)"), r"<path>/\1"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b(?![.\d+-])"), "<ip>"),
    (re.compile(r"\blocalhost(?::\d+)?\b"), "<local>"),
]


def mask(s):
    """Remove everything that could identify the machine from a string."""
    if not isinstance(s, str):
        return s
    for rx, rep in _RX:
        s = rx.sub(rep, s)
    for word in (_HOST, _HOME, _USER):
        if word and len(word) > 2:
            s = s.replace(word, "<id>")
    return s


def clean(v):
    if isinstance(v, str):
        return mask(v)
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return repr(v)
    return v


def sha(data):
    return hashlib.sha256(data if isinstance(data, (bytes, bytearray)) else str(data).encode()).hexdigest()


_FILE_SHA = {}


def file_sha(path):
    if path not in _FILE_SHA:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            _FILE_SHA[path] = h.hexdigest()
        except OSError:
            _FILE_SHA[path] = None
    return _FILE_SHA[path]


def _kernel_sources(obj, out, depth=0):
    """Collect generated sources from a kernel cache value (kernel object, or a tuple / dict of them)."""
    if depth > 3 or obj is None:
        return
    get = getattr(obj, "get_kernel_source", None)
    if callable(get):
        try:
            out.append(str(get()))
        except Exception as e:  # noqa: BLE001
            out.append("ERR " + type(e).__name__)
        return
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _kernel_sources(x, out, depth + 1)


def code_fingerprint(objs):
    """Hash the code behind an arm. objs: modules, callables, or "so:<substring>" for a mapped shared library.
    Per module: sha256 of its source (or binary) file; if it keeps a `_KERNELS` cache, sha256 over the sorted
    generated sources of every kernel compiled so far and their count. Only hashes and base names are returned."""
    fp = {}
    for o in objs or ():
        if isinstance(o, str) and o.startswith("so:"):
            want = o[3:]
            try:
                with open("/proc/self/maps") as f:
                    paths = sorted({ln.split()[-1] for ln in f if "/" in ln and want in os.path.basename(ln.split()[-1])})
            except OSError:
                paths = []
            for p in paths[:4]:
                fp["so:" + os.path.basename(p)] = file_sha(p)
            if not paths:
                fp[o] = None
            continue
        mod = o if inspect.ismodule(o) else sys.modules.get(getattr(o, "__module__", None) or "")
        if mod is None:
            fp["obj:" + type(o).__name__] = None
            continue
        f = getattr(mod, "__file__", None)
        fp[mod.__name__] = file_sha(f) if f else None
        cache = getattr(mod, "_KERNELS", None)
        if isinstance(cache, dict):
            src = []
            _kernel_sources(cache, src)
            fp[mod.__name__ + ":generated"] = sha("\n@@\n".join(sorted(src)))
            fp[mod.__name__ + ":generated_n"] = len(src)
    return fp


def tree_sha(root, suffixes=(".h", ".cuh", ".cu", ".cc", ".py"), limit=4000):
    """sha256 over (relative name, content hash) of the source files under root; None if root is missing."""
    if not root or not os.path.isdir(root):
        return None
    items = []
    for d, _, files in sorted(os.walk(root)):
        for fn in sorted(files):
            if fn.endswith(suffixes):
                p = os.path.join(d, fn)
                items.append(os.path.relpath(p, root) + ":" + str(file_sha(p)))
                if len(items) >= limit:
                    break
    return sha("\n".join(items)) if items else None


# ------------------------------------------------------------------------------------------------- co-resident server
_METRICS = ("request_success_total", "prompt_tokens_total", "generation_tokens_total", "num_requests_running",
            "num_requests_waiting")
_MRX = re.compile(r"^vllm:(\w+?)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


def _base_url():
    return os.environ.get("VLLM_LOCAL_URL", "http://localhost:8001").rstrip("/")


def server_counters():
    """Sums of a few request counters of the co-resident server, labels dropped. None if they cannot be read."""
    try:
        txt = urllib.request.urlopen(_base_url() + "/metrics", timeout=3).read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for ln in txt.splitlines():
        m = _MRX.match(ln)
        if m and m.group(1) in _METRICS:
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass
    return out or None


def server_sleeping():
    """True / False from the server's is_sleeping endpoint, None if it cannot be read."""
    try:
        txt = urllib.request.urlopen(_base_url() + "/is_sleeping", timeout=3).read().decode("utf-8", "replace")
        return bool(json.loads(txt).get("is_sleeping"))
    except Exception:  # noqa: BLE001
        return None


def counters_moved(a, b):
    """True if the server did work between two counter readings; None if unknown."""
    if not a or not b:
        return None
    for k in set(a) | set(b):
        if k.endswith("_total") and a.get(k) != b.get(k):
            return True
        if not k.endswith("_total") and (a.get(k, 0) or b.get(k, 0)):
            # With P1_ALLOW_SLEEPING=1 only the running gauge and the totals count as work.
            if k == "num_requests_waiting" and os.environ.get("P1_ALLOW_SLEEPING") == "1":
                continue
            return True
    return False


# ------------------------------------------------------------------------------------------------- power
class Power:
    """GPU power from NVML (pynvml), else from an nvidia-smi loop. The reading is for the whole physical GPU: it
    includes whatever the co-resident process does. Nothing is subtracted anywhere."""

    def __init__(self, hz=20.0):
        self.hz = hz
        self.nv = None
        self.h = None
        self.source = None
        self.energy_counter = False
        self.index = int(os.environ.get("P1_NVML_INDEX", "0"))
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            pynvml.nvmlDeviceGetPowerUsage(self.h)
            self.source = "nvml"
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
                self.energy_counter = True
            except Exception:  # noqa: BLE001
                self.energy_counter = False
        except Exception:  # noqa: BLE001
            self.nv = None
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i",
                                    str(self.index)], capture_output=True, text=True, timeout=10)
                float(r.stdout.strip().splitlines()[0])
                self.source = "nvidia-smi"
            except Exception:  # noqa: BLE001
                self.source = None

    def describe(self):
        d = {"source": self.source, "energy_counter": self.energy_counter, "target_hz": self.hz}
        if self.nv is not None:
            try:
                v = self.nv.nvmlSystemGetDriverVersion()
                d["driver"] = v.decode() if isinstance(v, bytes) else str(v)
            except Exception:  # noqa: BLE001
                pass
        return d

    def _read(self):
        nv, h = self.nv, self.h
        r = {"w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0}
        for key, fn in (("temp_c", lambda: nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)),
                        ("util", lambda: nv.nvmlDeviceGetUtilizationRates(h).gpu),
                        ("clk_gr", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS)),
                        ("clk_mem", lambda: nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))):
            try:
                r[key] = fn()
            except Exception:  # noqa: BLE001
                r[key] = None
        return r

    def snapshot(self):
        """One instantaneous reading (used between timed arms; never during a timed window)."""
        if self.nv is None:
            return {}
        try:
            r = self._read()
            return {"w_now": r["w"], "temp_c": r["temp_c"], "util": r["util"], "clk_gr": r["clk_gr"],
                    "clk_mem": r["clk_mem"]}
        except Exception:  # noqa: BLE001
            return {}

    def window(self):
        return _Window(self)


def _die_with_parent():
    """The sampler child must not outlive this process (Linux: parent-death signal)."""
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass


class _Window:
    def __init__(self, p):
        self.p = p
        self.samples = []
        self.result = {}

    def __enter__(self):
        p = self.p
        self.stop = threading.Event()
        self.proc = None
        self.e0 = None
        if p.source == "nvml":
            if p.energy_counter:
                self.e0 = p.nv.nvmlDeviceGetTotalEnergyConsumption(p.h)

            def loop():
                dt = 1.0 / p.hz
                while not self.stop.is_set():
                    try:
                        r = p._read()
                        r["t"] = time.perf_counter()
                        self.samples.append(r)
                    except Exception:  # noqa: BLE001
                        pass
                    self.stop.wait(dt)
            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        elif p.source == "nvidia-smi":
            q = "power.draw,temperature.gpu,utilization.gpu,clocks.current.graphics,clocks.current.memory"
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits", "-lms",
                                          str(int(1000 / p.hz)), "-i", str(p.index)], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, preexec_fn=_die_with_parent)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        p = self.p
        t1 = time.perf_counter()
        wall = t1 - self.t0
        res = {"source": p.source, "wall_s": round(wall, 4)}
        if p.source == "nvml":
            e1 = p.nv.nvmlDeviceGetTotalEnergyConsumption(p.h) if p.energy_counter else None
            self.stop.set()
            self.thread.join(timeout=2)
            if e1 is not None and self.e0 is not None:
                res["energy_j_counter"] = round((e1 - self.e0) / 1000.0, 4)
        elif p.source == "nvidia-smi":
            self.proc.terminate()
            try:
                txt = self.proc.communicate(timeout=5)[0]
            except Exception:  # noqa: BLE001
                txt = ""
            for ln in txt.splitlines():
                f = [x.strip() for x in ln.split(",")]
                try:
                    num = [float(x) if x not in ("", "N/A", "[N/A]") else None for x in f]
                    self.samples.append({"w": num[0], "temp_c": num[1], "util": num[2], "clk_gr": num[3],
                                         "clk_mem": num[4], "t": None})
                except (ValueError, IndexError):
                    pass
        s = [x for x in self.samples if x.get("w") is not None]
        res["n_samples"] = len(s)
        if s:
            ws = [x["w"] for x in s]
            ts = [x["t"] for x in s]
            if len(s) > 1 and all(t is not None for t in ts) and ts[-1] > ts[0]:
                area = sum((ws[i] + ws[i + 1]) / 2 * (ts[i + 1] - ts[i]) for i in range(len(s) - 1))
                mean = area / (ts[-1] - ts[0])
            else:
                mean = sum(ws) / len(ws)
            res.update(watts_mean=round(mean, 3), watts_min=round(min(ws), 3), watts_max=round(max(ws), 3),
                       watts_distinct=len(set(ws)), sample_hz=round(len(s) / wall, 2) if wall > 0 else None,
                       energy_j_integrated=round(mean * wall, 4))
            for k in ("temp_c", "util", "clk_gr", "clk_mem"):
                v = [x[k] for x in s if x.get(k) is not None]
                if v:
                    res[k + "_mean"] = round(sum(v) / len(v), 2)
                    res[k + "_range"] = [min(v), max(v)]
        self.result = res
        return False


# ------------------------------------------------------------------------------------------------- arms and cells
class Arm:
    """One arm of a cell.
    run()      zero-argument callable doing `calls` kernel invocations over the rotated buffers; returns the list of
               output tensors of its LAST invocation (the tensors a graph replay writes into).
    role       new | vendor | vendor_alt | control | separator | info
    ref        list of fp64 tensors matching run()'s outputs (relative L2 is checked against `limit`), or None
    exact      list of tensors that run()'s outputs must equal bit for bit (zeros of either sign count as equal)
    stateful   the kernel updates its own input state: repeat / graph equality is not checked here (the target
               script checks bit-identity on fresh state itself)
    code       objects for code_fingerprint()"""

    def __init__(self, name, role, run, calls, ref=None, limit=None, exact=None, stateful=False, code=(), note=None,
                 energy=True):
        self.name, self.role, self.run, self.calls = name, role, run, int(calls)
        self.ref, self.limit, self.exact, self.stateful = ref, limit, exact, stateful
        self.code, self.note, self.energy = list(code), note, energy


class Cell:
    """reps / warm override the command-line values for slow cells (recorded in every time record)."""

    def __init__(self, cid, params, mib, est_s, energy=False, reps=None, warm=None):
        self.id, self.params, self.mib, self.est_s, self.energy = cid, params, float(mib), float(est_s), energy
        self.reps, self.warm = reps, warm


def base_parser(desc, mem_budget_mib):
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="JSONL file of this block; must not exist yet")
    p.add_argument("--reverse", action="store_true", help="second job: reversed arm order")
    p.add_argument("--rot", type=int, default=0, help="extra rotation of the arm order")
    p.add_argument("--cycles", type=int, default=9)
    p.add_argument("--warm", type=int, default=10, help="untimed replays before each timed window")
    p.add_argument("--reps", type=int, default=50, help="timed replays per arm per cycle")
    p.add_argument("--seed", type=int, default=988)
    p.add_argument("--ws-mib", type=int, default=513,
                   help="rotated working set per arm and format (the L2 cache is 128 MiB)")
    p.add_argument("--mem-budget-mib", type=int, default=mem_budget_mib,
                   help="GPU memory this process may use; cells above it are skipped and recorded")
    p.add_argument("--reserve-mib", type=int, default=RESERVE_MIB)
    p.add_argument("--time-budget-s", type=float, default=540.0,
                   help="no new cell starts if it is estimated to end after this many seconds")
    p.add_argument("--hard-limit-s", type=float, default=690.0,
                   help="the process exits by itself after this many seconds, even if it hangs in a library call")
    p.add_argument("--energy-seconds", type=float, default=5.0)
    p.add_argument("--idle-seconds", type=float, default=5.0)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--resume", nargs="*", default=[], help="earlier block files of the same job (same direction)")
    p.add_argument("--cells", help="comma-separated cell ids to restrict the run to")
    p.add_argument("--pkg-dir", help="directory put first on sys.path (staged kernel package)")
    p.add_argument("--warm-only", action="store_true",
                   help="build every cell and run each arm once (fills the kernel cache); no checks, no timing")
    p.add_argument("--dry-run", action="store_true", help="parse arguments, print the cell plan, import no torch")
    return p


def est_seconds(a, arms, setup_s, energy, n_energy, reps=None, warm=None):
    """Rough wall time of one cell. arms: (microseconds per call, calls per replay) per arm, control included.
    An estimate only: the first real run of each script shows the true cost."""
    if a.warm_only:
        return setup_s + 1.0
    reps = a.reps if reps is None else reps
    warm = a.warm if warm is None else warm
    t = setup_s + sum(a.cycles * (warm + reps) * calls * us * 1e-6 for us, calls in arms)
    t += sum(2 * (5 + warm) * calls * us * 1e-6 for us, calls in arms)
    if energy and not a.no_energy:
        t += n_energy * (a.energy_seconds + 0.6)
    return t


class Race:
    def __init__(self, args, target, fixed_mib, description):
        if args.cycles < 9 and not args.dry_run:
            raise SystemExit("at least 9 cycles are required")
        self.a, self.target, self.fixed_mib, self.description = args, target, fixed_mib, description
        self.t0 = time.monotonic()
        self.f = None
        self.torch = None
        self.power = None
        self.done_cells, self.failed_cells, self.left_cells = [], [], []
        self.contaminated = []
        self.prefix = []

    # ---------------------------------------------------------------------------------------------- plan
    def plan(self, cells):
        """Apply --cells, --resume and the memory budget. Dry run: print the plan and return None."""
        a = self.a
        want = set(a.cells.split(",")) if a.cells else None
        finished = set()
        for fn in a.resume:
            try:
                with open(fn) as f:
                    rows = [json.loads(ln) for ln in f if ln.startswith("{")]
            except OSError:
                continue
            head = next((r for r in rows if r.get("t") == "header"), {})
            if head.get("target") != self.target or bool(head.get("reverse")) != bool(a.reverse):
                continue
            if bool(head.get("warm_only")) != bool(a.warm_only):
                continue
            finished |= {r["cell"] for r in rows if r.get("t") in ("cell_done", "warm_done")}
        todo, skipped = [], []
        for c in cells:
            if want is not None and c.id not in want:
                continue
            if c.id in finished:
                skipped.append((c, "done_in_resume_file"))
            elif c.mib + self.fixed_mib > a.mem_budget_mib:
                skipped.append((c, "over_memory_budget"))
            else:
                todo.append(c)
        need = int(max([c.mib for c in todo] or [0]) + self.fixed_mib + a.reserve_mib)
        self.need_free_mib = need
        self.skipped = skipped
        if a.dry_run:
            t = 2 * a.idle_seconds + 20
            fit = 0
            print(f"# {self.target} {'reverse' if a.reverse else 'forward'}: {self.description}")
            print(f"# cycles {a.cycles}, warm {a.warm}, reps {a.reps}, ws {a.ws_mib} MiB, budget {a.mem_budget_mib} MiB,"
                  f" energy {'off' if a.no_energy else str(a.energy_seconds) + ' s per arm'}")
            for c in todo:
                t += c.est_s
                ok = t <= a.time_budget_s
                fit += ok
                print(f"  {c.id:<34s} {c.mib:8.0f} MiB  est {c.est_s:6.1f} s  cum {t:7.1f} s  "
                      f"{'E' if c.energy and not a.no_energy else '-'}  {'this block' if ok else 'later block'}")
            for c, why in skipped:
                print(f"  {c.id:<34s} {c.mib:8.0f} MiB  SKIPPED {why}")
            print(f"PLANNED_CELLS={len(todo)} FIT_THIS_BLOCK={fit} SKIPPED={len(skipped)} EST_SECONDS={t:.0f}")
            print(f"REQUIRED_FREE_MIB={need}")
            return None
        return todo

    # ---------------------------------------------------------------------------------------------- output
    def emit(self, **rec):
        rec = clean(rec)
        line = json.dumps(rec, allow_nan=False, default=str)
        print(line, flush=True)
        if self.f is not None:
            self.f.write(line + "\n")
            self.f.flush()

    def start(self, extra=None):
        """Open the output, check the co-resident server, record versions and idle power. Returns torch."""
        a = self.a
        if not a.out:
            raise SystemExit("--out is required")
        asleep = server_sleeping()
        if asleep is True and os.environ.get("P1_ALLOW_SLEEPING") != "1":
            # P1_ALLOW_SLEEPING=1 allows a co-resident server that reports itself asleep; the header records
            # server_sleeping, and the request counters still mark contamination.
            raise SystemExit("REFUSED: the co-resident server reports is_sleeping true")
        import faulthandler
        faulthandler.dump_traceback_later(max(30.0, a.hard_limit_s - (time.monotonic() - self.t0)), exit=True)
        if a.pkg_dir:
            sys.path.insert(0, a.pkg_dir)
        import torch
        self.torch = torch
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.f = open(a.out, "x")
        self.power = Power()
        vers = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
        try:
            from importlib import metadata
            for name in ("vllm", "flashinfer-python", "triton", "tilelang", "nvidia-ml-py", "pynvml", "safetensors",
                         "numpy"):
                try:
                    vers[name] = metadata.version(name)
                except Exception:  # noqa: BLE001
                    vers[name] = None
        except Exception:  # noqa: BLE001
            pass
        self.counters0 = server_counters()
        self.emit(t="header", schema=SCHEMA, target=self.target, description=self.description,
                  reverse=bool(a.reverse), warm_only=bool(a.warm_only), rot=a.rot, cycles=a.cycles, warm=a.warm, reps=a.reps, seed=a.seed,
                  ws_mib=a.ws_mib, mem_budget_mib=a.mem_budget_mib, energy_seconds=None if a.no_energy else
                  a.energy_seconds, versions=vers, device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()), power=self.power.describe(),
                  graph_replay=True, capture_error_mode="thread_local", statistic="per-cycle mean of reps replays",
                  server_sleeping=asleep, server_counters=self.counters0, staged_pkg=bool(a.pkg_dir),
                  shared_gpu=True, extra=extra or {},
                  skipped=[{"cell": c.id, "why": why} for c, why in self.skipped])
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        self.idle("start")
        return torch

    def idle(self, tag):
        if self.a.no_energy or self.a.warm_only or self.power.source is None:
            return
        c0 = server_counters()
        with self.power.window() as w:
            time.sleep(self.a.idle_seconds)
        c1 = server_counters()
        self.emit(t="idle", tag=tag, server_counters_moved=counters_moved(c0, c1), **w.result)

    # ---------------------------------------------------------------------------------------------- helpers
    def capture(self, fn, tries=5):
        torch = self.torch
        fn()
        torch.cuda.synchronize()
        last = None
        for attempt in range(tries):
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    outs = fn()
                torch.cuda.synchronize()
                return g, outs, attempt
            except Exception as e:  # noqa: BLE001  (the time-slicing layer can break a capture at random)
                last = e
                torch.cuda.synchronize()
                time.sleep(0.2)
        raise RuntimeError("CUDA-graph capture failed: " + str(last)[:300])

    def bits_diff(self, x, y):
        torch = self.torch
        if x.shape != y.shape or x.dtype != y.dtype:
            return -1
        if not x.dtype.is_floating_point:
            return int((x != y).sum())
        it = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
        xi, yi = x.contiguous().view(it), y.contiguous().view(it)
        return int(((xi != yi) & ~((x == 0) & (y == 0))).sum())

    def rel_l2(self, out, ref):
        d = out.double() - ref.double()
        return float((d.norm() / ref.double().norm()).item())

    def nonfinite(self, outs):
        torch = self.torch
        return int(sum((~torch.isfinite(o)).sum().item() for o in outs if o.dtype.is_floating_point))

    def _poison(self, outs):
        for o in outs:
            o.fill_(float("nan") if o.dtype.is_floating_point else -1)

    def over_time(self, cell):
        return time.monotonic() - self.t0 + cell.est_s > self.a.time_budget_s

    # ---------------------------------------------------------------------------------------------- one cell
    def run_cell(self, cell, arms, index=0, gate=True):
        """arms: list of Arm in canonical order, without the control arm (it is added here from the `new` arm).
        gate: result of the target script's own correctness checks for this cell (stateful kernels)."""
        torch, a = self.torch, self.a
        t_cell = time.monotonic()
        if a.warm_only:
            try:
                for arm in arms:
                    arm.run()
                torch.cuda.synchronize()
                self.emit(t="warm_done", cell=cell.id, seconds=round(time.monotonic() - t_cell, 2),
                          code={x.name: code_fingerprint(x.code) for x in arms})
                self.done_cells.append(cell.id)
            except Exception as e:  # noqa: BLE001
                self.failed_cells.append(cell.id)
                self.emit(t="cell_error", cell=cell.id, stage="warm", error=type(e).__name__ + ": " + str(e)[:600])
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            return
        c0 = server_counters()
        torch.cuda.reset_peak_memory_stats()
        new = next(x for x in arms if x.role == "new")
        ctl = Arm("ctl", "control", new.run, new.calls, stateful=new.stateful, code=new.code,
                  note="same kernel objects and buffers as the new arm, second graph")
        arms = list(arms) + [ctl]
        graphs = {}
        try:
            for arm in arms:
                rec = {"t": "check", "cell": cell.id, "arm": arm.name, "role": arm.role, "calls": arm.calls,
                       "stateful": arm.stateful}
                eager = None
                if arm.role != "separator":
                    o1 = arm.run()
                    torch.cuda.synchronize()
                    eager = [o.clone() for o in o1]
                    rec["nonfinite"] = self.nonfinite(eager)
                    if not arm.stateful:
                        o2 = arm.run()
                        torch.cuda.synchronize()
                        rec["repeat_equal"] = all(self.bits_diff(x, y) == 0 for x, y in zip(eager, o2))
                    if arm.ref is not None:
                        rec["rel_l2_fp64"] = max(self.rel_l2(o, r) for o, r in zip(eager, arm.ref))
                        rec["limit"] = arm.limit
                    if arm.exact is not None:
                        rec["exact_diff"] = sum(abs(self.bits_diff(o, r)) for o, r in zip(eager, arm.exact))
                g, outs, attempts = self.capture(arm.run)
                rec["capture_retries"] = attempts
                if eager is not None and not arm.stateful:
                    ok = True
                    for _ in range(2):
                        self._poison(outs)
                        g.replay()
                        torch.cuda.synchronize()
                        ok = ok and all(self.bits_diff(x, y) == 0 for x, y in zip(eager, outs))
                    rec["graph_equal"] = ok
                rec["code"] = code_fingerprint(arm.code)
                bad = (rec.get("nonfinite", 0) != 0 or rec.get("repeat_equal") is False
                       or rec.get("graph_equal") is False or rec.get("exact_diff", 0) != 0
                       or (arm.limit is not None and "rel_l2_fp64" in rec and not rec["rel_l2_fp64"] <= arm.limit))
                rec["pass"] = not bad
                if arm.note:
                    rec["note"] = arm.note
                if arm.role in ("new", "control") and bad:
                    gate = False
                self.emit(**rec)
                graphs[arm.name] = (g, outs, arm)
                del eager
            # ---- timing pass: no sampler thread exists here
            names = [x.name for x in arms]
            if a.reverse:
                names = names[::-1]
            n = len(names)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            med = {}
            reps = a.reps if cell.reps is None else cell.reps
            warm = a.warm if cell.warm is None else cell.warm
            for cyc in range(a.cycles):
                for pos in range(n):
                    nm = names[(pos + cyc + a.rot) % n]
                    g, _, arm = graphs[nm]
                    for _ in range(warm):
                        g.replay()
                    torch.cuda.synchronize()
                    e0.record()
                    for _ in range(reps):
                        g.replay()
                    e1.record()
                    e1.synchronize()
                    us = e0.elapsed_time(e1) * 1000.0 / (reps * arm.calls)
                    med.setdefault(nm, []).append(us)
                    self.emit(t="time", cell=cell.id, cycle=cyc, pos=pos, arm=nm, role=arm.role, us=round(us, 4),
                              calls=arm.calls, reps=reps, warm=warm, replay=True, **self.power.snapshot())
            # ---- energy pass: tight loop per arm with the background sampler
            if cell.energy and not a.no_energy and self.power.source is not None:
                en = [x for x in names if graphs[x][2].role != "separator" and graphs[x][2].energy]
                k = (index + a.rot) % len(en)
                for pos, nm in enumerate(en[k:] + en[:k]):
                    g, _, arm = graphs[nm]
                    per_replay_us = statistics.median(med[nm]) * arm.calls
                    batch = max(1, min(2000, int(20000.0 / max(per_replay_us, 1.0))))
                    for _ in range(warm):
                        g.replay()
                    torch.cuda.synchronize()
                    s0 = server_counters()
                    replays = 0
                    with self.power.window() as w:
                        t0 = time.perf_counter()
                        while time.perf_counter() - t0 < a.energy_seconds:
                            for _ in range(batch):
                                g.replay()
                            torch.cuda.synchronize()
                            replays += batch
                    s1 = server_counters()
                    self.emit(t="energy", cell=cell.id, pos=pos, arm=nm, role=arm.role, replays=replays,
                              calls_per_replay=arm.calls, calls=replays * arm.calls, batch=batch,
                              server_counters_moved=counters_moved(s0, s1), **w.result)
            # ---- after timing: outputs still finite?
            post = {nm: self.nonfinite(outs) for nm, (g, outs, arm) in graphs.items() if arm.role != "separator"}
            c1 = server_counters()
            moved = counters_moved(c0, c1)
            if moved:
                self.contaminated.append(cell.id)
            self.emit(t="cell_done", cell=cell.id, params=cell.params, gate_pass=gate, nonfinite_after=post,
                      server_counters_moved=moved, planned_mib=round(cell.mib, 1),
                      peak_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                      seconds=round(time.monotonic() - t_cell, 2), est_seconds=cell.est_s,
                      elapsed=round(time.monotonic() - self.t0, 1))
            self.done_cells.append(cell.id)
        except Exception as e:  # noqa: BLE001
            self.failed_cells.append(cell.id)
            self.emit(t="cell_error", cell=cell.id, error=type(e).__name__ + ": " + str(e)[:600])
        finally:
            graphs.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def cell_failed(self, cell, e):
        self.failed_cells.append(cell.id)
        self.emit(t="cell_error", cell=cell.id, stage="setup", error=type(e).__name__ + ": " + str(e)[:600])
        gc.collect()
        self.torch.cuda.empty_cache()

    def finish(self, cells):
        left = [c.id for c in cells if c.id not in self.done_cells and c.id not in self.failed_cells]
        self.idle("end")
        c1 = server_counters()
        self.emit(t="done", complete=not left and not self.failed_cells, cells_done=len(self.done_cells),
                  cells_failed=self.failed_cells, cells_left=left, contaminated_cells=self.contaminated,
                  server_counters=c1, server_counters_moved=counters_moved(self.counters0, c1),
                  seconds=round(time.monotonic() - self.t0, 1))
        self.f.close()
        print("P1_BLOCK_DONE left=%d failed=%d" % (len(left), len(self.failed_cells)), flush=True)
        return 0 if not self.failed_cells else 4


def copies_for(ws_mib, bytes_per_copy, minimum=2):
    """Number of rotated copies so that the working set exceeds ws_mib."""
    return max(minimum, int(ws_mib * 2**20 // max(1, bytes_per_copy)) + 1)
