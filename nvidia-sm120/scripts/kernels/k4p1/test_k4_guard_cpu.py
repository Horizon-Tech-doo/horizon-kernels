# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# CPU-only test of the K4 timeout guard's host side: Watchdog trips once on a set flag, k4seg._trip writes the marker
# and exits (os._exit stubbed), k4seg.install() refuses to enable K4 while the marker exists.
import os, sys, tempfile, time, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.cuda.is_available = lambda: False
from tilelang.contrib.horizon_k4 import runtime as rt
sys.path.insert(0, sys.argv[1])                     # dir that holds the horizon_kernels package under test
import horizon_kernels.k4seg as ks


class Fake:
    mapped = True
    def __init__(self): self.v = 0
    def get(self): return self.v


tripped = []
wd = rt.Watchdog(lambda n: tripped.append(n), period_s=0.01)
f1, f2 = Fake(), Fake()
wd.add("a", f1); wd.add("b", f2)
time.sleep(0.05)
assert not tripped, tripped
f2.v = 1
time.sleep(0.1)
assert tripped == ["b"], tripped
print("watchdog: trips once on the set flag: OK")

d = tempfile.mkdtemp()
os.environ["HORIZON_K4_DISABLE_MARKER"] = os.path.join(d, "k4_disabled")
logs = []
ks._log = lambda: types.SimpleNamespace(critical=lambda *a: logs.append(a), warning=lambda *a: logs.append(a),
                                        info=lambda *a: None, error=lambda *a: logs.append(a))
codes = []
ks.os._exit = lambda c: codes.append(c)
ks._trip("test")
assert codes == [70] and os.path.exists(ks._marker()) and "timed out" in open(ks._marker()).read(), (codes, logs)
print("trip: marker written, engine exit(70): OK")

# per-step check: a wrapped get_output returns normally while the flags are clear, raises once one is set
os.remove(ks._marker())


class Out:
    def get_output(self):
        return "tokens"


ks._wrap_get_output(Out, "Out.get_output")
g1, g2 = Fake(), Fake()
g1.dptr, g2.dptr = 1, 2
ks._STATE["flags"].update({"1": g1, "2": g2})
assert Out().get_output() == "tokens"
g2.v = 1
try:
    Out().get_output()
    raise AssertionError("step check did not raise")
except ks.K4TimeoutError:
    pass
assert os.path.exists(ks._marker())
ks._STATE["flags"].clear()
print("step check: tokens pass while clear, step raises (and marker written) when set: OK")

os.environ["HORIZON_K4_SEGMENT"] = "1"
called = []
ks.install.__globals__["torch"] = torch
import importlib
before = len(logs)
ks.install()
assert len(logs) == before + 1 and "stays OFF" in logs[-1][0], logs[-1:]
print("install: refuses while the marker exists: OK")
print("K4_GUARD_CPU_OK")
