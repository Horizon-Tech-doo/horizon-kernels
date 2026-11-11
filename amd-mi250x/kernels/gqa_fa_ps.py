"""gqa_fa.py with the attention weights scaled by a power of two before the fp16 cast (track P1, 11 Nov 2026). gqa_fa.py is not edited: this file imports a private copy of it and adds one edit to
the device source that copy compiles.

Why. The fp16 MFMA of gfx90a returns zero for subnormal fp16 operands (evidence/P1/RESULTS.md, job 23627461). The kernel casts the unnormalised weights exp(s - rowmax), which lie in (0, 1], to fp16
for the P x V product, so every weight below 6.1e-5 of its row's maximum is dropped. On the served 27B's long prompts those weights carry 0.2 to 0.6% of a row's attention (up to 8% in single rows) and
the output error against float64 rises from bf16 rounding (1.7e-3) to 3e-3 .. 1.1e-2.

The change. In the generated source the exponent s * scale - rowmax * scale gets + B ln 2 (B = GQA_PS_BITS, default 10): every weight is multiplied by 2^B. The row sum l_i is accumulated from the
same scaled weights and the output is acc_o / l_i, so the factor cancels exactly; the rescaling between K tiles (sf) is a ratio and does not change. Scaled weights are at most 2^B = 1024 (fp16 holds
65504) and the drop threshold moves to 6.1e-5 / 2^B = 6e-8 of the row maximum. The Python kernel and gqa_fa's own source edits (_fast_edit, _vt_edit) are untouched: the + B ln 2 is applied to
their output, on the four lines that form the exponent, and raises if those lines are not found exactly once each. Compiled kernels carry their own cache flag.

gqa_fa installs one compile hook per process and skips installing if one is there. load() stacks the copy's hook on top of whatever is installed, so the copy works next to an already loaded gqa_fa;
a gqa_fa loaded AFTER this module would find a hook in place and compile without its edits: load gqa_fa first, or use only one of the two in a process (the serving plugin does).

Use: m = gqa_fa_ps.load(); m.gqa_flash_fast(...), m.d256_tile(...) as in gqa_fa."""
import importlib.util, math, os, re, tempfile

_m = None
_EXPO = re.compile(r"(__\d+\.([xyzw]) = )\((__\d+\.\2-v__\d+\.\2)\);")


def shift(code, bits):
    """+ bits * ln 2 on the exponent of the weights: the four component lines '__a.c = (__b.c-v__d.c);' that feed __expf."""
    hits = _EXPO.findall(code)
    if len(hits) != 4 or sorted(h[1] for h in hits) != ["w", "x", "y", "z"] or "__expf(" not in code:
        raise RuntimeError(f"gqa_fa_ps: exponent lines not found as expected ({len(hits)})")
    c = f"{bits * math.log(2.0):.9e}f"
    return _EXPO.sub(lambda m: f"{m.group(1)}(({m.group(3)}) + {c});", code)


def load(path=None):
    global _m
    if _m is None:
        p = path or os.environ.get("GQA_FA") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa_fa.py")
        bits = int(os.environ.get("GQA_PS_BITS", "10"))
        src = open(p).read()
        if src.count("-DGQA_FAST_N5_V2") != 3 or "def _fast_edit(s, c)" not in src or "def _vt_edit(s, c)" not in src:
            raise RuntimeError("gqa_fa_ps: gqa_fa.py is not the source this file was written for")
        src = src.replace("-DGQA_FAST_N5_V2", f"-DGQA_FAST_N5_V2_PSB{bits}")
        d = tempfile.mkdtemp(prefix="gqa_fa_ps_")          # TileLang reads a kernel's Python source back from its file
        f = os.path.join(d, f"gqa_fa_psb{bits}_impl.py")
        with open(f, "w") as o:
            o.write(src)
        spec = importlib.util.spec_from_file_location(f"gqa_fa_psb{bits}_impl", f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fe, ve = m._fast_edit, m._vt_edit
        m._fast_edit = lambda s, c: fe(s, c) if c.get("vt") else shift(fe(s, c), bits)     # with vt the vt edit runs after this one and gets the shift
        m._vt_edit = lambda s, c: shift(ve(s, c), bits)
        from tilelang.contrib import hipcc
        cur = hipcc.compile_hip
        hipcc.compile_hip = lambda code, *a, **k: cur(code, *a, **k)                         # no marker: the copy installs its own hook on top
        m._install_hook()
        m.PS_BITS = bits
        _m = m
    return _m
