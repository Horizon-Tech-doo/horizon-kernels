"""Wheel process, wheel's own headers, CPU. Generated fork-main source plus only the RegVec definition from patch 0007, inserted after the includes. Writes bytes3/<name>_frozen.hsaco. No file of the wheel or fork-main is changed. Usage: freeze3.py BYTES OUT INCLUDE names..."""
import hashlib, json, os, re, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tilelang
from tilelang.contrib import hipcc
SHIM = """
namespace tl {
template <class S> struct RegVec;
template <int N> struct RegVecBase { typedef uint Vec __attribute__((ext_vector_type(N))); Vec v; };
#define TL_DEFINE_REG_VEC(S, N) template <> struct RegVec<S> : RegVecBase<N> { TL_DEVICE operator S() const { return __builtin_bit_cast(S, v); } TL_DEVICE RegVec &operator=(const S &x) { v = __builtin_bit_cast(Vec, x); return *this; } TL_DEVICE RegVec &operator=(const RegVec &x) { v = x.v; return *this; } };
TL_DEFINE_REG_VEC(uint2, 2)
TL_DEFINE_REG_VEC(uint4, 4)
#undef TL_DEFINE_REG_VEC
}
"""
D, out, inc = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
for n in sys.argv[4:]:
    s = open(f"{D}/{n}.cc").read(); i = s.index('extern "C"')
    r = dict(name=n, ver=tilelang.__version__)
    try:
        b = hipcc.compile_hip(s[:i] + SHIM + s[i:], target_format="hsaco", arch="gfx90a", options=["-std=c++17", "-I" + inc])
        open(f"{out}/{n}_frozen.hsaco", "wb").write(bytes(b)); r["md5"] = hashlib.md5(bytes(b)).hexdigest()
    except Exception as e:  # noqa: BLE001
        r["err"] = str(e)[-1000:]
    print(json.dumps(r), flush=True)
