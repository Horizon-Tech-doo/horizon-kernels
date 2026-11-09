"""Wheel process, CPU. Compiles fork-main's generated source for the J arms under the wheel's own templates and flags, with no TileLang change: does the port need more than the generated source? Writes bytes3/<name>_frozen.hsaco. Instruction streams compared by asmcmp2.py."""
import hashlib, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tilelang
from tilelang.contrib import hipcc
D = sys.argv[1]; out = sys.argv[2]; os.makedirs(out, exist_ok=True)
opts = sys.argv[3].split("|")
for n in sys.argv[4:]:
    r = dict(name=n, ver=tilelang.__version__)
    try:
        b = hipcc.compile_hip(open(f"{D}/{n}.cc").read(), target_format="hsaco", arch="gfx90a", options=opts)
        open(f"{out}/{n}_frozen.hsaco", "wb").write(bytes(b)); r["md5"] = hashlib.md5(bytes(b)).hexdigest()
    except Exception as e:  # noqa: BLE001
        r["err"] = str(e)[-1200:]
    print(json.dumps(r), flush=True)
