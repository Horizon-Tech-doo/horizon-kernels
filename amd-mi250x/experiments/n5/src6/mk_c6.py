"""Writes c6.py from C2P's c3.py with the additions this round needs. c3.py itself is not edited. Each replacement must hit exactly once."""
import sys
s = open(sys.argv[1]).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep('from n5lib import CASES, inputs, ref\nimport tilelang\n',
    'from n5lib import CASES, inputs, ref\nfrom n5lib_r5 import extend\nextend(CASES)\nimport tilelang\n'
    'from tilelang.contrib import hipcc\n_orig = hipcc.compile_hip\nCOMPILES = [0]\n\n\ndef _cnt(code, *x, **k):\n    COMPILES[0] += 1\n    return _orig(code, *x, **k)\n\n\nhipcc.compile_hip = _cnt\n')
rep('t0 = time.time(); r = dict(name=c["name"])\n', 't0 = time.time(); n0 = COMPILES[0]; r = dict(name=c["name"])\n')
rep('        r["t"] = round(time.time() - t0, 1)\n', '        r["compiles"] = COMPILES[0] - n0\n        r["t"] = round(time.time() - t0, 1)\n')
rep('land=a.land)), flush=True)\nrot =',
    'land=a.land, gcd=dict(rocr=os.environ.get("ROCR_VISIBLE_DEVICES"), ordinal=os.environ.get("GPU_DEVICE_ORDINAL"), uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "")), name=torch.cuda.get_device_name(0)))), flush=True)\nrot =')
rep('print(json.dumps(dict(base, name=case + "_ck", impl="ck", ms=round(ms, 4)', 'print(json.dumps(dict(base, name=case + "_ck", impl="ck", slot="pre", ms=round(ms, 4)')
rep('                k = build(c)\n                O = k(Q, K, V)\n', '                n0 = COMPILES[0]\n                k = build(c)\n                r["compiles"] = COMPILES[0] - n0\n                O = k(Q, K, V)\n')
rep('        if os.environ.get("C3_SDPA_END"):',
    '        if os.environ.get("C3_STACK", "image") == "image":   # CK after the arms, same process\n'
    '            ms = bench(lambda: flash_attn_func(Q, K, V, causal=causal), warmup=10, rep=rep)\n'
    '            print(json.dumps(dict(base, name=case + "_ck", impl="ck", slot="post", ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)\n'
    '        if os.environ.get("C3_SDPA_END"):')
open(sys.argv[2], "w").write(s)
