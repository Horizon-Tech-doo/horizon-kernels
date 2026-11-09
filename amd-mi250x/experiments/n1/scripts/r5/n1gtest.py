import subprocess, sys
_co = subprocess.check_output
subprocess.check_output = lambda c, *a, **k: b"Name: gfx90a\n" if isinstance(c, list) and str(c[0]).endswith("rocminfo") else _co(c, *a, **k)
sys.path.insert(0, sys.argv[1])
import moe_fused as M, n1grid as G
k = M._build(G.align_count(136, 256, 64, 1024)); k2 = M._build(G.align_slots(136, 256, 64, 256)); print("grid align builds", flush=True)
