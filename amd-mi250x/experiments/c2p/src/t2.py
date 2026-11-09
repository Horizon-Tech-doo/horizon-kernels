import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
sys.path.insert(0, os.environ["F"] + "/testing/python/amd")
import test_tilelang_hip_reducer_scale as t
for a in [(64,64,3,64),(128,64,3,256),(64,64,3,256)]:
    try:
        s = t._scaled_sum(*a).get_kernel_source()
        print(a, "ok", "workspace" in s, s.count("AllReduce"), [l.strip()[:90] for l in s.split("\n") if "acc" in l and ("*" in l or "AllReduce" in l) and "acc[" not in l][:3])
        print([l.strip()[:100] for l in s.split("\n") if "AllReduce" in l or " acc" in l][:6])
    except Exception as e:
        print(a, "ERR", str(e)[-300:])
try:
    t._scaled_sum(64,64,2,64,one_thread_per_row=True).get_kernel_source(); print("no error")
except Exception as e:
    print("expected", str(e)[-200:])
