"""Build FusedMoE(T) defaults for many T into one shared kernel cache, 12 forked workers (each T's keys are its own). Prints build seconds."""
import json, multiprocessing as mp, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe_fused as M
def one(Tn):
    t0 = time.time()
    try:
        M.FusedMoE(Tn)
        return dict(kind="prebuilt", T=Tn, build_s=round(time.time() - t0, 1))
    except Exception as ex:
        return dict(kind="prebuild_failed", T=Tn, err=str(ex)[-300:])
if __name__ == "__main__":
    with mp.get_context("fork").Pool(12) as p:
        for r in p.imap_unordered(one, [int(x) for x in sys.argv[1].split(",")]):
            print(json.dumps(r), flush=True)
