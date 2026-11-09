# Fills the private TileLang cache on the login node, same call as run_sweep.
import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.argv=['x']
import gemm_prefetch as g
for M,N,K in [(8192,8192,8192),(16384,16384,16384),(4096,4096,4096),(4096,17408,5120),(32768,1024,2048)]:
    for bM,bN,bK,th,pad in g.SWEEP_TILES:
        if M%bM or N%bN or K%bK: continue
        for sw,o in g.SWEEP_SWIZZLE:
            g.gemm(M,N,K,bM,bN,bK,th,2,"base",sw,o,pad)
    print(M,N,K,flush=True)
