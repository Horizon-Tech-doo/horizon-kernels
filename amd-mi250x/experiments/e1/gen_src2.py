import os,sys,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch,tilelang
sys.path.insert(0,'.')
from gemm_prefetch import gemm
from bench_gemm import matmul
os.makedirs('src',exist_ok=True)
vars_={'b12864':(128,128,64,'base'),'b25632':(256,128,32,'base'),'r12864':(128,128,64,'regstage'),'r25632':(256,128,32,'regstage')}
for N in [3200,3072,5632,5888,4224,2816,2944]:
    for vn,(bM,bN,bK,v) in vars_.items():
        k=gemm(bM*8,N,4096,bM,bN,bK,256,2,v)
        open(f'src/{vn}_n{N}.cu','w').write(k.get_kernel_source())
    print(N,flush=True)
