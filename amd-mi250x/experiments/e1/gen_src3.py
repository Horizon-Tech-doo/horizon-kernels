import os,sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch,tilelang
sys.path.insert(0,'.')
from gemm_prefetch import gemm
from bench_gemm import matmul
for M,sn in [(16384,'16k')]:
    for vn,(bM,bK,v) in {'b12864':(128,64,'base'),'b25632':(256,32,'base'),'r25632':(256,32,'regstage'),'r12864':(128,64,'regstage')}.items():
        open(f'src/{vn}_{sn}.cu','w').write(gemm(M,M,M,bM,128,bK,256,2,v).get_kernel_source())
for M,sn in [(8192,'8k'),(16384,'16k')]:
    open(f'src/b25632sw8_{sn}.cu','w').write(matmul(M,M,M,256,128,32,'float16',256,0,2,8).get_kernel_source())
