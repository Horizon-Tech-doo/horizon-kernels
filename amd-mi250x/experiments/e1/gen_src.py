# Generates TileLang sources (private cache) for the E1 GEMM variants; compile happens in tilelang.jit.
# Output: src/<base>_<shape>.cu raw device_kernel.cu text
import json,os,sys,time,glob,traceback
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
import torch, tilelang
sys.path.insert(0,'.')
from gemm_prefetch import gemm
shapes={'4k':(4096,4096,4096),'8k':(8192,8192,8192),'32k':(32768,1024,2048),'17k':(4096,17408,5120)}
vars_={'b12864':(128,128,64,'base'),'b25632':(256,128,32,'base'),'r12864':(128,128,64,'regstage'),'r25632':(256,128,32,'regstage')}
os.makedirs('src',exist_ok=True)
for sn,(M,N,K) in shapes.items():
  for vn,(bM,bN,bK,v) in vars_.items():
    t0=time.time()
    k=gemm(M,N,K,bM,bN,bK,256,2,v)
    s=k.get_kernel_source()
    open(f'src/{vn}_{sn}.cu','w').write(s)
    print(vn,sn,len(s),round(time.time()-t0,1),flush=True)
