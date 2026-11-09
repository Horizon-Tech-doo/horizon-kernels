# one short pass per shape under AMD_LOG_LEVEL=3 to capture the kernel name and grid; no timing.
import torch,sys
torch.manual_seed(0)
for M,N,K in [(4096,4096,4096),(8192,8192,8192),(32768,1024,2048),(4096,17408,5120)]:
    print(f"===SHAPE {M} {N} {K}",flush=True); sys.stderr.write(f"===SHAPE {M} {N} {K}\n"); sys.stderr.flush()
    a=torch.randn(M,K,device='cuda',dtype=torch.float16); b=torch.randn(N,K,device='cuda',dtype=torch.float16)
    torch.backends.cuda.preferred_blas_library('cublaslt')
    for _ in range(3): c=a@b.T
    torch.cuda.synchronize(); del a,b,c
