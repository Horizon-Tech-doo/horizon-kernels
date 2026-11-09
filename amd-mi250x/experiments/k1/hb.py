# hipBLASLt through torch, on the same fp16 data as the harness, and the distance to the TileLang-style result.
import sys, time, json, numpy as np, torch
d=sys.argv[1]; rnd=int(sys.argv[2])
for M,tag in [(4096,'4'),(8192,'8')]:
    an=np.fromfile(f'{d}/A{tag}.bin',dtype=np.float16).reshape(M,M); bn=np.fromfile(f'{d}/B{tag}.bin',dtype=np.float16).reshape(M,M)
    a=torch.from_numpy(an.copy()).cuda(); b=torch.from_numpy(bn.copy()).cuda()
    torch.backends.cuda.preferred_blas_library('cublaslt')
    t=b.T
    for _ in range(5): c=a@t
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(40): c=a@t
    torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/40*1e3
    ours=torch.from_numpy(np.fromfile(f'{d}/C{tag}.bin',dtype=np.float16).reshape(M,M).copy()).cuda()
    diff=(c.float()-ours.float())
    rel=(diff.norm()/ours.float().norm()).item(); mx=diff.abs().max().item()
    if rnd==0:
        ref=a.float()@t.float(); rel2=((ours.float()-ref).norm()/ref.norm()).item(); relh=((c.float()-ref).norm()/ref.norm()).item()
    else: rel2=relh=None
    print(json.dumps({"t":"hipblaslt","M":M,"K":M,"round":rnd,"ms":round(ms,4),"TF":round(2.0*M*M*M/ms*1e-9,2),"rel_l2_vs_ours":rel,"max_abs_vs_ours":mx,"ours_rel_vs_fp32":rel2,"hb_rel_vs_fp32":relh,"blas":str(torch.backends.cuda.preferred_blas_library())}),flush=True)
    del a,b,t,c,ours,diff; torch.cuda.empty_cache()
