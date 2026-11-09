"""Track 5: RCCL inside the vLLM image, before any model load. Short all-reduce and all-to-all, one task per GCD.
Tells a transport failure from a vLLM failure and gives the bandwidth the image's RCCL reaches."""
import json
import os
import time

import torch
import torch.distributed as dist

rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
t0 = time.time()
dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", 0))
torch.cuda.set_device(0)
if rank == 0:
    print(json.dumps({"comm_init_s": round(time.time() - t0, 1), "world": world, "nodes": int(os.environ.get("SLURM_NNODES", 1)), "torch": torch.__version__, "hip": torch.version.hip}), flush=True)
for mb in (1, 16, 256):
    n = mb * 2 ** 20 // 2
    x = torch.ones(n, device="cuda", dtype=torch.float16)
    a2a = torch.ones(n - n % world, device="cuda", dtype=torch.float16)
    rx = torch.empty_like(a2a)
    for name, fn in (("all_reduce", lambda: dist.all_reduce(x)), ("all_to_all", lambda: dist.all_to_all_single(rx, a2a))):
        ts = []
        for i in range(12):
            dist.barrier()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        ms = sorted(ts[2:])[5] * 1e3
        if rank == 0:
            print(json.dumps({"op": name, "mb_per_rank": mb, "median_ms": round(ms, 3), "gb_s_total": round(world * mb / 1024 / (ms / 1e3), 2)}), flush=True)
    # data check, added for the OFI adapter arms: chunk j from rank r holds r*10+j+1, so a wrong or stale byte shows as a mismatch
    c = a2a.numel() // world
    src = torch.cat([torch.full((c,), float(rank * 10 + j + 1), device="cuda", dtype=torch.float16) for j in range(world)])
    want = torch.cat([torch.full((c,), float(s * 10 + rank + 1), device="cuda", dtype=torch.float16) for s in range(world)])
    dist.all_to_all_single(rx, src)
    y = torch.ones(n, device="cuda", dtype=torch.float16)
    dist.all_reduce(y)
    bad = torch.tensor([(rx != want).sum().item(), (y != world).sum().item()], device="cuda", dtype=torch.int64)
    dist.all_reduce(bad)
    if rank == 0:
        print(json.dumps({"check": "data", "mb_per_rank": mb, "a2a_bad": int(bad[0]), "reduce_bad": int(bad[1])}), flush=True)
dist.destroy_process_group()
