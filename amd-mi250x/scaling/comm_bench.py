"""Track 5: RCCL collectives on LUMI-G via torch.distributed (backend "nccl" = RCCL on ROCm).
all_reduce and all_to_all_single, fp16, 1 MB ... 1 GB per rank. Reports algorithm and bus bandwidth (nccl-tests
conventions: all_reduce busbw = algbw * 2(n-1)/n, all_to_all busbw = algbw * (n-1)/n). Rank 0 prints JSON lines."""
import json, os, socket, time
import torch
import torch.distributed as dist


def main():
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ["SLURM_LOCALID"])
    torch.cuda.set_device(0)                                    # each task sees one GCD (ROCR_VISIBLE_DEVICES)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", 0))
    if rank == 0:
        print(json.dumps({"world": world, "nodes": int(os.environ.get("SLURM_NNODES", 1)), "torch": torch.__version__,
                          "host": socket.gethostname()}), flush=True)
    for op in ("all_reduce", "all_to_all"):
        for mb in (1, 8, 64, 256, 1024):
            n = mb * 1024 * 1024 // 2
            n -= n % world
            x = torch.ones(n, device="cuda", dtype=torch.float16)
            y = torch.empty_like(x)
            fn = (lambda: dist.all_reduce(x)) if op == "all_reduce" else (lambda: dist.all_to_all_single(y, x))
            for _ in range(5):
                fn()
            torch.cuda.synchronize(); dist.barrier()
            iters = 20 if mb <= 256 else 10
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            t = torch.tensor([dt], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if rank == 0:
                algbw = n * 2 / t.item() / 1e9
                factor = 2 * (world - 1) / world if op == "all_reduce" else (world - 1) / world
                print(json.dumps({"op": op, "world": world, "MB_per_rank": mb, "ms": round(t.item() * 1e3, 3),
                                  "algbw_GBs": round(algbw, 1), "busbw_GBs": round(algbw * factor, 1)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
