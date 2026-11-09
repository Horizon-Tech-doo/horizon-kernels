"""Track 5: expert-parallel MoE layer across GCDs and nodes on LUMI-G, using our TileLang fused-MoE kernels locally.

Shape as moe_fused.py (H 2048, I 512, 256 experts, top-8). W ranks; rank r owns experts [r*E/W, (r+1)*E/W). Per step:
  dispatch : every (token, expert) pair's row goes to the rank owning the expert (all_to_all_single, uneven splits)
  compute  : local fused MoE on the received rows (each row = one pair, top-1 local), padded to a fixed capacity
             (capacity factor 1.3 x the average; rows beyond it would be dropped and are counted) with a dummy expert
             for the padding so the kernels keep fixed shapes (no recompiles)
  combine  : rows go back (reverse all_to_all), are weighted and summed per token
Reports per-phase time (max over ranks) and tokens per second for the whole job; W = 1 is the single-GCD baseline.

Arms, all in one process and rotated per round (EP_ARMS):
  old  : the pre-N1 forms moe_align_gpu, moe_gate_up, moe_down, as used for the first scaling numbers
  land : FusedMoE, the landed kernels, on capacity rows. Padding rows are computed against the zero dummy expert
  skip : land, with the dummy expert's blocks marked unused after align, so padding rows launch but compute nothing
At W = 1, EP_DIRECT=1 also times FusedMoE(T) directly: the landed call with no capacity padding and no row duplication layer.

EP_PRECOMPILE="1,8,16,32" builds every kernel for those W in one process and exits; run it once on one GCD, so that the
ranks of a large job load kernels from the shared cache. Rank 0 builds first in any case, then the others.
Run with srun, one task per GCD (see ep.sbatch)."""
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist

_here = os.path.dirname(os.path.abspath(__file__))
for _d in (os.environ.get("TL_KERNELS", ""), os.path.join(_here, "..", "kernels"), os.path.join(os.environ.get("TL_ROOT", ""), "scripts")):
    if _d and os.path.exists(os.path.join(_d, "moe_fused.py")):
        sys.path.insert(0, _d)
        break
import moe_fused as M  # noqa: E402

H, I, E, TOPK = 2048, 512, 256, 8
OLD = dict(bM=64, bN=64, bK=64, th=256, bN2=128, kp=2)


def build(arm, T, El):
    """Kernels for one arm. Returns compute(rows_buf, eid_buf, W13, W2, ones) -> y [cap or cap + 1, H]."""
    cap = int(1.3 * T * TOPK)
    if arm == "old":
        c = OLD
        ka = M.moe_align_gpu(cap, El + 1, c["bM"])
        P = (cap + (El + 1) * (c["bM"] - 1) + c["bM"] - 1) // c["bM"] * c["bM"]
        k1 = M.moe_gate_up(cap, 1, H, I, El + 1, P, c["bM"], c["bN"], c["bK"], c["th"], "row", c["kp"])
        k2 = M.moe_down(cap, 1, H, I, El + 1, P, c["bM"], c["bN2"], c["bK"], c["th"], "row", c["kp"])

        def compute(rows, eid, W13, W2, ones):
            s_, be_ = ka(eid)
            return k2(k1(rows, W13, s_, be_), W2, s_, be_, ones)
        return compute, {k: c[k] for k in c}
    m = M.FusedMoE(cap, 1, H, I, El + 1)

    def compute(rows, eid, W13, W2, ones):
        s_, be_ = m.align(eid)
        if arm == "skip":
            be_ = be_.masked_fill(be_ == El, -1)
        return m.k2(m.k1(rows, W13, s_, be_), W2, s_, be_, ones)
    return compute, dict(m.cfg, unroll=m.unroll)


def precompile(worlds, tokens, arms):
    for w in worlds:
        for T in tokens:
            for a in arms:
                t0 = time.perf_counter()
                build(a, T, E // w)
                print(json.dumps({"precompile": a, "world": w, "T": T, "s": round(time.perf_counter() - t0, 1)}), flush=True)
    for T in tokens:
        M.FusedMoE(T, TOPK, H, I, E)


def main():
    rank, world = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    torch.cuda.set_device(0)
    tokens = [int(x) for x in os.environ.get("EP_TOKENS", "512,2048").split(",")]
    arms = os.environ.get("EP_ARMS", "old,land,skip").split(",")
    rounds = int(os.environ.get("EP_ROUNDS", 4))
    steps = int(os.environ.get("EP_STEPS", 30))
    moe_md5 = hashlib.md5(open(M.__file__, "rb").read()).hexdigest()
    want = os.environ.get("EP_MOE_MD5", "")
    if want and moe_md5 != want:
        raise SystemExit(f"moe_fused.py md5 {moe_md5}, expected {want}")
    if os.environ.get("EP_PRECOMPILE"):
        print(json.dumps({"header": 1, "moe_md5": moe_md5, "clang": M.clang_major()}), flush=True)
        precompile([int(x) for x in os.environ["EP_PRECOMPILE"].split(",")], tokens, arms)
        if world != 1:
            raise SystemExit("precompile runs on one rank")
    if world > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", 0))
    if rank == 0:
        print(json.dumps({"header": 1, "world": world, "nodes": int(os.environ.get("SLURM_NNODES", 1)), "moe_md5": moe_md5, "clang": M.clang_major(),
                          "torch": torch.__version__, "arms": arms, "rounds": rounds, "steps": steps, "job": os.environ.get("SLURM_JOB_ID")}), flush=True)
    El = E // world
    torch.manual_seed(1234 + rank)
    W13 = (torch.randn(El + 1, 2 * I, H, device="cuda") * H ** -0.5).half()
    W2 = (torch.randn(El + 1, H, I, device="cuda") * I ** -0.5).half()
    W13[El].zero_(); W2[El].zero_()                       # dummy expert for capacity padding
    for T in tokens:
        X = torch.randn(T, H, device="cuda").half()
        wts, ids = torch.topk(torch.softmax(torch.randn(T, E, device="cuda"), -1), TOPK, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float()
        flat = ids.flatten()
        dest = flat // El
        order = torch.argsort(dest, stable=True)
        tok_of = order // TOPK
        send_rows = X[tok_of]                              # [T*topk, H] in destination order
        send_eid = (flat[order] % El).int()
        send_counts = torch.bincount(dest, minlength=world).int()
        cap = int(1.3 * T * TOPK)
        # rank 0 fills the shared kernel cache, then the others load from it
        kern, cfgs = {}, {}
        t0 = time.perf_counter()
        if rank != 0 and world > 1:
            dist.barrier()
        for a in arms:
            kern[a], cfgs[a] = build(a, T, El)
        if rank == 0 and world > 1:
            dist.barrier()
        if rank == 0:
            print(json.dumps({"T": T, "build_s": round(time.perf_counter() - t0, 1), "cfg": cfgs}), flush=True)
        ones = torch.ones(cap, device="cuda", dtype=torch.float32)
        rows_buf = torch.zeros(cap, H, device="cuda", dtype=torch.float16)
        eid_buf = torch.full((cap,), El, device="cuda", dtype=torch.int32)

        def dispatch():
            if world > 1:
                recv_counts = torch.empty_like(send_counts)
                dist.all_to_all_single(recv_counts, send_counts)
                rc, sc = recv_counts.tolist(), send_counts.tolist()
                n_in = sum(rc)
                recv_rows = torch.empty(n_in, H, device="cuda", dtype=torch.float16)
                recv_eid = torch.empty(n_in, device="cuda", dtype=torch.int32)
                dist.all_to_all_single(recv_rows, send_rows, rc, sc)
                dist.all_to_all_single(recv_eid, send_eid, rc, sc)
            else:
                rc, sc, n_in = [T * TOPK], [T * TOPK], T * TOPK
                recv_rows, recv_eid = send_rows, send_eid
            return rc, sc, n_in, recv_rows, recv_eid

        def combine(yb, rc, sc):
            if world > 1:
                back = torch.empty(T * TOPK, H, device="cuda", dtype=yb.dtype)
                dist.all_to_all_single(back, yb, sc, rc)
            else:
                back = yb
            out = torch.zeros(T, H, device="cuda", dtype=torch.float32)
            out.index_add_(0, tok_of, back.float() * wts.flatten()[order, None])
            return out, back

        def step(compute, times=None):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            ev[0].record()
            rc, sc, n_in, recv_rows, recv_eid = dispatch()
            dropped = max(0, n_in - cap)
            n_keep = min(n_in, cap)
            rows_buf[:n_keep].copy_(recv_rows[:n_keep]); rows_buf[n_keep:].zero_()
            eid_buf[:n_keep].copy_(recv_eid[:n_keep]); eid_buf[n_keep:].fill_(El)
            ev[1].record()
            y = compute(rows_buf, eid_buf, W13, W2, ones)   # one row per received pair, then padding
            ev[2].record()
            yb = torch.zeros(n_in, H, device="cuda", dtype=torch.float16)
            yb[:n_keep] = y[:n_keep]
            out, _ = combine(yb, rc, sc)
            ev[3].record()
            torch.cuda.synchronize()
            if times is not None:
                times.append((ev[0].elapsed_time(ev[1]), ev[1].elapsed_time(ev[2]), ev[2].elapsed_time(ev[3]), dropped))
            return out, y, recv_rows, recv_eid, rc, sc, n_keep

        def check(a):
            """Kernel output on this rank's rows against the fp32 reference, the whole path against the reference sent through the
            same comm, and an exact round trip of the rows. Comm carries both sides of the second check, so routing is covered by
            the third check and by construction only."""
            out, y, rr, re, rc, sc, n_keep = step(kern[a])
            R = M.reference(rr, W13, W2, re[:, None], torch.ones(rr.shape[0], 1, device="cuda"))
            rel_y = ((y[:n_keep].float() - R[:n_keep]).norm() / R[:n_keep].norm()).item()
            out_R, _ = combine(R, rc, sc)
            rel_out = ((out - out_R).norm() / out_R.norm()).item()
            _, back = combine(rr, rc, sc)
            exact = float(torch.equal(back, send_rows))
            bad = float((~torch.isfinite(out)).sum().item())
            s = torch.tensor([rel_y, rel_out, 1.0 - exact, bad], device="cuda", dtype=torch.float64)
            if world > 1:
                dist.all_reduce(s, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(json.dumps({"check": a, "world": world, "T": T, "rel_y_vs_fp32": round(s[0].item(), 7), "rel_out_vs_fp32": round(s[1].item(), 7),
                                  "round_trip_diff": int(s[2].item()), "nonfinite": int(s[3].item())}), flush=True)

        for r in range(rounds):
            seq = arms[r % len(arms):] + arms[:r % len(arms)]
            if r % 2:
                seq = seq[::-1]
            for pos, a in enumerate(seq):
                for _ in range(3):
                    step(kern[a])
                if world > 1:
                    dist.barrier()
                times = []
                t0 = time.perf_counter()
                for _ in range(steps):
                    step(kern[a], times)
                wall = (time.perf_counter() - t0) / steps * 1e3
                med = [sorted(t[i] for t in times)[len(times) // 2] for i in range(3)]
                stats = torch.tensor(med + [wall, float(max(t[3] for t in times))], device="cuda")
                if world > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                if rank == 0:
                    d, c, b, w, dr = stats.tolist()
                    print(json.dumps({"arm": a, "round": r, "pos": pos, "world": world, "nodes": int(os.environ.get("SLURM_NNODES", 1)), "experts_per_rank": El,
                                      "tokens_per_rank": T, "dispatch_ms": round(d, 3), "compute_ms": round(c, 3), "combine_ms": round(b, 3),
                                      "step_ms": round(w, 3), "dropped_max": int(dr),
                                      "tokens_per_s_total": round(world * T / (w / 1e3)), "tokens_per_s_per_gcd": round(T / (w / 1e3))}), flush=True)
        for a in arms:
            check(a)
        if world == 1 and os.environ.get("EP_DIRECT"):
            m = M.FusedMoE(T, TOPK, H, I, E)
            idi = ids.int()
            Wg, Wd = W13[:E], W2[:E]
            ms = []
            for i in range(40):
                t0 = time.perf_counter()
                o = m(X, Wg, Wd, idi, wts)
                torch.cuda.synchronize()
                if i >= 10:
                    ms.append((time.perf_counter() - t0) * 1e3)
            R = M.reference(X, Wg, Wd, idi, wts)
            print(json.dumps({"arm": "direct", "world": 1, "tokens_per_rank": T, "compute_ms": round(sorted(ms)[len(ms) // 2], 3), "cfg": m.cfg,
                              "rel_vs_fp32": round(((o.float() - R).norm() / R.norm()).item(), 7)}, default=str), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
