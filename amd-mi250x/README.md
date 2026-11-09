# horizon-kernels-gfx90a

TileLang attention, mixture-of-experts and GEMM kernels for AMD Instinct MI250X (gfx90a), with the benchmark harness,
vendor baselines, experiment source and the research log from our study on LUMI-G. Apache-2.0 (see `../LICENSE`, `../NOTICE`
and `../REUSE.toml`). Copyright 2026 Horizon Tech d.o.o.; author Hamza Abdagic.

TileLang compiler fixes are in `tilelang-patches/`, prepared against upstream main 6e8f599e; see
`tilelang-patches/README.md` for their state and licence.

## What is here

| Folder | Contents |
|---|---|
| `kernels/` | `gqa_fa.py` GQA prefill attention; `gqa_decode.py` flash-decoding, contiguous and paged KV on vLLM's layouts, simulated FP8 KV; `moe_fused.py` fused MoE (GPU alignment, gate/up+SiLU, down); `gemm_prefetch.py` dense and split-K GEMM; `bench_gemm.py` |
| `baselines/` | PyTorch SDPA, Composable Kernel flash-attention (prefill and decode), vLLM paged / Triton unified attention, vLLM fused MoE |
| `diagnostics/` | the four experiments that showed the upstream TileLang example's fp16 reference was wrong, not the kernel |
| `scaling/` | RCCL all-reduce / all-to-all, an expert-parallel MoE layer, 1 GCD to 4 nodes, and vLLM TP=4 across nodes with a hand-written OFI adapter (Slurm scripts) |
| `vllm-integration/` | LUMI overlay that runs the decode kernels in a live vLLM, with its client and analysis; research only |
| `lumi/` | reproducible TileLang 0.1.15 setup in the LUMI PyTorch container, including the gfx90a hipcc shim, `install.sh`, the GPU smoke test and the vLLM image pull |
| `results/RESEARCH-LOG.md` | every kernel, baseline, scaling and compiler experiment with setup, numbers, job IDs and conclusions, including negative results and later corrections |
| `experiments/` | working source behind the MB, E1, E2, E3, K1, N1, N3, N4, N5, C1, C2P, S5, D128, TRI3D, SMALLM, MOESWEEP and JPORT log entries, as run (see `experiments/README.md`) |
| `tilelang-patches/` | patches 0001–0009 for TileLang |

## Headline results (one MI250X GCD unless stated)

Ratios are ours over the vendor's throughput, higher is better, except decode (vLLM ms over ours, higher is better) and MoE (ours over vLLM's time, lower is better).
Details, job IDs and limits are in `results/RESEARCH-LOG.md`.

- **GQA prefill**, head size 256, 6 query heads per KV head, causal: 1.06–1.18× Composable Kernel (CK) across lengths and real inputs, lowest about 1.07× on random data at 16k, same process on ROCm 7.2.3; 1.38–1.93× SDPA. Head size 128: beats CK up to 2560 tokens with `d128_tile`, 0.994–0.997× at 3072 through the script; **form J, first a TileLang source form on the patched fork (`experiments/d128`): 1.017–1.038× at 2816 and 1.015–1.036× at 3072 on six inputs, 0.991–1.015× at 3584, 0.978–1.005× at 4096. Since then form J ships in `kernels/gqa_fa.py` for D 128 at 2816–3584 when hipcc is clang 22, through a source template on the stock wheel (`experiments/jport`, jobs 23561796 and 23561797, six inputs): 1.027–1.040× CK at 2816, 1.026–1.036× at 3072, 1.014–1.024× at 3328, 0.995–1.013× at 3584; clang 18 and other lengths keep the old dispatch; at 8k 1.172–1.230× SDPA, on random data median 1.186, so that 1.2× row is not reliably closed**; 0.946–0.971× from 5120 to 16k. MHA d128 4k 0.98–1.00×. Kernels are C edits on generated source or forms in TileLang source on the fork; stock TileLang reaches 0.93×, 0.79× and 0.80× at d256 8k, d128 4k and MHA 4k.
- **Paged decode**, BF16, page 784, vLLM's frame read from live starts, against vLLM's Triton kernel **in the 3D form a server runs up to a batch threshold (`experiments/tri3d`), about level: at or below about 0.87 up to 1.37× at 4 KV heads per GCD, 0.881–1.300× at 2, real losses at some shapes; replay uses a length-fitted kernel that serving cannot deploy, and the unfitted form reads below 1 at 5 of 8 served cells, lowest 0.856×.** Against the 2D form only: 1.10–5.33× on the four shapes one GCD can serve (LBNHC, needs the TRITON_ATTN flag); 2.79–13.47× ROCM_ATTN; 1.33–4.76× ROCM_ATTN with no vLLM setting changed. **In a running vLLM (`vllm-integration/`), two opposite-order jobs: 0.966–1.009× stock TRITON_ATTN, no token-rate gain,** which the 3D parity explains in those cells, since a step takes 30–51 ms; a plugin loss of up to 1.25 ms per step stays unexplained.
- **Fused MoE** vs vLLM 0.30 `fused_experts`, time, same process on vLLM's stack, compiler output: at most 0.964× from 2 to 2049 tokens; 0.892–0.998× at 14 sampled counts in 2049–4096 on quiet jobs, **but a fine sweep (`experiments/moesweep`, two opposite-order pairs, two routing seeds) puts about 2600–2875 tokens at parity, up to 2.5% over in the worst read, with the sign depending on seed and job; the band's lower edge is unmeasured.** Grid kernels are correct on every count from 1 to 4096; landed kernels on sampled counts. Real routing 0.858–0.966× at 1280–4096, 0.976–0.991× from 6144. Uniform 7168 reads 1.009–1.013×, 12288 and 16384 1.001–1.007×. Weight bandwidth at decode size 53–77% of HBM at T ≥ 4, under 50% at T = 1, 2. Each new token count costs 20–30 s of compile, excluded from these ratios.
- **Dense GEMM**, hand-written asm launched by TileLang, not compiler output: at the 500 W cap through the script, placed 1.016–1.075× hipBLASLt at 3072³, 6144³, 12288³; unplaced 1.005–1.069×. Earlier placed table at 8 shapes 1.016–1.093×. 7168×3584×4096, 2816³ and 1024×7680×2048 stay below 1. Compiled forms 0.77–1.08×.
- **Small-M GEMM**, hand asm on one allocation, explicit entries: 0.988–1.037× on N 17408; 0.979–1.037× on N 5120. Experimental, not landed (`experiments/smallm`): 1.015× at 16×17408×5120, 1.137–1.141× at 64×5120×5120. Compiled alone misses 0.8× at 16×17408×5120 (0.786–0.798×).
- **Scaling**, landed MoE, clang 18, synthetic routing: 76.4–76.7% weak-scaling efficiency at 8 GCDs, 39.4–39.7% at 16, 25.8–26.6% at 32, with more throughput than the old kernels' 99%, 56%, 38%. vLLM TP=4 across two nodes: 0.19–0.23× of one node on prefill over the image's sockets, 0.65–0.87× through a hand-written OFI adapter (`scaling/ofi_v6_adapter.c`, not a vendor result), and it hangs intermittently on both.

The log has the full tables, the method and what did not work.

## Requirements

- An MI250X (gfx90a) system. Everything was measured on one GCD of LUMI-G unless a log entry says otherwise; LUMI
  paths, partitions (`dev-g`, `small-g`, `standard-g`) and Slingshot settings appear in the Slurm scripts.
- Kernels and SDPA comparisons: the LUMI PyTorch container
  `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif` (PyTorch 2.7.1, ROCm 6.2.4) with `tilelang==0.1.15` from PyPI
  installed `--no-deps` into a venv, and the hipcc shim written by `lumi/setup.sh`. Elsewhere: any ROCm 6.2+ PyTorch
  with `tilelang==0.1.15` and the same shim (two lines of header; see `lumi/setup.sh`).
- Vendor baselines (Composable Kernel flash-attention 2.8.3, vLLM 0.30 kernels, hipBLASLt in the same process) and
  the vLLM runs: the official image `vllm/vllm-openai-rocm:v0.30.0` (ROCm 7.2.3, torch 2.12) pulled as a Singularity
  image with `lumi/pull.sbatch` (`TAG=v0.30.0`). ROCm 7.2.3 is outside LUMI's supported range; it ran here, as the log
  records.
- vLLM runs need a model checkpoint: the log used the public `Qwen/Qwen3.8-27B` (BF16) and, for real MoE inputs,
  `Qwen/Qwen3.6-35B-A3B`. Download them yourself and set `MODEL`.

## Setup on LUMI

```bash
export PROJ=project_<your number>          # your LUMI project; every script reads it
lumi/install.sh                            # copies the tree into $TL_ROOT (default /scratch/$PROJ/tilelang)
$TL_ROOT/scripts/setup.sh                  # login node: venv, TileLang wheel, upstream v0.1.15 source, hipcc shim
sbatch -A $PROJ $TL_ROOT/scripts/gpu_test.sbatch   # optional smoke test, one GCD, dev-g
```

The Slurm scripts carry no account: pass `-A $PROJ` to `sbatch`. Their logs go to `%x-%j.out` in the directory you
submit from. Paths can be overridden with `TL_ROOT`, `VLLM_SIF`, `MODEL`, `VLLM_DIR`, `OFI_DIR`, `V1_DIR`; each script
lists what it reads near its top.

## Running each benchmark

All commands run inside a GPU allocation, e.g. `salloc -A $PROJ -p dev-g --gpus-per-node=1 -t 00:30:00`, then
`srun --pty bash`. `T=$TL_ROOT/scripts/tl_run.sh` runs a command in the PyTorch container with the TileLang venv;
`V="singularity exec -B /pfs,/scratch,/flash $VLLM_SIF"` runs one in the vLLM image. Every script prints JSON lines
with correctness against a float64 or float32 reference and timings.

| Benchmark | Command (from `$TL_ROOT/scripts`) |
|---|---|
| GQA prefill vs SDPA | D 256: `$T "python gqa_fa.py 4096 - 256 1"`; D 128 with the landed tile rule: `$T "python gqa_fa.py 3072 auto 128 1"` (args: L, tile as JSON / `-` / `auto`, D, B, G; G 1 is MHA; `auto` covers D 128 only) |
| GQA prefill vs CK | `$V python3 ck_fa_baseline.py` (same shapes as `gqa_fa.py`; `TRY_AITER=1` also tries aiter) |
| Decode, contiguous / paged | `$T "python gqa_decode.py"`, `$T "python gqa_decode.py paged"`, simulated FP8 KV `$T "python gqa_decode.py fp8"` |
| Decode vs CK and vLLM | `$V python3 ck_decode_baseline.py`, `$V python3 vllm_paged_baseline.py` |
| Fused MoE | `$T "python moe_fused.py --tokens 1,16,64,256,1024,4096"` (`--H --I --E --topk` set the shape) |
| Fused MoE vs vLLM | `$V python3 vllm_moe_baseline.py`; same-process comparisons as in `experiments/e2`, `experiments/n1` |
| Dense GEMM | `$T "python bench_gemm.py"`; `$T "python bench_gemm.py --best 4 --placed"`; `gemm_prefetch.py` modes `auto`, `smallm`, `sweep`, `splitk`, `handsrc` (docstrings give their arguments) |
| Upstream example and its reference | `$T "python fa_bench.py $TL_SRC"`, `run_fa.py`, `fa_diag.py`, `fa_diag2.py $TL_SRC`, `fa_diag3.py $TL_SRC` |
| RCCL collectives | `sbatch -A $PROJ --nodes=N $TL_ROOT/scaling/comm.sbatch` |
| Expert-parallel MoE | `EP_WORLDS="1 8 16 32" sbatch -A $PROJ --nodes=4 --gpus-per-node=8 --ntasks-per-node=8 $TL_ROOT/scaling/ep.sbatch` (fill the cache first; see the script header) |
| vLLM TP=4, one node vs two | `ARMS="one two" MODEL=... sbatch -A $PROJ $TL_ROOT/scaling/vllm_mn.sbatch` |
| Decode kernels in a running vLLM | `ARMS="s_rocm i_rocm s_tri i_tri" GSM8K_JSONL=... MODEL=... sbatch -A $PROJ $TL_ROOT/exp/V1/vi/v1.sbatch` |

`$TL_SRC` is the upstream TileLang v0.1.15 source checkout that `setup.sh` creates. Same-process vendor comparisons
put the TileLang venv on the vLLM image's path:
`$V bash -c "PYTHONPATH=$TL_ROOT/venv/lib/python3.12/site-packages python3 ..."`, as `experiments/e2/job1.sh` does.
Each new shape costs a TileLang compile (20–30 s for MoE token counts), which the timings exclude.

Limits a stranger should know:
- `vllm-integration/v1.sbatch` needs `GSM8K_JSONL`, a JSON-lines file with `id`, `question` and `gold` (the final
  number) per line, e.g. converted from the GSM8K test split; it reads the first 64. The logged runs also passed a chat
  template of our own; leave `CHAT_TEMPLATE` empty to use the model's.
- The OFI arms of `scaling/ofi_comm.sbatch` and `scaling/vllm_mn.sbatch` need a directory (`OFI_DIR`) assembled by hand
  and not included: `adapter/` with `libnccl-net.so` built from `scaling/ofi_v6_adapter.c` (build line in its header),
  `asis/`, `aws-ofi-rccl/`, `libfabric/` and `shs-libcxi/` from LUMI's ROCm 6.2.4 stack, and `shim/` with a
  `libamdhip64` compatibility shim for ROCm 7.2.3. Socket arms (`one`, `two`, `sock`) need none of it.
- `experiments/` are working copies with the paths of the project they ran in, written as `project_XXXXXXXXX`; some
  need a TileLang tree built from `tilelang-patches/`. They are not library code.
- Kernel source forms marked "fork" in the log need TileLang built from upstream main 6e8f599e plus `tilelang-patches/`.

## Acknowledgement

We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LUMI, hosted
by CSC (Finland) and the LUMI consortium, through a EuroHPC Development Access call (EHPC-DEV-2026D11-419).
