# MOE-MAX tuned baseline (10 Nov 2026): our fused MoE against vLLM's Triton kernel tuned for MI250X

Every "ours over vLLM" ratio in rounds 1-6 is against vLLM's untuned default Triton config, because vLLM 0.30.0 ships no
fused-MoE config file for this device. This round tunes vLLM's config for the device and re-races. Our kernel is unchanged
(`moe_fused.py` 8faee1ee, landed dispatch).

**Result in one paragraph.** Tuning cuts the time of vLLM's kernel by 0-31% on uniform routing. Against the tuned kernel our lead
holds at T = 1 to 8 on every routing and at T = 24 on zipf2. It is a tie at T = 16 (zipf1, zipf2), T = 24 (uniform, zipf1)
and T = 4096 (all routings). **It reverses at T = 16 uniform, at T = 1024 and 2048 on every routing, and at T = 2700 uniform:
there our kernel is 3-18% slower than the tuned vLLM kernel.** T = 2700 on skewed routing still leads against the config tuned
vLLM's way (uniform routing) and reverses (4-6% slower) against a vLLM config tuned on zipf2 routing.

Ratios are time, ours over vLLM, lower is better. Graph replay only. Labels: [observed] read from a run, [computed] derived
from observed numbers, [inferred] not measured.

## Setup [observed]
- Shape as in rounds 1-6, confirmed from the harness: E = 256, N = 512 (vLLM's N = `w2.shape[2]`, the intermediate size),
  hidden 2048, top-k 8, fp16 weights and activations.
- vLLM 0.30.0 ROCm image, Triton 3.7.1, torch 2.12.0, clang 22 for our kernels. One GCD per job, small-g.
- vLLM's config file name for this device: `E=256,N=512,device_name=AMD_INSTINCT_MI200_(MCM)_OAM_LC_MBA_HPE_C2.json`. No such file is
  in the image, so vLLM logs "Using default MoE config" and runs `get_default_config`.

## 1. Tuning

### Method
vLLM's own tuner (`benchmarks/kernels/benchmark_moe.py --tune`) is in the image but does not run: it imports Ray, which is not
installed [observed]. `scripts/tune_moe.py` is an equivalent sweep:
- **vLLM's code, imported from the image through a stub `ray` module:** the ROCm search space (`get_configs_compute_bound`: BLOCK_SIZE_M
  and N in 16-256, BLOCK_SIZE_K in 16-256, GROUP_SIZE_M 1/4/8/16/32, num_warps 1/2/4/8, num_stages 2, waves_per_eu 0/1/2/4,
  matrix_instr_nonkdim 16/32, kpack 1/2; 40,000 configs) and its pruning (`prune_rocm_configs` for both GEMMs).
- **Measurement, mirroring vLLM's `benchmark_config`:** `override_config(cfg)`, `fused_topk` on random gating, `fused_experts`, a HIP
  graph of several calls, new gating before every replay, mean time per call. Differences: weights are scaled by 1/sqrt(fan-in) so
  the fp16 output stays finite (vLLM uses unscaled randn), and the weights are built once per T, not once per config.
- **Exhaustive mode (T = 1, 2, 3, 4, 8), as vLLM's tuner:** every pruned config is timed. 704 to 4,992 configs per T.
- **Staged mode (T = 16, 24, 1024, 2048, 2700, 4096):** the pruned space is 7,808 to 25,760 configs per T, too many to time exhaustively.
  Stage 1 times every pruned block shape and num_warps (400 configs) with GROUP_SIZE_M 1, waves_per_eu 0, matrix_instr_nonkdim 16,
  kpack 1. Stage 2 takes the 8 fastest block/warp choices and times every pruned combination of the other four parameters
  (192 to 632 configs).
- **Final pick, both modes:** the 5 fastest configs and vLLM's default, 3 interleaved rounds at vLLM's tuner settings (graph of 10
  calls, 20 replays). The fastest by median is the tuned config.
- **Skip rule:** a config whose eager call takes more than 3x the best eager call so far plus 1 ms is recorded and not graphed.
- **Compiler aborts:** one config per T >= 8 kills the process inside the compiler ("LLVM ERROR: Found 1 machine code errors":
  128x256x64 with 1 warp; at T = 8 128x256x32, 1 warp, matrix_instr_nonkdim 32). It is logged as failed and left out.
- Configs are compiled by 12 parallel processes first (Triton cache), then timed by one process alone on the device.

### Result [observed, `tab_tune.txt`; times are from the tuner's own inputs, uniform routing]
| T | mode | pruned space | configs timed | of those skipped | default us | tuned us | tuned/default | tuned config (M x N x K, group, warps, waves_per_eu, nonkdim, kpack) |
|---|---|---|---|---|---|---|---|---|
| 1 | exhaustive | 704 | 704 | 0 | 71.06 | 67.79 | 0.954 | 16x64x128, G1, w4, wpe 4, nk 16, kp 1 |
| 2 | exhaustive | 704 | 704 | 0 | 109.41 | 108.64 | 0.993 | 16x16x256, G1, w2, wpe 2, nk 16, kp 1 |
| 3 | exhaustive | 1,408 | 1,408 | 1 | 154.95 | 143.54 | 0.926 | 32x64x128, G1, w8, wpe 4, nk 16, kp 1 |
| 4 | exhaustive | 2,656 | 2,656 | 32 | 192.43 | 181.48 | 0.943 | 16x32x256, G1, w2, wpe 0, nk 16, kp 2 |
| 8 | exhaustive | 4,992 | 4,991 | 339 | 371.39 | 325.92 | 0.878 | 16x128x128, G1, w8, wpe 1, nk 16, kp 1 |
| 16 | staged | 7,808 | 591 | 28 | 586.99 | 514.04 | 0.876 | 16x128x128, G1, w8, wpe 0, nk 16, kp 2 |
| 24 | staged | 7,808 | 599 | 34 | 763.08 | 674.63 | 0.884 | 16x128x128, G1, w8, wpe 2, nk 16, kp 1 |
| 1024 | staged | 25,760 | 1,031 | 66 | 2248.23 | 1560.22 | 0.694 | 64x128x128, G1, w8, wpe 1, nk 16, kp 1 |
| 2048 | staged | 25,760 | 1,031 | 205 | 2614.18 | 2099.12 | 0.803 | 64x128x128, G1, w8, wpe 0, nk 16, kp 1 |
| 2700 | staged | 25,760 | 1,031 | 189 | 2740.06 | 2403.51 | 0.877 | 128x128x128, G1, w8, wpe 1, nk 16, kp 1 |
| 4096 | staged | 25,760 | 1,031 | 152 | 3712.93 | 3513.34 | 0.946 | 64x128x128, G1, w8, wpe 0, nk 16, kp 1 |

"Configs timed" excludes vLLM's default and the config that aborts the compiler. vLLM's default is 16x64x128, 4 warps at T <= 24 and
128x128x64, 8 warps at T >= 1024 (GROUP_SIZE_M 1, 2 stages).

- **Where the gain comes from [observed].** At T >= 1024 the tuned configs use BLOCK_SIZE_K 128 where the default uses 64 (and 64-row
  blocks at 1024, 2048 and 4096). At T = 8 to 24 they use BLOCK_SIZE_N 128 with 8 warps where the default uses 64 with 4. The other four
  parameters move the time by about 2% or less on the best block shape at T >= 16.
- **Is the staged search enough? [computed, `stage_check.py` on the exhaustive records]** Replaying the staged selection on the five
  exhaustive T finds the exhaustive best at T = 1, 2, 4 and 8 and a config 1.7% slower at T = 3 (single-pass timings). At T >= 1024
  stage 1 is sharply peaked: the best block shape beats the second by 2.6% to 12.6% at T = 1024, 2048 and 2700 and ties one other at
  4096, and GROUP_SIZE_M above 1 never wins on uniform routing. So the staged result is unlikely to be far from the full-space best, but
  this is not measured at T >= 16 [inferred].
- **Tuned config file:** `vllm-config/E=256,N=512,device_name=AMD_INSTINCT_MI200_(MCM)_OAM_LC_MBA_HPE_C2.json` (md5 70817463), vLLM's format with
  `triton_version`, keys 1, 2, 3, 4, 8, 16, 24, 1024, 2048, 2700, 4096. **vLLM itself loads it [observed, `cfgcheck` record in all
  four race jobs]:** with `VLLM_TUNED_CONFIG_FOLDER` set, `get_moe_configs` returns these keys and `try_get_optimal_moe_config` returns
  the file's entry at all 11 T. For any other T vLLM picks the nearest key; the file has no entry between 24 and 1024.

## 2. Race

`scripts/mxt.py` is round 6's harness reduced to five arms in one process, all captured as HIP graphs: **ship** (landed module),
**same** (a second, separately built instance: identical-code noise), **vllm** (vLLM's own lookup, which falls to the default
config), **vllm2** (the same call captured again: identical-code noise on vLLM's side) and **vllmt** (`fused_experts` under
`override_config` with the tuned entry). 6 rounds of 60 replays per arm, arm order rotated per round and reversed in the second job.
11 T x 3 routings x seeds 7 and 8 = 66 cells per job. Routing draws: `ms_mk.py` from `data_w.pt` (data_t_7.pt md5 50dd514c,
data_t_8.pt md5 0a93d558; not kept, `ms_mk.py` regenerates them).

### Rule (written in `tab_t.py` before the race jobs ran; same form as round 6)
noise(job) = max over every cell of that job of |same/ship - 1| and |vllm2/vllm - 1|. For a T and routing against a vLLM baseline:
**LEAD** if vLLM/ours > 1 + noise(job) in both seeds of every job, **REVERSED** if ours/vLLM > 1 + noise(job) in both seeds of every
job, **TIE** otherwise.

### Jobs [observed, sacct]
| job | GCD-s | what |
|---|---|---|
| 23585740 | 185 | tuning T = 1, first version of the staged search (not used for the config file) |
| 23585820-23 | 4 x 172 | tuning, cancelled: a compiler abort killed the prebuild shards; the tuner was made restartable |
| 23585950 | 2,147 | tuning, exhaustive, T = 4, 3, 2, 1, 8 |
| 23585951 | 680 | tuning, staged, T = 16, 24 |
| 23585952 | 1,113 | tuning, staged, T = 1024, 2048 |
| 23585953 | 1,058 | tuning, staged, T = 2700, 4096 |
| 23586384 | 408 | race fwd (5 arms) |
| 23586385 | 413 | race rev (5 arms) |
| 23586486, 23586487, 23586598 | 821 + 806 + 1,084 | tuning on zipf2 routing (section 3), staged, T = 2, 3, 4, 8 / 16, 24, 1 / 2700, 4096 |
| 23586779 | 435 | race fwd with the extra arm vllmz |
| 23586781 | 439 | race rev with the extra arm vllmz |

Total 10,277 GCD-s = **2.85 GCD-hours** [computed].

### Noise [computed]
noise(job) is 1.03% (23586384), 1.51% (23586385), 1.05% (23586779) and 0.90% (23586781). Ours alone (same/ship) reaches 1.06%, vLLM
alone (vllm2/vllm) 1.51%. Every arm has 6 rounds in every cell; no arm failed.

### Result [observed ranges over 2 seeds x 4 jobs; verdicts computed by the rule over all 4 jobs; `tab_tuned_all.txt`]
"best vLLM" is, per cell, the fastest vLLM arm raced: default or tuned in the first two jobs, plus the zipf2-tuned config of
section 3 in the last two. The zipf2-tuned column is from the last two jobs only.

| T | routing | ours / vLLM default | verdict | ours / vLLM tuned | verdict | ours / vLLM zipf2-tuned | ours / best vLLM | verdict | tuned / default |
|---|---|---|---|---|---|---|---|---|---|
| 1 | uniform | 0.842-0.891 | LEAD | 0.844-0.891 | LEAD | 0.849-0.889 | 0.844-0.891 | LEAD | 0.985-1.038 |
| 1 | zipf1 | 0.843-0.902 | LEAD | 0.845-0.900 | LEAD | 0.851-0.892 | 0.845-0.902 | LEAD | 0.986-1.038 |
| 1 | zipf2 | 0.839-0.896 | LEAD | 0.844-0.894 | LEAD | 0.855-0.894 | 0.844-0.896 | LEAD | 0.984-1.041 |
| 2 | uniform | 0.836-0.852 | LEAD | 0.827-0.876 | LEAD | 0.846-0.860 | 0.846-0.876 | LEAD | 0.962-1.024 |
| 2 | zipf1 | 0.843-0.855 | LEAD | 0.785-0.875 | LEAD | 0.836-0.861 | 0.843-0.875 | LEAD | 0.977-1.083 |
| 2 | zipf2 | 0.772-0.835 | LEAD | 0.670-0.681 | LEAD | 0.756-0.818 | 0.772-0.835 | LEAD | 1.150-1.226 |
| 3 | uniform | 0.830-0.850 | LEAD | 0.900-0.923 | LEAD | 0.847-0.855 | 0.900-0.923 | LEAD | 0.902-0.943 |
| 3 | zipf1 | 0.803-0.844 | LEAD | 0.851-0.874 | LEAD | 0.814-0.846 | 0.851-0.874 | LEAD | 0.928-0.976 |
| 3 | zipf2 | 0.787-0.825 | LEAD | 0.745-0.765 | LEAD | 0.797-0.809 | 0.792-0.825 | LEAD | 1.036-1.107 |
| 4 | uniform | 0.837-0.869 | LEAD | 0.885-0.916 | LEAD | 0.829-0.852 | 0.885-0.916 | LEAD | 0.922-0.975 |
| 4 | zipf1 | 0.788-0.833 | LEAD | 0.753-0.828 | LEAD | 0.773-0.817 | 0.799-0.833 | LEAD | 0.962-1.094 |
| 4 | zipf2 | 0.715-0.742 | LEAD | 0.554-0.612 | LEAD | 0.713-0.730 | 0.715-0.742 | LEAD | 1.199-1.298 |
| 8 | uniform | 0.839-0.870 | LEAD | 0.945-0.967 | LEAD | 0.821-0.851 | 0.945-0.967 | LEAD | 0.883-0.900 |
| 8 | zipf1 | 0.607-0.698 | LEAD | 0.676-0.774 | LEAD | 0.643-0.698 | 0.676-0.774 | LEAD | 0.898-0.912 |
| 8 | zipf2 | 0.469-0.631 | LEAD | 0.467-0.645 | LEAD | 0.490-0.626 | 0.469-0.645 | LEAD | 0.970-1.018 |
| 16 | uniform | 0.914-0.938 | LEAD | 1.027-1.077 | **REVERSED** | 1.044-1.070 | 1.027-1.077 | **REVERSED** | 0.871-0.890 |
| 16 | zipf1 | 0.931-0.956 | LEAD | 0.993-1.072 | TIE | 1.029-1.065 | 0.993-1.072 | TIE | 0.884-0.938 |
| 16 | zipf2 | 0.930-0.960 | LEAD | 0.943-1.006 | TIE | 0.966-1.006 | 0.943-1.006 | TIE | 0.932-0.991 |
| 24 | uniform | 0.867-0.885 | LEAD | 0.982-0.996 | TIE | 0.961-0.978 | 0.982-0.996 | TIE | 0.877-0.897 |
| 24 | zipf1 | 0.882-0.893 | LEAD | 0.966-0.992 | TIE | 0.943-0.970 | 0.966-0.992 | TIE | 0.896-0.917 |
| 24 | zipf2 | 0.897-0.944 | LEAD | 0.931-0.951 | LEAD | 0.897-0.987 | 0.931-0.987 | LEAD | 0.961-0.998 |
| 1024 | uniform | 0.752-0.766 | LEAD | 1.082-1.105 | **REVERSED** | not raced | 1.082-1.105 | **REVERSED** | 0.689-0.697 |
| 1024 | zipf1 | 0.856-0.875 | LEAD | 1.128-1.146 | **REVERSED** | not raced | 1.128-1.146 | **REVERSED** | 0.755-0.765 |
| 1024 | zipf2 | 0.927-0.945 | LEAD | 1.119-1.133 | **REVERSED** | not raced | 1.119-1.133 | **REVERSED** | 0.824-0.835 |
| 2048 | uniform | 0.906-0.916 | LEAD | 1.091-1.129 | **REVERSED** | not raced | 1.091-1.129 | **REVERSED** | 0.803-0.836 |
| 2048 | zipf1 | 0.884-0.904 | LEAD | 1.081-1.103 | **REVERSED** | not raced | 1.081-1.103 | **REVERSED** | 0.807-0.823 |
| 2048 | zipf2 | 0.940-0.950 | LEAD | 1.049-1.063 | **REVERSED** | not raced | 1.049-1.063 | **REVERSED** | 0.884-0.902 |
| 2700 | uniform | 0.992-1.017 | TIE | 1.150-1.179 | **REVERSED** | 0.863-0.914 | 1.150-1.179 | **REVERSED** | 0.844-0.863 |
| 2700 | zipf1 | 0.896-0.913 | LEAD | 0.938-0.970 | LEAD | 1.058-1.061 | 0.938-1.061 | TIE (see 3) | 0.937-0.961 |
| 2700 | zipf2 | 0.937-0.957 | LEAD | 0.923-0.950 | LEAD | 1.039-1.054 | 0.940-1.054 | TIE (see 3) | 0.997-1.023 |
| 4096 | uniform | 0.892-0.917 | LEAD | 0.976-0.994 | TIE | 0.964-0.972 | 0.976-0.994 | TIE | 0.902-0.936 |
| 4096 | zipf1 | 0.923-0.939 | LEAD | 1.003-1.013 | TIE | 1.000-1.009 | 1.003-1.013 | TIE | 0.914-0.927 |
| 4096 | zipf2 | 0.952-0.964 | LEAD | 0.997-1.007 | TIE | 1.002-1.017 | 0.997-1.017 | TIE | 0.950-0.961 |

Counts over the 33 cells [computed]: against the default 32 LEAD, 1 TIE. Against the tuned config 18 LEAD, 7 TIE, 8 REVERSED. Against
the best vLLM arm 16 LEAD, 9 TIE, 8 REVERSED.

### What this says, plainly
- **The default-config ratios of rounds 5 and 6 reproduce [observed].** Against the default we lead everywhere except the known parity at
  T = 2700 uniform (0.992-1.017).
- **Lead holds against tuned vLLM: T = 1, 2, 3, 4, 8 on every routing, and T = 24 zipf2.** Against the best vLLM arm: 0.844-0.902 at
  T = 1, 0.772-0.876 at T = 2, 0.792-0.923 at T = 3, 0.715-0.916 at T = 4, 0.469-0.967 at T = 8, 0.931-0.987 at T = 24 zipf2.
- **Lead shrinks:** T = 3 and 4 uniform go from 0.830-0.869 to 0.885-0.923, T = 8 uniform from 0.839-0.870 to 0.945-0.967, T = 8 zipf1
  from 0.607-0.698 to 0.676-0.774. T = 1 does not move: tuning gains vLLM nothing measurable there (tuned/default 0.984-1.041).
- **Lead disappears (TIE):** T = 16 zipf1 and zipf2, T = 24 uniform and zipf1, T = 4096 on all three routings (0.976-1.017).
- **Lead reverses:** T = 16 uniform (ours 2.7-7.7% slower), T = 1024 on all routings (8.2-14.6% slower), T = 2048 on all routings
  (4.9-12.9% slower), T = 2700 uniform (15.0-17.9% slower).
- **T = 2700 skewed depends on how vLLM is tuned.** Against the config tuned vLLM's way we lead (0.923-0.970). Against the config tuned
  on zipf2 routing we are 3.9-6.1% slower in both seeds of both jobs that raced it.
- **The uniform-tuned config is not always vLLM's fastest [observed].** On zipf2 at T = 2, 3 and 4 it is 4-30% slower than vLLM's default
  (tuned/default 1.036-1.298), and on one of two seeds at T = 2 and 4 zipf1. "Ours / tuned" alone would overstate our lead there
  (0.554-0.765 on zipf2); the "best vLLM" column removes that.
- Absolute, seed 7 uniform, job 23586384, ours / default / tuned in us [observed]: T = 1 59.4 / 70.5 / 69.9; T = 24 667.7 / 762.3 /
  674.5; T = 1024 1683.1 / 2230.8 / 1555.3; T = 2700 2794.4 / 2781.9 / 2391.8; T = 4096 3429.7 / 3754.1 / 3512.4.

### Correctness of the tuned config [observed, all four race jobs, 66 cases each]
- Relative L2 error against an fp64 per-expert reference, worst case: tuned 6.020e-4, default 6.020e-4, ours 3.804e-4. The fp32
  reference gives the same figures to four digits. 0 non-finite values in every arm.
- Tuned against default output: bitwise equal in 54 of 66 cases in every job; in the rest the largest difference is 4.883e-4, exactly one fp16
  unit in the last place for values of 0.5 to 1 [computed], relative L2 at most 7.98e-5, and at least 96.05% of the elements are bitwise
  equal. The zipf2-tuned config's worst error is 6.018e-4.
- In the tuner's final stage (uniform and zipf2 runs) the 5 finalists per T differ from the default's output by at most 1.11e-4
  relative L2, 0 non-finite.

## 3. Extra arm: a vLLM config tuned on zipf2 routing

vLLM's tuner draws uniform routing, and one config file serves every routing. Because the uniform-tuned config loses to the default
on skewed routing at small T, the staged search was repeated with the race data's zipf2 recipe as the gating
(`tune_moe.py --route zipf2`; not something vLLM's tuner offers) at T = 1, 2, 3, 4, 8, 16, 24, 2700 and 4096, and the result raced as
a third vLLM arm **vllmz** in two more jobs (23586779 fwd, 23586781 rev; `tab_tuned_z.txt`). T = 1024 and 2048 were left out: the
uniform-tuned config already beats us there on zipf2.

- **Tuner result [observed, `tab_tune_zipf2.txt`, tuned/default on the tuner's zipf2 inputs]:** 0.981 (T = 1), 0.986, 0.988, 0.990,
  0.968 (T = 8), 0.987 (T = 16), 0.928 (T = 24), 0.895 (T = 2700), 0.945 (T = 4096). At T = 2700 and 4096 the winner is 64x128x128 with
  GROUP_SIZE_M 32, where uniform tuning picks GROUP_SIZE_M 1. Config: `vllm-config-zipf2/` (same file name, md5 6a731f92).
- **Race [observed, 2 jobs]:** at T <= 8 the zipf2-tuned config is within a few percent of the default on zipf2, so the lead there
  stands (0.490-0.894 on zipf2). At T = 2700 it reverses the skewed cells: zipf1 1.058-1.061, zipf2 1.039-1.054. On uniform routing at
  T = 2700 it is the slowest vLLM arm (ours/zipf2-tuned 0.863-0.914). T = 4096 stays a tie (1.000-1.017 on skewed routing).
- A deployment has one config per batch size. Which of the two tuned files is the right baseline at T = 2700 depends on the routing
  the deployment sees; both are reported.

## Limits and open items
1. **Staged, not exhaustive, at T >= 16.** 591 to 1,031 of 7,808 to 25,760 pruned configs were timed per T. A full sweep can only make
   vLLM faster, so the ours/tuned ratios at T >= 16 are lower bounds: the reversals stand, a tie or a remaining lead could move
   against us [inferred].
2. **T not tuned or raced here:** 5-7, 9-15, 17-23, 2200, 2875, 3600 (quoted in round 5 and in the paper). vLLM would use the nearest
   key of the file.
3. **Routing-aware tuning is partial:** zipf2 only, not zipf1, not T = 1024 and 2048.
4. **Two synthetic routing seeds**, as in rounds 5 and 6. Not real model routing.
5. **The tuner's timing includes `fused_topk`** inside the graph, as vLLM's does. It is the same for every config.
6. **The first race pair ran `mxt.py` md5 ac656d87**; the kept file (cd2e544f) adds only the vllmz arm and `--tuned2`. The earlier file
   was not kept.
7. **Our kernel was not changed or re-tuned** against the tuned baseline. The tuned vLLM configs point at BLOCK_SIZE_K 128 and, on
   skewed routing, block grouping; whether our dispatch can use that is future work.

## Files
- `scripts/`: `tune_moe.py` (2613424), `tune.sbatch`, `mk_cfg.py`, `stage_check.py`, `mxt.py`, `mxt.sbatch`, `cfgcheck.py`, `tab_t.py`,
  `ms_mk.py` (round 6's, 66c0aeac), `sgt.sh`.
- `vllm-config/`, `vllm-config-zipf2/`: the tuned files in vLLM's format.
- `raw/` (not included in this release): `tune_T<T>[_zipf2]_<job>.jsonl` (every config timed, 20,515 timings), `mxt_<job>.jsonl` (races), job logs.
- `tab_tune.txt`, `tab_tune_zipf2.txt`, `tab_tuned.txt` (first race pair), `tab_tuned_z.txt` (second pair), `tab_tuned_all.txt` (all four).
