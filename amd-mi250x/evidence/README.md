# Evidence of the served-model tracks

Result write-ups, result tables and the frozen source of the tracks behind the end-to-end numbers. Each folder has a
`RESULTS.md` with what ran, the tables, the limits and the corrections; statements are labelled [observed], [computed]
or [inferred]. Job numbers are LUMI Slurm job ids.

| Folder | Track | What it holds |
|---|---|---|
| `E2E/` | decode token rate and energy in a served model (decode attention plugin) | `RESULTS.md`, `TABLES-per-repetition.md`, `src/` |
| `PROFILE/` | where a prefill step's time goes; the bf16 GEMM races and its tile table | `RESULTS.md`, `report.md`, `gemm-bf16-tables.md`, per-job tables, `src/` |
| `P1/` | prefill attention in a served model; the fp16 subnormal finding and the scaled kernel | `RESULTS.md`, `tables*.md`, `src/` |
| `G1/` | the bf16 GEMM plugin in a served model, first table | `RESULTS.md`, `check-tables.md`, per-job tables, `src/` |
| `G2/` | per-row-count GEMM table (candidates, landed entries), GEMM with scaled prefill attention (jobs named g2 and g3) | `RESULTS.md`, `candidates.md`, `forms-candidates.txt`, `landed*.md`, `summary-*.md`, per-job tables, `src-md5.txt`, `src/` |
| `QW/` | query-window prefill attention for chunks with a cached prefix (jobs named g4) | `RESULTS.md`, `check-tables.md`, `summary-*.md`, per-job tables, `src/` |
| `MOE-MAX/tuned/` | stock vLLM fused MoE with a configuration tuned for MI250X, as a baseline | `RESULTS.md`, `tab_*.txt`, `scripts/`, `vllm-config*/` |
| `UPSTREAM/vllm-rocm-wvsplitk-5-rows/` | reproduction of a slow small-row GEMM path in stock vLLM | see its `README.md` |

What is not here:

- Raw per-job output (`raw/<job>/` in the write-ups: client records, server logs, counters, power samples, Slurm
  output). The per-job tables in each folder are built from it by the analysis scripts in `src/`. Paths that start with
  `raw/` in a write-up refer to these files.
- Model checkpoints, dumped activations and compiled kernel caches.
- `E2E/src/probe.sbatch`, a one-off probe of the node's power counters.
- `P1/tables-ctx-23626852.md` (incomplete; its numbers are in `P1/RESULTS.md`).

`src/` holds the scripts and kernel copies as they ran, some under the name they were frozen with (a short md5 in the
file name). The current versions are in `../kernels/` and `../vllm-integration/`. Three things differ from the files as
run, so the md5 values quoted in the write-ups, in the file names and in `src-md5.txt` hold for the files as run and
not for these copies: the project number is written as `project_XXXXXXXXX`, node names as `node<N>`, and the second
docstring line of every plugin copy (`horizon_lumi_*__init__*.py`) was reworded. The `src-md5.txt` lists name the files
by their working paths (`tilelang/<file>` is `../kernels/<file>` here, `overlay/` and `vllm-integration/` are
`../vllm-integration/`). Docstrings name the same working paths.

Known artefacts in the tables, which are kept as the analysis scripts wrote them: a spread printed as `nan%` means the
arm has one repetition in that job, so the verdict next to it rests on the other arm's spread only (or on none); the
`dc32` cell of the PROFILE tables has no decode window at default chunking, and its negative step time and zero rate are
artefacts; "N kernels" lists show the eight most used; in `gemm-bf16-tables.md` an arm that did not run shows `None`
and an error of 0; rows with phase `w` in `UPSTREAM/.../served/client-*.jsonl` are warm-up requests and are not used.

`MOE-MAX/tuned/vllm-config/` holds the tuned fused-MoE configuration in vLLM's own format. vLLM looks the file up by
its name (`E=256,N=512,device_name=AMD_INSTINCT_MI200_(MCM)_OAM_LC_MBA_HPE_C2.json`) in the folder named by
`VLLM_TUNED_CONFIG_FOLDER` (`MOE-MAX/tuned/scripts/cfgcheck.py` checks that vLLM loads it); `vllm-config-zipf2/` is the same tuned on skewed routing.
The research log and the header lines of `tab_tune*.txt` call these folders `config/` and `config_zipf2/`.

Limits that hold for every served-model number here: one model (a public 27B hybrid model, BF16), one accelerator
type, tensor parallel 2 on two GCDs (the two dies of one card in E2E, G1, G2 and QW; dies of two shared cards in P1 and
PROFILE), synthetic prompts, and two mirrored jobs per reported pair
unless the write-up says one job. Decode steps run stock code in the prefill tracks. Greedy outputs are not identical
to stock, and no quality benchmark was run.
