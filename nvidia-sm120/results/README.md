# Results

These are kernel-level measurements: kernel races, microbenchmarks, and ISA and tile sweeps, one folder per experiment,
named `<topic>_<dd-mm-yyyy>`. Each folder holds the raw outputs (JSON, JSON lines, tables) of a kernel run; the scripts
that produce them are in `../scripts/kernels/`. Serving-level measurements are not included.

Paths inside these files are generalised: `<data>` is the shared data directory, `<venv>` the Python environment and
`<work>` the experiment directory.

`paper1_11-11-2026/` is the re-measurement of 11 November 2026 (kit: `../scripts/kernels/paper1/`): `RESULTS.md` is
the summary; `analysis/` (seed 988) and `analysis-seed989/` hold the per-cell analysis output of `p1_analyze.py` (text
and JSON) plus a summary table built from it. The per-cell records of the blocks that `p1_analyze.py` reads (about
20 MB of JSON lines) are not in the repository; they are available on request. The code hashes quoted in the analysis
are of the files as run; the published files carry licence headers and generalised paths, so they do not match this tree.
