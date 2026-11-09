# Results

These are kernel-level measurements: kernel races, microbenchmarks, and ISA and tile sweeps, one folder per experiment,
named `<topic>_<dd-mm-yyyy>`. Each folder holds the raw outputs (JSON, JSON lines, tables) of a kernel run; the scripts
that produce them are in `../scripts/kernels/`. Serving-level measurements are not included.

Paths inside these files are generalised: `<data>` is the shared data directory, `<venv>` the Python environment and
`<work>` the experiment directory.
