# Experiment source

Working copies behind the results/RESEARCH-LOG.md entries MB, E1, E2, E3, K1, N1, N3, N4, N5, C1, C2P, S5, D128, TRI3D, SMALLM, MOESWEEP and JPORT, as
they ran on LUMI. They are not library code. Paths are those of the project they ran in, with the project number replaced by
`project_XXXXXXXXX`: put your own project and directory layout in before running one. Inputs they read that are not in this
release (model checkpoints, prompt files, the patched TileLang tree, generated kernels) have to be supplied the same way.
The real-input captures (N1, N5) read a JSON-lines prompt file of our own, one chat `messages` list per line, under
`data/prompts/`; it is not included, so those real-input numbers cannot be reproduced bit for bit from this release.
Files whose paths were replaced no longer match the md5 values the log quotes. kernels/, baselines/ and diagnostics/ are
copied unchanged except for comment wording in the files listed in kernels/RELEASE-EDITS.md. Landed fixes are in kernels/; these folders hold the arms that
produced them, including superseded versions (for example n1/kernels/v8 to v10, n4/src/gemm_prefetch_<md5>.py). Some scripts
need a tree built from ../tilelang-patches. E3's patch.py and N5's research modules edit generated C. C2P's forms are TileLang
source on the patched fork, not the stock wheel. D128's J form later landed in kernels/gqa_fa.py for D 128 at 2816-3584 (JPORT).
SMALLM's `rng` and 32x96 forms are not landed; they live only in these folders. MOESWEEP's arms beyond `ship` are timing and correctness controls, and `ship_z0` zeroes a real token on purpose,
so it is valid only as a k1 timing control. Raw output, disassembly and generated kernels are not included.
Some scripts record the GPU's uuid or unique id in their output when run: strip those values before sharing that output.
Three imports need something outside this tree: c2p/src/t2.py imports TileLang's own test modules through sys.path (use a
TileLang source checkout), n5/src/n5lib.py imports gqa_vt_gen, a file generated at run time, and ../baselines/ck_fa_baseline.py
imports aiter, AMD's package.
