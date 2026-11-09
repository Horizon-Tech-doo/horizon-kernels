# Docs

A curated selection of the design notes, results and reference material for the SM120 kernels. Dates in file names are
dd-mm-yyyy. The numeric data behind the results is in `../results/`.

Only technical documents are published here, and some of them were shortened to their technical content. Some documents refer to notes or scripts that are not part of this release; those references read "(not in
this release)". `upstream/` holds our drafts of pull requests and issues for vLLM, flash-linear-attention and TileLang.

## Design

- [`gdn-state-design.md`](gdn-state-design.md): GDN recurrent-state work: cost model, 16-bit state and replay-ring kernels, gates
- [`k4-design.md`](k4-design.md): K4 persistent decode-step kernel: task graph, scheduler, expected gain
- [`kv-fp4-design.md`](kv-fp4-design.md): FP4 KV cache: format, kernels, vLLM integration, gates and risks
- [`moe-plan.md`](moe-plan.md): Qwen3.6-35B-A3B MoE on SM120: kernel selection, grouped NVFP4 kernel design, measurements

## Results

- [`gdn-padding-results-28-09-2026.md`](gdn-padding-results-28-09-2026.md): Persistent GDN padding: correctness and timing results
- [`gdn-recovery-results-28-09-2026.md`](gdn-recovery-results-28-09-2026.md): GDN recovery (G2): standalone timings and limits of the evidence
- [`k4-segment-results-28-09-2026.md`](k4-segment-results-28-09-2026.md): K4 persistent segment against the per-op chain: correctness and timing
- [`segment-geometry-results-28-09-2026.md`](segment-geometry-results-28-09-2026.md): Generic persistent segment: GPU correctness cases
- [`stable-experts-and-slice-profile-28-09-2026.md`](stable-experts-and-slice-profile-28-09-2026.md): Stable MoE expert execution and the persistent-schedule profile

## Reference

- [`kernel-model-portability.md`](kernel-model-portability.md): Requirements for reusable kernels and evidence needed per model
- [`kernel-names.md`](kernel-names.md): Glossary of the kernel track labels used in these documents
- [`kernel-source-reuse.md`](kernel-source-reuse.md): Open-source kernels and addons reviewed as references
