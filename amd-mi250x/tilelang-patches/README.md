# TileLang patches

Fixes and additions to TileLang's ROCm backend, found while writing the kernels in this tree. Each file
is a commit message followed by its diff against upstream TileLang main 6e8f599e; apply them in order.
`results/RESEARCH-LOG.md` describes their tests.

| Patch | Subject |
|---|---|
| `0001-rocm-emit-occupancy-from-annotate_min_blocks_per_sm.patch` | rocm emit occupancy from annotate_min_blocks_per_sm |
| `0002-rocm-fix-8-bit-integer-broadcast.patch` | rocm fix 8 bit integer broadcast |
| `0003-rocm-explain-gemm-limits-on-small-tiles.patch` | rocm explain gemm limits on small tiles |
| `0004-rocm-add-occupancy-report.patch` | rocm add occupancy report |
| `0005-rocm-honour-compile-flags-on-hip.patch` | rocm honour compile flags on hip |
| `0006-rocm-no-workspace-for-reduce-within-a-wavefront.patch` | rocm no workspace for reduce within a wavefront |
| `0007-rocm-keep-staged-fp16-tiles-in-registers.patch` | rocm keep staged fp16 tiles in registers |
| `0008-add-reducer-scale.patch` | add reducer scale |
| `0009-rocm-schedule-dense-gemm-k-loop.patch` | rocm schedule dense gemm k loop |

## Upstream status (9 November 2026)

Patches 0001 to 0008 are prepared as eight pull requests against tile-ai/tilelang; none has been opened yet.
Patch 0009 is held. The AMD kernels in this tree were built and measured with the series applied.

Licence: TileLang is under the MIT License, Copyright (c) Tile-AI (`../../LICENSES/MIT.txt`). These
diffs carry short context lines from TileLang, which stay under that licence; the changes themselves are
offered under Apache-2.0 (`../../REUSE.toml`: `Apache-2.0 AND MIT`).
