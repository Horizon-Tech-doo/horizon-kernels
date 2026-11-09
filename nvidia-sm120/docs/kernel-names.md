# Kernel names

Use names that explain the operation and, when relevant, the optimization.
Keep `horizon_kernels` as the package name. Historical track labels remain useful
for linking benchmarks, scripts, switches and commits; they are not sufficient as
the only names in new reports.

| Existing label | Descriptive name |
|---|---|
| K1 | Speculative draft head optimization |
| K2 | Low-batch matrix multiplication |
| K3 | Native NVFP4 tensor-core matrix multiplication |
| K4 | Persistent decode |
| K4 segment / phase 2 | Persistent projection-and-MLP segment |
| K4 GDN / phase 3a | Persistent gated delta-rule layer |
| K5 | Fused normalization and activation operations |
| G1 | Gated delta-rule decode |
| G2 | Speculative recurrent-state recovery |
| S3 | Output projection quantization |
| FSE | Shared-expert fusion |
| route_align | Expert routing and token alignment |
| ra / hw=2 | Cross-task weight-prefetch pipeline |
