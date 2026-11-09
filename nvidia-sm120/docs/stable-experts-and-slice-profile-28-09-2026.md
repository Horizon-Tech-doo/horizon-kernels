# Stable experts and persistent scheduling profile

### Stable expert execution passes; persistent profile identifies delayed downstream starts

Corrected stable alignment passes all15 cases (five geometries, spread/skew/invalid IDs),20 repeats and three graph replays with changed inputs. Real expert execution passes all9 combinations of layers3/7/11 and M32/128/256. Fresh GPU alignment reaches Marlin on every checked invocation; gate/up, down and final outputs match CPU-canonical ordering bitwise, including20 repeats and three original/reversed/original graph replays. This establishes tested operator determinism and integration, not full-model quality, stock-order equivalence, speed or deployment eligibility. MoE stays undeployed.

Persistent task profile completed12 samples with48 raw per-layer traces; both M4/32 preserve exact outputs/intermediates, counter reset and graph replay. Median instrumented per-layer time original/slice: M4 171.520/194.816us; M32 203.016/268.808us. Instrumented behavior agrees directionally with the earlier uninstrumented regression.

Median of per-layer task medians, original→slice (microseconds): M32 down setup/prefetch/wait11.120→77.024 while body/signaling36.880→36.144; norm1 pre-body53.600→131.280; qkvz pre-body46.960→92.144 while body31.808→31.808. M4 down pre-body7.024→55.680 and norm1 43.152→85.168. Gate/up bodies are faster under slice order, yet total completion is slower. This points to task placement/dependency scheduling as a useful next hypothesis; the combined pre-body metric cannot isolate pure waiting or prove cache/occupancy causation. Retain original order, investigate scheduling delays before extending the megakernel around slice order.

Evidence: results/stable_align_28-09-2026/v2-results.jsonl; stable_experts_28-09-2026/results.jsonl; slice_profile_28-09-2026/results.jsonl and compressed raw timestamps. Local completeness checks pass. No deployment or config change.
