### GDN padding correctness passes; persistent performance regresses

Completed the full six-shape/two-implementation GPU gate:12/12 reference comparisons and12/12 padding reuse checks pass, including all checked intermediates, convolution/SSM state, and two poisoned-buffer graph replays per padding case. Terminal END reports ok_all=true. This verifies the standalone candidate on the listed shapes, not a model integration or a general padding proof.

|Requests x tokens|Separate kernels us|Persistent hw0 us|Persistent hw2 us|
|---|---|---|---|
|1x4|178.87|187.93|188.19|
|2x4|187.79|199.28|198.68|
|8x4|247.98|277.34|278.38|
|4x2|186.77|199.58|199.52|
|16x2|260.12|294.52|295.98|
|32x1|292.37|332.06|334.70|

Paired timing differences are+4.76% to+14.62% (slower), with positive95% bootstrap intervals in every reported comparison. Benchmark uses four rotated layers and30 alternating timing iterations, with synthetic GDN state/inputs. It is not a full production-model benchmark. No tested configuration merits promotion; retain the padding correctness change as a candidate fix and keep separate kernels as the speed baseline. The persistent path still needs performance work.

Evidence: results/gdn_padding_28-09-2026/results.jsonl, validated locally for all expected12 shape/implementation combinations and complete terminal record. The original partial snapshot remains.

