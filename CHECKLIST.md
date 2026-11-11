# Checklist: before quoting a kernel-against-vendor ratio

Apply before publishing any claim of the form "kernel X runs at r times vendor library Y on device Z".
32 items in six groups. The letter and number in brackets is the pitfall each item comes from
(Table "Thirty-two pitfalls" in the paper, with the failure measured for each).

## Baseline

1. Compute the reference in float64 and report the vendor's error against it beside yours. [A1]
2. Start the real runtime with real weights; record the kernel it dispatches, its arguments, data type, page size and layout; reproduce exactly that in the lab. [A2, A3]
3. Prove in the lab that the intended vendor path ran. [A2]
4. Check that the vendor's tuning covers your device; if not, tune it with the vendor's tool or label the baseline "untuned default" in the caption. [A4]
5. Benchmark the form that ships, and count calls into your kernel in the integration. [A5]
6. Run both arms in one process, one allocation, one software stack; mark every cross-process ratio. [A6]

## Noise

7. Add an identical-code control arm to every race and compute a noise bound per job from it. [B1]
8. Use at least five timed cycles per arm per cell; never decide on two. [B2]
9. Keep every per-round value; rerun disturbed shapes in a fresh job and say so. [B3]
10. Rotate arm order each cycle and reverse it in a second job; read the vendor before and after your arm. [B4]
11. Report medians of rounds; show fastest-call values only for both arms and beside the median. [B5]
12. Compare arms only on the same buffers in the same process; report the range over all reads. [B6]
13. Write the keep rule, the cells and the reference arm into a file before the race. [B7]

## Hardware state

14. Validate any clock or power counter against an independent probe before dividing by it; otherwise quote the share of nominal peak only. [C1]
15. Sweep the relative address offset of the operands once per kernel family; state placed or unplaced. [C2]
16. State the load condition (one device or all, capped or not) with every number. [C3]
17. Probe telemetry between arms, never during them. [C4]

## Inputs and call path

18. Race on several input kinds, including real captured inputs, and require the win on every kind; check the error on the real inputs as well. [D1]
19. Use random normal data by default; never fixed-exponent or constant data. [D2]
20. Time both arms under graph replay if the runtime replays graphs; label eager numbers as call-path results. [D3]
21. Warm up every arm at every shape, log a compile counter per timed call, and report compile cost separately. [D4]
22. Assert that both arms return the same shape, strides and data type. [D5]

## Correctness and identity

23. Check float64 error, non-finite count and repeat-launch equality at every dispatched shape, including zero-length and padded rows. [E1, E5]
24. State how many dispatched shapes are timed and how many are interpolated. [E1]
25. Freeze a per-job copy of every module, print its hash in the job, and never edit a file a running job imports. [E2]
26. Race the final landed file through its public entry point and confirm that timed kernels were cache hits. [E3]
27. Hash each arm's generated code: variants must differ, the control must not. [E4]
28. Repeat the race on each compiler that ships. [E6]

## Reporting

29. Give numerator, denominator, quantity and direction in every caption. [F1]
30. Name the unit of every count (cell, job, reading, process) and the code hash behind every range. [F2]
31. Count result records per arm and cell; do not trust the scheduler's exit state. [F3]
32. Label statements observed, computed, inferred or hypothesis, and keep dated corrections. [F4]

Note on items 18 and 19: random normal data is the default for timing. For correctness it is not enough: on AMD gfx90a
the fp16 matrix-core instruction returns zero for subnormal operands, which random inputs did not exercise and real
activations did.

## How to cite

H. Abdagic, "Benchmarking TileLang Kernels Against Vendor Libraries on AMD MI250X: Method, Checklist and Findings",
2026. Repository: https://github.com/Horizon-Tech-doo/horizon-kernels. An arXiv identifier will be added
when the paper is posted.
