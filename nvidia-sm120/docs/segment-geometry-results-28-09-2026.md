### Generic persistent segment passes seven GPU cases

All seven frozen segment-geometry cases pass exact standalone-reference comparisons, graph replay, counter reset and two trials. Coverage includes legacy H/I/KO5120/17408/6144; alternate4096/14336/4096 at M4 with hw0/1/2 and M32 with hw0/2; and nq16384 with explicitly disabled auxiliary projection. Alternate cases use gate/up split2 versus legacy split1, exercising the corrected gate/up scheduling. Auxiliary width80 is independent of next projection width12288 in the alternate cases.

This establishes execution correctness for the listed segment shapes and scheduling modes, including intermediate comparisons in the harness. It does not establish arbitrary-model support, tensor-parallel integration, end-to-end decode quality, or performance improvement. No candidate deployment. Local artifact completeness checks pass; terminal SEGMENT_GEOMETRY_GPU_PASS7 is present.

Evidence: results/segment_geometry_28-09-2026/results.jsonl; frozen candidate segment-geometry-v1. The tested source is the earlier explicit-geometry candidate, not the later GDN-head changes awaiting their own gate.

