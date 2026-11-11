"""Check that vLLM itself loads the tuned file: run with VLLM_TUNED_CONFIG_FOLDER set to the folder that holds it.
For every T raced, try_get_optimal_moe_config must return the JSON entry with the nearest key. argv: tuned.json T,T,..."""
import json, os, sys, torch
from vllm.model_executor.layers.fused_moe import fused_moe as FM
H, I, E, topk = 2048, 512, 256, 8
J = {int(k): v for k, v in json.load(open(sys.argv[1])).items() if k != "triton_version"}
dt = FM._get_config_dtype_str(dtype=torch.float16)
name = FM.get_config_file_name(E, I, dt)
got = FM.get_moe_configs(E, I, dt)
bad = []
for T in [int(t) for t in sys.argv[2].split(",")]:
    c = FM.try_get_optimal_moe_config((E, 2 * I, H), (E, H, I), topk, dt, T, block_shape=None)
    if c != J[min(J, key=lambda k: abs(k - T))]:
        bad.append(T)
print(json.dumps({"kind": "cfgcheck", "jobid": os.environ.get("SLURM_JOB_ID"), "folder_set": bool(os.environ.get("VLLM_TUNED_CONFIG_FOLDER")),
                  "file_name_vllm_expects": name, "file_name_matches": name == os.path.basename(sys.argv[1]), "loaded_keys": sorted(got) if got else None,
                  "mismatch_T": bad, "ok": bool(got) and not bad and name == os.path.basename(sys.argv[1])}), flush=True)
