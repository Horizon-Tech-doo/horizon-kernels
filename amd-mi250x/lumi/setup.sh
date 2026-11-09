#!/usr/bin/env bash
# Reproducible TileLang setup on LUMI (login node is fine: pip + git only, no compile).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$TL_ROOT" "$PIP_CACHE_DIR" "$TL_ROOT/cache"
cd "$TL_ROOT"
# Upstream source for examples/tests (tag that matches the wheel). main is kept for reference.
if [ ! -d src ]; then git clone --recursive -q https://github.com/tile-ai/tilelang.git src; fi
if [ ! -d src-v0.1.15 ]; then git -C src worktree add -q ../src-v0.1.15 v0.1.15; fi
singularity exec -B "$TL_BIND" "$TL_SIF" bash -ec "
  \$WITH_CONDA
  [ -d $TL_ROOT/venv ] || python -m venv --system-site-packages $TL_ROOT/venv
  source $TL_ROOT/venv/bin/activate
  # --no-deps: keep the container's ROCm torch; skip torch-c-dlpack-ext (CUDA-only wheel).
  pip install -q --no-deps tilelang==$TL_VERSION
  pip install -q 'apache-tvm-ffi>=0.1.11,<0.1.13' 'z3-solver>=4.13.0,<4.15.5' cloudpickle ml-dtypes psutil tqdm pytest
  python -c 'import tilelang, torch; print(\"tilelang\", tilelang.__version__, \"torch\", torch.__version__, torch.version.hip)'
"

# --- gfx90a hipcc shim -------------------------------------------------------------------------
# Two ROCm-6.2 incompatibilities in TileLang 0.1.15 HIP templates, fixed by a force-included header:
# (1) reduce.h/scan.h call __builtin_amdgcn_wavefrontsize(), which the
# AMD clang in ROCm 6.2.4 (container) and 6.3.4 (LUMI host) does not provide. gfx90a is always
# wave64, so a wrapper defines it as a macro when (and only when) compiling for gfx90a.
# The prefix mirrors /opt/rocm via symlinks so TileLang's find_hipcc() header check passes.
W=$TL_ROOT/hipcc-wrap
mkdir -p $W/bin
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "for e in /opt/rocm/*; do b=\$(basename \$e); [ \$b = bin ] || ln -sfn \$e $W/\$b; done; for e in /opt/rocm/bin/*; do b=\$(basename \$e); [ \$b = hipcc ] || ln -sfn \$e $W/bin/\$b; done"
cat > $W/gfx90a_wave64_shim.h <<'HDR'
// TileLang-on-LUMI: old AMD clang lacks __builtin_amdgcn_wavefrontsize(); gfx90a is wave64.
#if !__has_builtin(__builtin_amdgcn_wavefrontsize)
#define __builtin_amdgcn_wavefrontsize() 64
#endif
// gfx90a has no FP8 hardware; ROCm 6.2 headers only define the FNUZ FP8 types
// (TileLang's hip_fp8.h otherwise picks the OCP types that need newer ROCm headers).
#ifndef TILELANG_FP8_E4M3_VARIANT
#define TILELANG_FP8_E4M3_VARIANT 1
#endif
#ifndef TILELANG_FP8_E5M2_VARIANT
#define TILELANG_FP8_E5M2_VARIANT 1
#endif
HDR
cat > $W/bin/hipcc <<WRAP
#!/bin/bash
# TileLang-on-LUMI shim, see setup.sh. Only touches gfx90a compiles.
for a in "\$@"; do case "\$a" in *gfx90a*) exec /opt/rocm/bin/hipcc -include $W/gfx90a_wave64_shim.h "\$@";; esac; done
exec /opt/rocm/bin/hipcc "\$@"
WRAP
chmod +x $W/bin/hipcc
echo "hipcc shim: $W/bin/hipcc (export ROCM_PATH=$W in tl_run.sh)"
