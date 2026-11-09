R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I. -S -x hip keep/h9_25664c8d1p32_8k.cu -o keep/h9_25664c8d1p32_8k.s 2>&1 | grep " error"
