# Applies the C1 source edit to the fork tree on LUMI scratch.
import re
import os; F=os.environ.get('FORK','/scratch/project_XXXXXXXXX/tilelang/fork/')
def sub(path, old, new, count=1):
    s=open(F+path).read(); assert old in s, (path, old); s=s.replace(old,new,count); open(F+path,'w').write(s)

sub('src/cuda/op/builtin.h', '''// Minimum CUDA thread blocks per SM for __launch_bounds__ emission.
static constexpr const char *kMinBlocksPerSM = "tl.min_blocks_per_sm";

''', '')
sub('src/op/builtin.h', '''static constexpr const char *kSafeValueMap = "safe_value_map";
''', '''static constexpr const char *kSafeValueMap = "safe_value_map";

// Minimum thread blocks per SM (CUDA) or compute unit (ROCm) for
// __launch_bounds__ emission.
static constexpr const char *kMinBlocksPerSM = "tl.min_blocks_per_sm";
''')
sub('src/rocm/codegen/codegen_hip.cc', '''        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};
''', '''        threadIdx_z_ext = op->value;
      }
    } else if (op->attr_key == tl::attr::kMinBlocksPerSM) {
      if (const IntImmNode *v = op->value.as<IntImmNode>()) {
        min_blocks_per_cu = v->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
  int64_t min_blocks_per_cu = 0; // 0: not annotated
};
''')
sub('src/rocm/codegen/codegen_hip.cc', '''    if (threadIdx_ext_int->value >= 64) {
      stream << " __launch_bounds__(" << threadIdx_ext_int->value << ")";
    }
''', '''    if (threadIdx_ext_int->value >= 64) {
      stream << " __launch_bounds__(" << threadIdx_ext_int->value;
      if (extractor.min_blocks_per_cu > 0) {
        // HIP's second argument is amdgpu_waves_per_eu, not CUDA's blocks per
        // SM. A block needs ceil(threads / wave size) wavefronts and they
        // share the 4 SIMDs of a compute unit, so N blocks per compute unit
        // need ceil(N * waves_per_block / 4) waves per SIMD.
        constexpr int64_t kSimdsPerCU = 4;
        const int64_t wave_size = tl::TargetRocmGetWarpSize(target_);
        const int64_t waves_per_block =
            (threadIdx_ext_int->value + wave_size - 1) / wave_size;
        stream << ", "
               << (extractor.min_blocks_per_cu * waves_per_block + kSimdsPerCU -
                   1) /
                      kSimdsPerCU;
      }
      stream << ")";
    }
''')
s=open(F+'src/rocm/codegen/codegen_hip.cc').read()
if '"rocm/target_utils.h"' not in s:
    sub('src/rocm/codegen/codegen_hip.cc','#include "rocm/op/builtin.h"\n','#include "rocm/op/builtin.h"\n#include "rocm/target_utils.h"\n')
sub('tilelang/rocm/language/__init__.py','''from tilelang.language.common import __all__ as _COMMON_ALL
''','''from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.annotations import annotate_min_blocks_per_sm  # noqa: F401
''')
sub('tilelang/rocm/language/__init__.py','''__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_ROCM_ALL, *_KERNEL_ALL, *_GEMM_ALL)))''','''__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_ROCM_ALL, *_KERNEL_ALL, *_GEMM_ALL, "annotate_min_blocks_per_sm")))''')
sub('tilelang/language/annotations.py','''    which can improve occupancy at the cost of potentially more register
    spilling.
''','''    which can improve occupancy at the cost of potentially more register
    spilling.

    On HIP, ``n`` is the number of blocks per compute unit. HIP's second
    ``__launch_bounds__`` argument is ``amdgpu_waves_per_eu``, so codegen emits
    ``ceil(n * ceil(threads / wave_size) / 4)``: 2 blocks of 256 threads per
    compute unit gives ``__launch_bounds__(256, 2)``, and 2 blocks of 128
    threads gives ``__launch_bounds__(128, 1)``. Without the annotation HIP
    emits only the first argument.
''')
open(F+'testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py','w').write('''"""T.annotate_min_blocks_per_sm on HIP: blocks per compute unit to waves per EU."""

import pytest
import tilelang as tl
import tilelang.rocm.language as T
import tilelang.testing


@tl.jit(out_idx=[-1], target="hip")
def _kernel(threads, blocks):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=threads) as bx:
            if blocks:
                T.annotate_min_blocks_per_sm(blocks)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("threads", "blocks", "waves_per_eu"),
    [
        (64, 4, 1),
        (128, 2, 1),
        (128, 4, 2),
        (256, 1, 1),
        (256, 2, 2),
        (256, 3, 3),
        (512, 1, 2),
        (512, 2, 4),
        (768, 1, 3),
    ],
)
def test_annotation_maps_blocks_per_cu_to_waves_per_eu(threads, blocks, waves_per_eu):
    src = _kernel(threads, blocks).get_kernel_source()
    assert f"__launch_bounds__({threads}, {waves_per_eu})" in src


@tilelang.testing.requires_rocm
def test_no_annotation_keeps_single_argument():
    src = _kernel(256, 0).get_kernel_source()
    assert "__launch_bounds__(256)" in src


if __name__ == "__main__":
    tilelang.testing.main()
''')
