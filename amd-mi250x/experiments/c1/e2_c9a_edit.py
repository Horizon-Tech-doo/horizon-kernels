import os; F=os.environ.get('FORK','/scratch/project_XXXXXXXXX/tilelang/fork/')
def sub(path, old, new):
    s=open(F+path).read(); assert s.count(old)==1, (path, old, s.count(old)); s=s.replace(old,new); open(F+path,'w').write(s)
sub('src/rocm/codegen/codegen_hip.cc','''  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 &&
      lanes == 4) {
    // make_int8x4
    const int64_t *p = as_const_int(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (op->dtype.is_uint()) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }
''','''  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8 &&
      (lanes == 4 || lanes == 8 || lanes == 16)) {
    // Vectors of 8-bit integers are stored as 32-bit words (see PrintType), so
    // broadcast one packed word per 4 lanes. The value is not always constant:
    // LegalizeSafeMemoryAccess binds it to a variable.
    std::ostringstream word;
    word << (op->dtype.is_uint() ? "(uint)" : "(int)");
    if (const int64_t *p = as_const_int(op->value)) {
      int64_t v = *p & 0xFF;
      word << ((v << 24) | (v << 16) | (v << 8) | v);
    } else {
      word << "((uint)(uchar)(" << PrintExpr(op->value) << ") * 0x01010101u)";
    }
    if (lanes == 4) {
      os << word.str();
      return;
    }
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    for (int i = 0; i < lanes / 4; ++i) {
      if (i != 0)
        os << ", ";
      os << word.str();
    }
    os << ')';
    return;
  }
''')
s=open(F+'src/rocm/codegen/codegen_hip.cc').read()
if '#include <sstream>' not in s:
    sub('src/rocm/codegen/codegen_hip.cc','#include <optional>\n','#include <optional>\n#include <sstream>\n')
open(F+'testing/python/amd/test_tilelang_hip_uint8_broadcast.py','w').write('''"""Safe-memory zero else-branch on dynamically indexed uint8 copies (4, 8 and 16 byte lanes)."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
import torch


@tilelang.jit(target="hip")
def _copy_row_tile(cols, threads):
    @T.prim_func
    def main(
        A: T.Tensor((256, cols), T.uint8),
        idx: T.Tensor((1,), T.int32),
        out: T.Tensor((16, cols), T.uint8),
    ):
        with T.Kernel(1, threads=threads):
            tile = T.alloc_shared((16, cols), T.uint8)
            n = idx[0]
            T.copy(A[n * 16, 0], tile)
            for i, j in T.Parallel(16, cols):
                out[i, j] = tile[i, j]

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(("cols", "threads"), [(64, 256), (64, 128), (128, 128)])
def test_uint8_copy_with_safe_memory_else_branch(cols, threads):
    kernel = _copy_row_tile(cols, threads)
    A = torch.randint(1, 255, (256, cols), dtype=torch.uint8, device="cuda")
    out = torch.zeros(16, cols, dtype=torch.uint8, device="cuda")

    idx = torch.tensor([3], dtype=torch.int32, device="cuda")
    kernel(A, idx, out)
    assert torch.equal(out, A[48:64])

    # Past the end of A: the copy must read the safe value, which is zero.
    idx = torch.tensor([16], dtype=torch.int32, device="cuda")
    kernel(A, idx, out)
    assert torch.count_nonzero(out) == 0


if __name__ == "__main__":
    tilelang.testing.main()
''')
