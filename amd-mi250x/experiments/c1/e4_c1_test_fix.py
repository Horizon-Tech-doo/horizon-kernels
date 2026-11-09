import os; F=os.environ.get('FORK','/scratch/project_XXXXXXXXX/tilelang/fork/')+'testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py'
s=open(F).read()
old=s[s.index('@tl.jit'):s.index('@tilelang.testing.requires_rocm')]
new='''@tl.jit(out_idx=[-1], target="hip")
def _annotated(threads, blocks):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=threads) as bx:
            T.annotate_min_blocks_per_sm(blocks)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tl.jit(out_idx=[-1], target="hip")
def _plain(threads):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=threads) as bx:
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


'''
s=s.replace(old,new).replace('_kernel(threads, blocks).get_kernel_source()','_annotated(threads, blocks).get_kernel_source()').replace('_kernel(256, 0).get_kernel_source()','_plain(256).get_kernel_source()')
open(F,'w').write(s)
