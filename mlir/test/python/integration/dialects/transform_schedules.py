# RUN: export MLIR_RUNNER_UTILS=/Users/martin/development/llvm-project/build/lib/libmlir_runner_utils.dylib &&\
# RUN: export MLIR_C_RUNNER_UTILS=/Users/martin/development/llvm-project/build/lib/libmlir_c_runner_utils.dylib &&\
# RUN: %PYTHON %s 2>&1 | FileCheck %s

from typing import Callable
from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, arith, linalg, memref, tensor
from mlir.dialects.transform import (
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
)
from mlir.dialects.transform.structured import *
from mlir.dialects.builtin import module, ModuleOp
from mlir.dialects.transform.extras import *
from mlir.execution_engine import ExecutionEngine
from mlir.extras import types as T
from mlir.extras.utils import (
    execute,
    lower,
    print_module,
    run_transform,
    construct_module,
    eraseTransformScript,
)


@execute(inputs=[[11.0, 12.0], [[25.0], [12.0]], [[0.0, 0.0], [0.0, 0.0]]])
@lower
@run_transform
@construct_module
def test_tiling(module_: Module):
    test_callback = func.FuncOp(
        "customCallback", ([T.memref(T.f32())], []), visibility="private"
    )
    print_fun = func.FuncOp(
        "printMemrefF32", ([T.memref(T.f32())], []), visibility="private"
    )

    @func.func(
        T.memref(2, 1, T.f32()),
        T.memref(1, 2, T.f32()),
        T.memref(2, 2, T.f32()),
    )
    def matmul_signed_on_buffers(lhs: Value, rhs: Value, out: Value):
        linalg.matmul(lhs, rhs, outs=[out])
        # func.CallOp(print, [memref.CastOp(T.memref(T.f32()), out)])

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            matmul = target.match_ops("linalg.batch_matmul")
            loops = matmul.tile(loop=TileLoopKind.FORALL, tile_sizes=[8]).loops


# CHECK: 275.
# CHECK-SAME: 132
# CHECK-NEXT: 300
# CHECK-SAME: 144


if __name__ == "__main__":
    test_tiling()

# @construct_and_print_in_module
# def test_loop_noop(module_: Module):
#     @func.func(
#         tensor(32, 32, 32, element_type=T.f32()),
#         tensor(32, 32, 32, element_type=T.f32()),
#         tensor(32, 32, 32, T.f32()),
#     )
#     def matmul_signed_on_buffers(lhs: Value, rhs: Value, out: Value) -> Value:
#         return linalg.batch_matmul(lhs, rhs, outs=[out])

#     @func.func()
#     def loop_unroll_op():
#         for i in scf.for_(0, 42, 5):
#             v = arith.addi(i, i)
#             scf.yield_([])

#     @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
#     def mod():
#         @named_sequence("__transform_main", [any_op_t()], [])
#         def basic(target: any_op_t()):
#             matmul = target.match_ops("linalg.batch_matmul")
#             loops = matmul.tile(loop=TileLoopKind.FORALL, tile_sizes=[8]).loops
#             # for loop in loops:
#             #     loop_unroll(loop, 2)

#     print(module_)

#     pm = PassManager.parse("builtin.module(transform-interpreter)")
#     pm.run(module_.operation)

#     print(module_)
