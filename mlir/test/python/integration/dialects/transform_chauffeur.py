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
    run_transform,
    construct_module,
    eraseTransformScript,
    print_module,
)


@execute(inputs=[[11.0, 12.0], [[25.0], [12.0]], [[0.0, 0.0], [0.0, 0.0]]])
@run_transform
@construct_module
def test_tiling(module_: Module):
    @module
    def payload():
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

        matmul_signed_on_buffers.func_op.attributes[
            "llvm.emit_c_interface"
        ] = UnitAttr.get()
        test_callback.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        print_fun.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(module: any_op_t()):
            matmul = module.match_ops("linalg.matmul")
            func_op = matmul.get_parent("func.func")
            ir_module = func_op.get_parent("builtin.module")
            tiled_op, loops = matmul.tile(loop=TileLoopKind.FORALL, tile_sizes=[8])

            ir_module = ir_module.apply_pass("lower-affine")
            ir_module = ir_module.apply_pass("convert-linalg-to-loops")
            ir_module = ir_module.apply_pass("convert-scf-to-cf")
            ir_module = ir_module.apply_pass("expand-strided-metadata")
            ir_module = ir_module.apply_pass("lower-affine")
            ir_module = ir_module.apply_pass("convert-arith-to-llvm")
            ir_module = ir_module.apply_pass("convert-func-to-llvm")
            ir_module = ir_module.apply_pass("finalize-memref-to-llvm")
            ir_module = ir_module.apply_pass("reconcile-unrealized-casts")


# CHECK: 275.
# CHECK-SAME: 132
# CHECK-NEXT: 300
# CHECK-SAME: 144

if __name__ == "__main__":
    test_tiling()
