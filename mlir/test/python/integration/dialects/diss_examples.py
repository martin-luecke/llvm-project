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
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import named_sequence, apply_patterns
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


@print_module
@run_transform
@construct_module
def test_commute(module_: Module):
    @func.func(T.f32(), T.f32())
    def addition(lhs: Value, rhs: Value):
        return arith.AddFOp(lhs, rhs)

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            binary_op = target.match_ops("arith.addf")
            transform.commute(any_op_t(), binary_op)
            transform.print_(target=target)


# CHECK: @addition(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32)
# CHECK:  %[[VAL_2:.*]] = arith.addf %[[VAL_1]], %[[VAL_0]] : f32
# CHECK:  return %[[VAL_2]] : f32

if __name__ == "__main__":
    test_commute()
