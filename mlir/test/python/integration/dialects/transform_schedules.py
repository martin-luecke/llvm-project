# RUN: %PYTHON %s 2>&1 | FileCheck %s

from typing import Callable
from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, arith, linalg
from mlir.dialects.transform import (
    get_parent_op,
    apply_patterns_canonicalization,
    apply_cse,
    any_op_t,
)
from mlir.dialects.transform.structured import *
from mlir.dialects.transform.loop import loop_unroll
from mlir.dialects.transform.extras import named_sequence, apply_patterns
from mlir.extras import types as T
from mlir.extras.types import memref, tensor
from mlir.dialects.builtin import module, ModuleOp
from mlir.dialects.transform.extras import *


def construct_and_print_in_module(f: Callable[[Module], None]):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


@construct_and_print_in_module
def test_tiling(module_: Module):
    @func.func(
        tensor(32, 32, 32, element_type=T.f32()),
        tensor(32, 32, 32, element_type=T.f32()),
        tensor(32, 32, 32, T.f32()),
    )
    def matmul_signed_on_buffers(lhs: Value, rhs: Value, out: Value) -> Value:
        return linalg.batch_matmul(lhs, rhs, outs=[out])

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            matmul = target.match_ops("linalg.batch_matmul")
            loops = matmul.tile(loop=TileLoopKind.FORALL, tile_sizes=[8]).loops
            # for loop in loops:
            #     loop_unroll(loop, 2)

    print(module_)

    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print(module_)


@construct_and_print_in_module
def test_loop_noop(module_: Module):
    @func.func(
        tensor(32, 32, 32, element_type=T.f32()),
        tensor(32, 32, 32, element_type=T.f32()),
        tensor(32, 32, 32, T.f32()),
    )
    def matmul_signed_on_buffers(lhs: Value, rhs: Value, out: Value) -> Value:
        return linalg.batch_matmul(lhs, rhs, outs=[out])

    @func.func()
    def loop_unroll_op():
        for i in forall(0, 42, 5):
            v = arith.addi(i, i)
            scf.yield_([])

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            matmul = target.match_ops("linalg.batch_matmul")
            loops = matmul.tile(loop=TileLoopKind.FORALL, tile_sizes=[8]).loops
            # for loop in loops:
            #     loop_unroll(loop, 2)

    print(module_)

    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print(module_)
