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
from mlir.dialects.builtin import module, ModuleOp


def construct_and_print_in_module(f: Callable[[ModuleOp], None]):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


@construct_and_print_in_module
def test_tiling(module_: ModuleOp):
    @func.func()
    def payload_ir():
        for i in scf.for_(0, 42, 1):
            v = arith.addi(i, i)
            # linalg.batch_matmul()
            scf.yield_([])

    @func.func(
        MemRefType.get([32, 32], T.f32()),
        MemRefType.get([32, 32], T.f32()),
        MemRefType.get([32, 32], T.f32()),
    )
    def matmul_signed_on_buffers(lhs: Value, rhs: Value, out: Value):
        linalg.matmul(lhs, rhs, outs=[out])

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(target: any_op_t()):
            m = structured_match(any_op_t(), target, ops=["arith.addi"])
            loop = get_parent_op(pdl.op_t(), m, op_name="scf.for")

            loop_unroll(loop, 2)

    print(module_)

    pm = PassManager.parse("builtin.module(transform-interpreter)")
    pm.run(module_.operation)

    print(module_)
