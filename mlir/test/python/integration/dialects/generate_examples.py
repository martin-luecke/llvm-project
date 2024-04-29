# RUN: %PYTHON %s 2>&1 | FileCheck %s

from typing import Callable
from mlir.passmanager import PassManager
from mlir.ir import Context, Location, Module, InsertionPoint, UnitAttr
from mlir.dialects import scf, pdl, func, affine, arith, linalg, memref, tensor, llvm
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
from numpy import complex_


def generate_memref_example_func(
    print_fun: func.FuncOp, name: str = "function", complex_memref: bool = True
) -> func.FuncOp:

    @func.func(T.index(), name=name)
    def function(idx: Value):
        c0 = arith.constant(T.index(), 0)
        c1 = arith.constant(T.index(), 1)
        c4 = arith.constant(T.index(), 4)
        c5 = arith.constant(T.index(), 5)
        cf4 = arith.constant(T.f32(), 4.5)

        index_type = IndexType.get()
        f32 = F32Type.get()
        memref_type = MemRefType.get([1024], f32)
        c1 = arith.ConstantOp(index_type, 1)
        c2 = arith.ConstantOp(index_type, 2)
        c3 = arith.ConstantOp(index_type, 3)
        c9 = arith.ConstantOp(index_type, 9)

        A = memref.alloc(
            T.memref(64, 64, element_type=T.f32()),
            dynamic_sizes=[],
            symbol_operands=[],
        )
        a = llvm.ConstantOp(T.f32(), ir.FloatAttr.get(T.f32(), 2))
        b = llvm.ConstantOp(T.f32(), ir.FloatAttr.get(T.f32(), 20))
        if complex_memref:
            a_ = memref.subview(
                source=A,
                offsets=[idx, idx],
                sizes=[4, 4],
                strides=[1, 1],
            )
        else:
            a_ = memref.subview(
                source=A,
                offsets=[0, 0],
                sizes=[4, 4],
                strides=[1, 1],
            )
        for_ = scf.ForOp(c0, c4, c1)
        with InsertionPoint(for_.body):
            i = for_.induction_variable
            memref.store(cf4, a_, [i, i])
            scf.YieldOp(for_.inner_iter_args)
        res2 = memref.load(A, indices=[c0, c0])
        func.CallOp(print_fun, [res2])
        return res2

    function.func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    return function.func_op


@print_module
@construct_module
def benchmark_lower_affine(module_: Module):
    print_fun = func.FuncOp("printF32", ([T.f32()], []), visibility="private")

    num_complex_funcs = 100000
    num_simple_funcs = 1

    funcs: list[func.FuncOp] = []
    for i in range(num_complex_funcs):
        funcs.append(
            generate_memref_example_func(
                print_fun, f"function_{i}", complex_memref=True
            )
        )

    for i in range(num_complex_funcs, num_complex_funcs + num_simple_funcs):
        funcs.append(
            generate_memref_example_func(
                print_fun, f"function_{i}", complex_memref=False
            )
        )

    @func.func()
    def entry():
        c1 = arith.ConstantOp(IndexType.get(), 0)
        for function in funcs:
            func_call = func.CallOp(function, [c1])

    entry.func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()


if __name__ == "__main__":
    # test_commute()
    # test_lowering_example()
    benchmark_lower_affine()
