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


# CHECK-LABEL: TEST: test_commute
# CHECK: @addition(%[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32)
# CHECK:  %[[VAL_2:.*]] = arith.addf %[[VAL_1]], %[[VAL_0]] : f32
# CHECK:  return %[[VAL_2]] : f32


@execute()
@print_module
@run_transform
@construct_module
def test_lowering_example(module_: Module):
    @module
    def payload():
        print_fun = func.FuncOp("printF32", ([T.f32()], []), visibility="private")

        @func.func()
        def entry():
            c0 = arith.constant(T.index(), 0)
            c1 = arith.constant(T.index(), 1)
            c4 = arith.constant(T.index(), 4)
            c5 = arith.constant(T.index(), 5)
            cf4 = arith.constant(T.f32(), 4.5)

            index_type = IndexType.get()
            f32 = F32Type.get()
            memref_type = MemRefType.get([1024], f32)
            c1 = arith.ConstantOp(index_type, 1)
            # CHECK: %[[C2:.*]] = arith.constant 2 : index
            c2 = arith.ConstantOp(index_type, 2)
            c3 = arith.ConstantOp(index_type, 3)
            c9 = arith.ConstantOp(index_type, 9)

            #### Affine experiment from here
            ac0 = AffineConstantExpr.get(2)
            d0 = AffineDimExpr.get(0)
            d1 = AffineDimExpr.get(1)
            s0 = AffineSymbolExpr.get(0)
            lb = AffineMap.get(1, 1, [ac0, d0 + s0])
            ub = AffineMap.get(2, 0, [d0 - 2, 32 * d1])
            sum_0 = arith.ConstantOp(f32, 0.0)

            A = memref.AllocOp(
                T.memref(64, 64, element_type=T.f32()),
                dynamicSizes=[],
                symbolOperands=[],
            )
            sum = affine.AffineForOp(
                lb,
                ub,
                2,
                iter_args=[sum_0],
                lower_bound_operands=[c2, c3],
                upper_bound_operands=[c9, c1],
            )
            with InsertionPoint(sum.body):
                # CHECK: %[[TMP:.*]] = memref.load %[[BUFFER]][%[[INDVAR]]] : memref<1024xf32>
                tmp = memref.LoadOp(A, [sum.induction_variable, sum.induction_variable])
                sum_next = arith.AddFOp(sum.inner_iter_args[0], tmp)
                memref.StoreOp(
                    sum_next, A, [sum.induction_variable, sum.induction_variable]
                )
                affine.AffineYieldOp([sum_next])

            ### To here

            a = llvm.ConstantOp(T.f32(), ir.FloatAttr.get(T.f32(), 2))
            b = llvm.ConstantOp(T.f32(), ir.FloatAttr.get(T.f32(), 20))
            a_ = memref.subview(
                T.memref(
                    4,
                    4,
                    element_type=T.f32(),
                    layout=StridedLayoutAttr.get(0, [64, 1]),
                ),
                source=A,
                static_offsets=[0, 0],
                static_sizes=[4, 4],
                static_strides=[1, 1],
                offsets=[],
                sizes=[],
                strides=[],
            )
            for_ = scf.ForOp(c0, c4, c1)
            with InsertionPoint(for_.body):
                i = for_.induction_variable
                memref.store(cf4, a_, [i, i])
                scf.YieldOp(for_.inner_iter_args)
            res = llvm.fadd(a, b)
            res2 = memref.load(A, indices=[c0, c0])
            func.CallOp(print_fun, [res])
            func.CallOp(print_fun, [res2])
            return res2

        entry.func_op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(module: any_op_t()):
            # module.print()
            specific_op = module.match_ops("func.call")
            func_op = specific_op.get_parent("func.func")
            ir_module = func_op.get_parent("builtin.module")
            ir_module = ir_module.apply_pass("convert-scf-to-cf")
            ir_module = ir_module.apply_pass("convert-func-to-llvm")
            ir_module = ir_module.apply_pass("expand-strided-metadata")
            ir_module = ir_module.apply_pass("finalize-memref-to-llvm")
            ir_module = ir_module.apply_pass("reconcile-unrealized-casts")


# CHECK-LABEL: TEST: test_lowering_example

if __name__ == "__main__":
    # test_commute()
    test_lowering_example()
