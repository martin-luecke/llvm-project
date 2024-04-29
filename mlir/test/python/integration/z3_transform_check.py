from lib2to3.pytree import convert
from typing import Any, Sequence
from z3 import Distinct, Bool, Int, Function, Solver, IntSort, Real, solve
import z3
from mlir import ir
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    UnitAttr,
    Value,
    IndexType,
    F32Type,
    MemRefType,
)
from mlir.dialects.builtin import module, ModuleOp
from mlir.dialects import (
    scf,
    pdl,
    func,
    affine,
    arith,
    linalg,
    memref,
    tensor,
    llvm,
    cf,
)
from mlir.dialects.transform.extras import memrefComplexIR
from mlir.extras import types as T
from mlir.extras.utils import (
    execute,
    run_transform,
    construct_module,
    eraseTransformScript,
    print_module,
)


@construct_module
def get_example_module(module_: Module):
    @module
    def payload():
        print_fun = func.FuncOp("printF32", ([T.f32()], []), visibility="private")

        @func.func(T.memref(1, T.index()))
        def entry(idx_arr: Value):

            c0 = arith.constant(T.index(), 0)
            idx = memref.load(idx_arr, indices=[c0])
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
            complex_memref = True
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
            res = llvm.fadd(a, b)
            res2 = memref.load(A, indices=[c0, c0])
            func.CallOp(print_fun, [res])
            func.CallOp(print_fun, [res2])
            return res2


def to_index(value: Any):
    if value in all_dialects:
        return all_dialects.index(value)
    elif value in all_normalforms:
        return len(all_dialects) + all_normalforms.index(value)
    else:
        raise ValueError("Invalid value")


def to_value(index: int):
    if index < len(all_dialects):
        return all_dialects[index]
    else:
        return all_normalforms[index - len(all_dialects)]


# Config
all_dialects = [arith, linalg, memref, tensor, llvm, scf, cf, pdl, func, affine]
all_normalforms = [memrefComplexIR]
num_transforms = 100

# Implementation
properties = [False for _ in range(len(all_dialects) + len(all_normalforms))]
# property_sequence = [properties for _ in range(num_transforms)]
transform_index = 0

property_sequence = [
    [Bool("prop_%s%s" % (i, j)) for i in range(len(properties))]
    for j in range(num_transforms)
]


def print_properties(properties: Sequence[bool]):
    for i in range(len(properties)):
        try:
            if properties[i]:
                print(to_value(i).__name__.split(".")[-1], end="")
                if i < len(properties) - 1:
                    print(", ", end="")
        except:
            return
    print()


def enforce_state(values: Sequence[Any]):
    global transform_index
    values = [to_index(value) for value in values]
    constraints = []
    for i in range(len(property_sequence[transform_index])):
        if i in values:
            constraints.append(property_sequence[transform_index][i] == True)
        else:
            constraints.append(property_sequence[transform_index][i] == False)
    transform_index = transform_index + 1
    return constraints


def assumed_state(values: Sequence[Any]):
    global transform_index
    values = [to_index(value) for value in values]
    constraints = []
    for i in range(len(property_sequence[transform_index - 1])):
        if i in values:
            constraints.append(property_sequence[transform_index - 1][i] == True)
        else:
            constraints.append(property_sequence[transform_index - 1][i] == False)
    transform_index = transform_index + 1
    return constraints


def new_transform(name: str, ins: Sequence[Any], outs: Sequence[Any]):
    def impl():
        # make sure I add constraints for the existing, unrelated states as well.
        global transform_index
        print(to_index(ins[0]))
        input_constraints = [
            property_sequence[transform_index - 1][to_index(_in)] == True for _in in ins
        ]
        in_indexes = [to_index(_in) for _in in ins]
        out_indexes = [to_index(out) for out in outs]
        output_constraints = []
        for i in range(len(properties)):
            if i in out_indexes:
                output_constraints.append(property_sequence[transform_index][i] == True)
            elif i in in_indexes:
                output_constraints.append(
                    property_sequence[transform_index][i] == False
                )
            else:
                output_constraints.append(
                    property_sequence[transform_index][i]
                    == property_sequence[transform_index - 1][i]
                )
        # property_sequence[transform_index][to_index(_out)] == True for _out in outs
        transform_index = transform_index + 1
        return input_constraints + output_constraints

    return impl


convert_arith_to_llvm = new_transform("convert_arith_to_llvm", [arith], [llvm])
convert_scf_to_cf = new_transform("convert_scf_to_cf", [scf], [cf])
convert_cf_to_llvm = new_transform("convert_cf_to_llvm", [cf], [llvm])
lower_affine = new_transform("lower_affine", [affine], [arith, scf])
convert_func_to_llvm = new_transform("convert_func_to_llvm", [func], [llvm])
expand_strided_metadata = new_transform(
    "expand_strided_metadata", [memref], [memrefComplexIR, affine]
)
finalize_memref_to_llvm = new_transform(
    "finalize_memref_to_llvm", [memrefComplexIR, memref], [llvm]
)
# def finalize_convert_to_llvm():
#     def impl():


#     return impl

if __name__ == "__main__":
    module = get_example_module()
    print(module)
    # for Dialect in module.context.dialects:
    #     print(Dialect)
    solver = z3.Solver()
    # initial state:
    solver.add(enforce_state([arith, scf, func, memref]))
    solver.add(convert_arith_to_llvm())
    solver.add(convert_scf_to_cf())
    solver.add(convert_func_to_llvm())
    solver.add(expand_strided_metadata())
    solver.add(assumed_state([llvm, cf, affine, memrefComplexIR]))

    if solver.check() == z3.sat:
        model = solver.model()
        eval = [
            [model.eval(property_sequence[j][i]) for i in range(len(properties))]
            for j in range(num_transforms)
        ]
        for i in range(num_transforms):
            pass
            # print(eval[i])
            # print_properties(eval[i])
    else:
        print("failed to solve")

    # x = Int("x")
    # y = Int("y")
    # f = Function("f", IntSort(), IntSort())
    # solve(f(f(x)) == x, f(x) == y, x != y)

    # What I eventually want:
    # solve(finalize_memref_to_llvm(convert_func_to_llvm(convert_cf_to_llvm)))
