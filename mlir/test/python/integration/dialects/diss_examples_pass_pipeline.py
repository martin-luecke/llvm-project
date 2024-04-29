# RUN: %PYTHON %s 2>&1 | FileCheck %s
from __future__ import annotations

from mlir.ir import Module, UnitAttr
from mlir.dialects.transform import any_op_t
from mlir.dialects.transform.extras import *
from mlir.dialects.builtin import module

from mlir.extras.utils import (
    execute,
    run_transform,
    construct_module,
    eraseTransformScript,
    print_module,
)

from mlir.dialects.transform.extras import memrefComplexIR


@print_module
@construct_module
def test_pass_pipeline(module_: Module):
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(module: any_op_t()):
            assert isinstance(module, OpHandle)
            module.normalForm |= {scfIR, funcIR, arithIR, memrefIR, memrefComplexIR}
            print(module.normalForm)

            module = module.expand_strided_metadata()
            print(module.normalForm)
            module = module.lower_affine()
            print(module.normalForm)
            module = module.convert_arith_to_llvm()
            print(module.normalForm)

            module = module.convert_scf_to_cf()
            print(module.normalForm)

            module = module.convert_cf_to_llvm()
            print(module.normalForm)

            module = module.convert_func_to_llvm()
            print(module.normalForm)

            module = module.finalize_memref_to_llvm()
            print(module.normalForm)

            # module = module.convert_arith_to_llvm()
            # module = module.convert_scf_to_cf()
            # module = module.convert_cf_to_llvm()

            module = module.finalize_llvm_conversion()
            print(module.normalForm)


# CHECK-LABEL: TEST: test_lowering_example


@print_module
@construct_module
def test_synthesize_pipeline(module_: Module):
    @module(attrs={"transform.with_named_sequence": UnitAttr.get()})
    def mod():
        @named_sequence("__transform_main", [any_op_t()], [])
        def basic(module: any_op_t()):
            ir_module = module.finalize_llvm_conversion()


if __name__ == "__main__":
    test_pass_pipeline()
