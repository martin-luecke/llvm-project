from ast import mod
from calendar import c
from dataclasses import field
from functools import wraps
import os
from time import sleep
from typing import Any, Callable, List, Optional, Sequence, Union
from mlir.runtime.np_to_memref import get_unranked_memref_descriptor

# from sys import devnull

from ..execution_engine import ExecutionEngine, ctypes
from ..ir import Module, Context, Location, InsertionPoint, UnitAttr
import numpy as np
from ..passmanager import PassManager
from ..runtime import (
    get_ranked_memref_descriptor,
    ranked_memref_to_numpy,
    make_nd_memref_descriptor,
)
from ..dialects import func
from numpy.typing import NDArray, ArrayLike, DTypeLike


def bufferize(module: Module) -> Module:
    pm = PassManager.parse(
        r"""builtin.module(
                one-shot-bufferize{bufferize-function-boundaries},
                expand-realloc,
                canonicalize,
                ownership-based-buffer-deallocation,
                canonicalize,
                buffer-deallocation-simplification,
                bufferization-lower-deallocations,
                cse,
                canonicalize,
                func.func(finalizing-bufferize),
                convert-bufferization-to-memref
        )"""
    )
    pm.run(module.operation)
    return module


def eraseTransformScript(module: Module) -> Module:
    for op in module.operation.regions[0].blocks[0].operations:
        if (
            op.name == "builtin.module"
            and "transform.with_named_sequence" in op.attributes
        ):
            op.erase()
        elif op.name == "builtin.module":
            eraseTransformScript(op.operation)
    return module


def flatten_module(module: Module) -> Module:
    for op in module.operation.regions[0].blocks[0].operations:
        if op.name == "builtin.module":
            for nested_op in op.operation.regions[0].blocks[0].operations:
                nested_op.operation.move_after(
                    module.operation.regions[0].blocks[0].operations[0]
                )

            # op.operation.regions[0].blocks[0].append_to(module.operation.regions[0])
            op.erase()
    return module


def lowerLinalg(module: Module) -> Module:
    pm = PassManager.parse("builtin.module(convert-linalg-to-loops,convert-scf-to-cf)")
    pm.run(module.operation)
    return module


def lowerToLLVM(module: Module) -> Module:
    # add emit_c_interface attribute to func
    for op in module.operation.regions[0].blocks[0].operations:
        if isinstance(op, func.FuncOp):
            op.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    pm = PassManager.parse(
        "builtin.module(convert-complex-to-llvm,convert-func-to-llvm,finalize-memref-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


@ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(
        # get_ranked_memref_descriptor(
        #     np.array([[11.0, 12.0], [11.0, 12.0]]).astype(np.float32)
        # ).__class__,
        make_nd_memref_descriptor(2, np.ctypeslib.as_ctypes_type(np.float32)),
        # get_unranked_memref_descriptor(
        #     np.array([[11.0, 12.0], [11.0, 12.0]]).astype(np.float32)
        # ).__class__,
    ),
)
def callback(any_memref):
    print("Inside Callback: ")
    arr = ranked_memref_to_numpy(any_memref)
    print(arr)


def get_util_libaries() -> List[str]:
    c_runner_utils = os.getenv("MLIR_C_RUNNER_UTILS", "")
    assert os.path.exists(c_runner_utils), (
        f"{c_runner_utils} does not exist."
        f" Please pass a valid value for"
        f" MLIR_C_RUNNER_UTILS environment variable."
    )
    runner_utils = os.getenv("MLIR_RUNNER_UTILS", "")
    assert os.path.exists(runner_utils), (
        f"{runner_utils} does not exist."
        f" Please pass a valid value for MLIR_RUNNER_UTILS"
        f" environment variable."
    )
    return [c_runner_utils, runner_utils]


def lower(f: Callable[[], Module]) -> Callable[[], Module]:
    def wrapped():
        module = f()
        with module.context:
            bufferize(module)
            lowerLinalg(module)
            lowerToLLVM(module)
        return module

    return wrapped


def execute(
    inputs: Union[Sequence[ArrayLike], None] = None,
    dtype: DTypeLike = np.float32,
    print_results: bool = False,
) -> Callable[[], Callable[[], Module]]:
    def outer(f: Callable[[], Module]) -> Callable[[], Module]:
        def wrapped():
            nonlocal inputs
            if inputs is None:
                inputs = []
                input_ptrs = [ctypes.pointer(ctypes.pointer(ctypes.c_void_p()))]
            else:
                inputs = [np.array(input).astype(dtype) for input in inputs]
                input_ptrs = [
                    ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(arg)))
                    for arg in inputs
                ]
            module = f()
            eraseTransformScript(module)
            flatten_module(module)
            with module.context:
                execution_engine = ExecutionEngine(
                    module, shared_libs=get_util_libaries(), opt_level=2
                )

                execution_engine.register_runtime("customCallback", callback)
                execution_engine.invoke("entry", *input_ptrs)
                if print_results:
                    print(f"args: \n{[input for input in inputs]}")

            return module

        return wrapped

    @wraps(outer)
    def maybe_no_args(f: Callable[[], Module]) -> Callable[[], Module]:
        """Not strictly necessary anymore as execute always takes args now."""
        # if maybe_f:
        #     return outer(maybe_f)()
        # else:
        return outer(f)

    return maybe_no_args


def run_transform(f: Callable[[], Module]) -> Callable[[], Module]:
    def wrapped():
        module = f()
        with module.context:
            pm = PassManager.parse("builtin.module(transform-interpreter)")
            pm.run(module.operation)
            return module

    return wrapped


def print_module(f: Callable[[], Module]) -> Callable[[], Module]:
    def wrapped():
        module = f()
        print(module)
        return module

    return wrapped


def construct_module(
    f: Callable[[Module], None], print_name: bool = True
) -> Callable[[], Module]:
    def wrapped():
        if print_name:
            print("\nTEST:", f.__name__)
        with Context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f(module)
            return module

    return wrapped
