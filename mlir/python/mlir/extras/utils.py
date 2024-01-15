from ast import Call
from functools import wraps
from typing import Callable, Optional
from ..execution_engine import ExecutionEngine
from ..ir import Module, Context, Location, InsertionPoint
import numpy as np
from ..passmanager import PassManager
from ..runtime import ctypes, get_ranked_memref_descriptor, ranked_memref_to_numpy


def bufferize(module: Module) -> Module:
    pm = PassManager.parse(
        r"""builtin.module(
                one-shot-bufferize{bufferize-function-boundaries}, 
                expand-realloc, 
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
            # print("erasing transform module")
            op.erase()
            # gc.collect()
    return module


def lowerLinalg(module: Module) -> Module:
    pm = PassManager.parse("builtin.module(convert-linalg-to-loops,convert-scf-to-cf)")
    pm.run(module.operation)
    return module


def lowerToLLVM(module: Module) -> Module:
    pm = PassManager.parse(
        "builtin.module(convert-complex-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"
    )
    pm.run(module.operation)
    return module


def execute(
    maybe_f: Optional[Callable[[], Module]] = None, test_arg: Optional[int] = None
) -> Callable[[Module], Callable[[], Module]]:
    def outer(f: Callable[[], Module]) -> Callable[[], Module]:
        def wrapped():
            if test_arg is not None:
                print(f"args[0] is not None: {test_arg}")
            module = f()
            with module.context:
                arg1 = np.array([11.0]).astype(np.float32)
                arg2 = np.array([25.0]).astype(np.float32)
                arg3 = np.array([0.0]).astype(np.float32)

                arg1_memref_ptr = ctypes.pointer(
                    ctypes.pointer(get_ranked_memref_descriptor(arg1))
                )
                arg2_memref_ptr = ctypes.pointer(
                    ctypes.pointer(get_ranked_memref_descriptor(arg2))
                )
                arg3_memref_ptr = ctypes.pointer(
                    ctypes.pointer(get_ranked_memref_descriptor(arg3))
                )
                # print(f"before bufferize\n{module}")
                bufferize(module)
                # print(f"after bufferize\n{module}")
                lowerLinalg(module)
                # print(f"after lower to linalg\n{module}")

                execution_engine = ExecutionEngine(lowerToLLVM(module))
                execution_engine.invoke(
                    "matmul_signed_on_buffers",
                    arg1_memref_ptr,
                    arg2_memref_ptr,
                    arg3_memref_ptr,
                )

                print(f"{arg1} * {arg2} = {arg3}")

            return module

        return wrapped

    @wraps(outer)
    def maybe_no_args(f: Optional[Callable[[], Module]] = None) -> Callable[[], Module]:
        if maybe_f:
            return outer(maybe_f)()
        else:
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


def construct_module(f: Callable[[Module], None]) -> Callable[[], Module]:
    def wrapped():
        with Context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f(module)
            return module

    return wrapped
