#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Optional, Sequence, Union

# from .utils import TileLoopKind
from ....extras.meta import region_op
from .... import ir
from ... import transform
from .. import (
    AnyOpType,
    AnyParamType,
    AnyValueType,
    OperationType,
    ParamType,
    NamedSequenceOp,
    YieldOp,
    SequenceOp,
    ApplyPatternsOp,
    any_op_t,
)
from .. import structured
from .utils import MultiHandleResult, TileLoopKind, TileResult
import abc
from dataclasses import dataclass, fields
from enum import Enum


class Handle(ir.Value):
    """
    Base class for wrappers around different types of transform handle with
    methods to chain further transforms.

    The fields `children` and `parent` are used to capture the relation of
    handles statically in order to enable further analysis. The payload
    operation of a child handle is nested into a region of the payload operation
    of the corresponding parent handle.
    """

    def __init__(
        self,
        v: ir.Value,
        *,
        parent: Optional["Handle"] = None,
        children: Optional[Sequence["Handle"]] = None,
    ):
        super().__init__(v)
        self.parent = parent
        self.children = children if children is not None else []

@ir.register_value_caster(AnyOpType.get_static_typeid())
@ir.register_value_caster(OperationType.get_static_typeid())
class OpHandle(Handle):
    """
    Wrapper around a transform operation handle with methods to chain further
    transforms.
    """

    def __init__(
        self,
        v: ir.Value,
        *,
        parent: Optional[Handle] = None,
        children: Optional[Sequence[Handle]] = None,
    ):
        super().__init__(v, parent=parent, children=children)

    def get_result(self, indices: Sequence[int] = [0]) -> "ValueHandle":
        """
        Emits a `transform.GetResultOp`.
        Returns a handle to the result of the payload operation at the given
        indices.
        """
        get_result_op = transform.GetResultOp(
            AnyValueType.get(),
            self,
            indices,
        )
        return get_result_op.result

    def match_ops(
        self,
        ops: Union[
            str,
            ir.OpView,
            structured.MatchInterfaceEnum,
            Sequence[Union[str, ir.OpView]],
        ],
    ) -> "OpHandle":
        """
        Emits a `transform.structured.MatchOp`.
        Returns a handle to payload ops that match the given names, types, or
        interface. If only a single type is given, the value wrapped by the
        resulting handle is populated with the respective type.
        """
        # Handle interface.
        if isinstance(ops, structured.MatchInterfaceEnum) or (
            isinstance(ops, str) and ops in structured.MatchInterfaceEnum.__members__
        ):
            if isinstance(ops, str):
                ops = structured.MatchInterfaceEnum[ops]
            match_op = structured.MatchOp(
                AnyOpType.get(),
                self,
                interface=ops,
            )

        # Handle op name(s), either given directly as string or given as op.
        else:
            if isinstance(ops, str):
                op_type = OperationType.get(ops)
                op_names = [ops]
            elif isinstance(ops, Sequence):
                op_type = AnyOpType.get()
                op_names = [
                    op if isinstance(op, str) else op.OPERATION_NAME for op in ops
                ]
            else:
                op_type = OperationType.get(ops.OPERATION_NAME)
                op_names = [ops.OPERATION_NAME]
            match_op = structured.MatchOp.match_op_names(
                op_type,
                self,
                op_names,
            )

        handle = OpHandle(match_op.results_, parent=self)
        self.children.append(handle)
        return handle

    def print(self, name: Optional[str] = None) -> "OpHandle":
        """
        Emits a `transform.PrintOp` to print this handle and an optional message.
        Returns the existing handle to facilitate further chaining.
        """
        transform.PrintOp(target=self, name=name)
        return self

    def _tile_using_for(
        self,
        *,
        tile_sizes: Sequence[Union[int, "ParamHandle"]],
        interchange: Optional[Sequence[int]] = None,
    ):
        op = structured.TileUsingForOp(
            self,
            sizes=tile_sizes,
            interchange=interchange,
        )
        # self._mlir_value = op.tiled_linalg_op
        # TODO: Think about how to point the python object to the new value
        return TileResult(
            tiled_op=self,
            loops=[OpHandle(loop) for loop in op.loops],
        )

    def _tile_using_forall(
        self,
        *,
        mapping: Optional[
            Union[str, ir.Attribute, Sequence[Union[str, ir.Attribute]]]
        ] = None,
        num_threads: Optional[Sequence[int]] = None,
        tile_sizes: Optional[Sequence[int]] = None,
    ) -> TileResult:
        """Creates a new `structured.TileUsingForallOp` op.

        The func.func payload op surrounding the payload this handle represents
        will be autonormalized to LoopNormalform if needed.

        This handle will be updated to represent the tiled op.
        """
        # TODO(mluecke): Remove string parsing of attributes once builders for GPU
        #                dialect attributes are available
        attr_or_parse = lambda x: ir.Attribute.parse(x) if isinstance(x, str) else x
        if isinstance(mapping, (str, ir.Attribute)):
            mapping = attr_or_parse(mapping)
        elif mapping is not None:
            mapping = ir.ArrayAttr.get([attr_or_parse(attr) for attr in mapping])

        op = structured.TileUsingForallOp(
            any_op_t(),
            any_op_t(),
            self,
            num_threads=num_threads,
            tile_sizes=tile_sizes,
            mapping=mapping,
        )
        # self._mlir_value = op.tiled_op
        return TileResult(
            loops=[OpHandle(op.forall_op)],
            tiled_op=self,
        )

    def tile(
        self,
        *,
        loop: TileLoopKind,
        tile_sizes: Optional[Sequence[Union[int, "ParamHandle"]]] = None,
        interchange: Optional[Sequence[int]] = None,
        num_threads: Optional[Sequence[int]] = None,
        mapping: Optional[
            Union[str, ir.Attribute, Sequence[Union[str, ir.Attribute]]]
        ] = None,
    ) -> TileResult:
        """Creates a new structured tiling operation.

        Depending on the `loop` kwarg, creates either a `structured.TileUsingFor` or
        `structured.TileUsingForall` transform operation. Additional kwargs
        parameterize the created op:

        `tile_sizes`: tile sizes to use in the loop, mandatory for `for` loops;
        `num_threads`: the number of iterations in the produced loop, only supported
                    in `forall` tiling at the moment;
        `interchange`: interchange of the dimensions, only supported in `for` tiling
                    at the moment;
        `mapping`: mapping of the generated loops to parallelism concepts such as
                GPU threads, only supported in `forall` loops (`for` loops are
                implicitly sequential).

        This handle will be updated to represent the tiled linalg op.
        """
        if loop == TileLoopKind.FOR:
            if tile_sizes is None:
                raise ValueError("Tile sizes must be provided.")
            if num_threads is not None or mapping is not None:
                raise ValueError(
                    "Cannot specify num threads or mapping when tiling to scf.for, use"
                    " scf.forall instead."
                )
            return self._tile_using_for(tile_sizes=tile_sizes, interchange=interchange)

        elif loop == TileLoopKind.FORALL:
            if tile_sizes is None and num_threads is None:
                raise ValueError("Must specify either tile sizes or num threads.")
            if interchange is not None:
                raise ValueError(
                    "Cannot specify interchange when tiling to scf.forall."
                )
            if tile_sizes and any(
                isinstance(tile_size, ParamHandle) for tile_size in tile_sizes
            ):
                raise ValueError(
                    "Cannot specify dynamic tile sizes when tiling to scf.forall."
                )
            # return None
            return self._tile_using_forall(
                tile_sizes=tile_sizes, num_threads=num_threads, mapping=mapping
            )

        raise ValueError(f"Uknown loop kind {loop}")


@ir.register_value_caster(AnyParamType.get_static_typeid())
@ir.register_value_caster(ParamType.get_static_typeid())
class ParamHandle(Handle):
    """Wrapper around a transform param handle."""

    def __init__(
        self,
        v: ir.Value,
        *,
        parent: Optional[Handle] = None,
        children: Optional[Sequence[Handle]] = None,
    ):
        super().__init__(v, parent=parent, children=children)


@ir.register_value_caster(AnyValueType.get_static_typeid())
class ValueHandle(Handle):
    """
    Wrapper around a transform value handle with methods to chain further
    transforms.
    """

    def __init__(
        self,
        v: ir.Value,
        *,
        parent: Optional[Handle] = None,
        children: Optional[Sequence[Handle]] = None,
    ):
        super().__init__(v, parent=parent, children=children)

    def get_defining_op(self) -> OpHandle:
        """
        Emits a `transform.GetDefiningOpOp`.
        Returns a handle to the defining op of the wrapped value.
        """
        get_defining_op = transform.GetDefiningOp(
            AnyOpType.get(),
            self,
        )
        return get_defining_op.result


def constant_param(value: Union[ir.Attribute, int]) -> ParamHandle:
    """
    Emits a `transform.ParamConstantOp`.
    Returns a handle to the newly created parameter. The type of the parameter
    is `transfrom.any_param` if the value is not an integer, otherwise the type
    is `transform.param` parametrized with the according integer type.
    """
    if isinstance(value, int):
        value = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
    if isinstance(value.type, ir.IntegerType):
        param_type = ParamType.get(value.type)
    else:
        param_type = AnyParamType.get()
    op = transform.ParamConstantOp(param_type, value)
    return op.param


def insert_transform_script(
    block_or_insertion_point: Union[ir.Block, ir.InsertionPoint],
    script: Callable[[OpHandle], None],
    dump_script: bool = False,
) -> None:
    """
    Inserts the transform script of the schedule into the module. The script
    should accept an instance of OpHandle as argument, which will be called with
    the block arg of the newly created named_sequence op.

    Example:
    This python code
    ```
    module = ir.Module.create()
    def test_match_ops_single(module: OpHandle):
        module.match_ops(scf.ForOp)
    insert_transform_script(module.body, script)
    ```
    generates the following IR:
    ```
    module {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
        ^bb0(%arg0: !transform.any_op):
            %0 = transform.structured.match ops{["scf.for"]} in %arg0
                 : (!transform.any_op) -> !transform.op<"scf.for">
        }
    }
    ```
    """
    if isinstance(block_or_insertion_point, ir.Block):
        context = block_or_insertion_point.owner.context
        insertion_point = ir.InsertionPoint.at_block_begin(block_or_insertion_point)
    else:
        context = block_or_insertion_point.block.owner.context
        insertion_point = block_or_insertion_point

    with context, ir.Location.unknown(context):
        with insertion_point:
            named_sequence_op = NamedSequenceOp(
                "__transform_main", [AnyOpType.get()], []
            )
        with ir.InsertionPoint(named_sequence_op.body):
            script(named_sequence_op.bodyTarget)
            YieldOp([])

    if dump_script:
        print(named_sequence_op)


sequence = region_op(SequenceOp.__base__, terminator=YieldOp)
named_sequence = region_op(NamedSequenceOp, terminator=YieldOp)
apply_patterns = region_op(ApplyPatternsOp)
