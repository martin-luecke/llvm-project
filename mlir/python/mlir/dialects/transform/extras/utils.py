#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import abc
from dataclasses import dataclass, fields
from enum import Enum
from typing import Sequence

# from ..extras import OpHandle


@dataclass
class MultiHandleResult(abc.ABC):
    """Base class for all classes that support returning named handles."""

    def __iter__(self):
        yield from [getattr(self, field.name) for field in fields(self)]


@dataclass
class TileResult(MultiHandleResult):
    tiled_op: "OpHandle"
    loops: Sequence["OpHandle"]


class TileLoopKind(Enum):
    """Kind of loop operation to produce in tiling."""

    FOR = "scf.for"
    FORALL = "scf.forall"
