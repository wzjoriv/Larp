"""Type stubs for larp.field.geometry.point (compiled from point.pyx)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from larp.field.geometry.base import MultiRGJGeometry, RGJGeometry
from larp.types import Point

FloatArray = npt.NDArray[np.float64]

class PointRGJ(RGJGeometry):
    RGJType: ClassVar[str]
    def repulsion_vector(self, x: FloatArray, **kwargs: Any) -> FloatArray: ...

class MultiPointRGJ(MultiRGJGeometry):
    RGJType: ClassVar[str]
    def __init__(self, coordinates: FloatArray, repulsion: FloatArray | None = ..., **kwargs: Any) -> None: ...
    def in_bbox(self, x: Point) -> bool: ...
    def repulsion_vector(self, x: FloatArray, min_dist_select: bool = ..., **kwargs: Any) -> FloatArray: ...
