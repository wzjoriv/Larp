"""Type stubs for larp.field.geometry.linestring (compiled from linestring.pyx)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from larp.field.geometry.base import MultiRGJGeometry

FloatArray = npt.NDArray[np.float64]

class LineStringRGJ(MultiRGJGeometry):
    RGJType: ClassVar[str]
    points_in_line_pair: FloatArray
    lines_n: int

    def __init__(self, coordinates: FloatArray, repulsion: FloatArray | None = ..., **kwargs: Any) -> None: ...
    def set_coordinates(self, new_coords: Any) -> None: ...
    def repulsion_vector(self, x: FloatArray, min_dist_select: bool = ..., **kwargs: Any) -> FloatArray: ...

class MultiLineStringRGJ(LineStringRGJ):
    RGJType: ClassVar[str]
    coordinates: list[FloatArray]  # type: ignore[assignment]

    def __init__(
        self,
        coordinates: list[FloatArray],
        repulsion: FloatArray | None = ...,
        properties: dict | None = ...,
        optional_dim: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def set_coordinates(self, new_coords: Any) -> None: ...
    def get_center_point(self) -> FloatArray: ...
