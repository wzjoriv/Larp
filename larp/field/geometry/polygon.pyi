"""Type stubs for larp.field.geometry.polygon (compiled from polygon.pyx)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from larp.field.geometry.base import MultiRGJGeometry, RGJGeometry
from larp.types import Point, RGeoJSONObject

FloatArray = npt.NDArray[np.float64]

class PolygonRGJ(RGJGeometry):
    RGJType: ClassVar[str]
    rings: list[FloatArray]
    points_in_line_pair: FloatArray
    metric: FloatArray

    def __init__(
        self,
        coordinates: FloatArray | list[Any],
        repulsion: FloatArray | None = ...,
        properties: dict | None = ...,
        optional_dim: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def set_coordinates(self, new_coords: Any) -> None: ...
    def set_repulsion(self, new_repulsion: Any) -> None: ...
    def in_bbox(self, x: Point) -> bool: ...
    def repulsion_vector(self, x: FloatArray, min_dist_select: bool = ..., **kwargs: Any) -> FloatArray: ...
    def toRGeoJSON(self) -> RGeoJSONObject: ...

class MultiPolygonRGJ(MultiRGJGeometry):
    RGJType: ClassVar[str]
    polygons: list[PolygonRGJ]
    points_in_line_pair: FloatArray

    def __init__(
        self,
        coordinates: list[Any],
        repulsion: FloatArray | None = ...,
        properties: dict | None = ...,
        optional_dim: int = ...,
        **kwargs: Any,
    ) -> None: ...
    def set_coordinates(self, new_coords: Any) -> None: ...
    def in_bbox(self, x: Point) -> bool: ...
    def get_center_point(self) -> FloatArray: ...
    def repulsion_vector(self, x: FloatArray, min_dist_select: bool = ..., **kwargs: Any) -> FloatArray: ...
    def toRGeoJSON(self) -> RGeoJSONObject: ...
