"""Type stubs for larp.field.geometry.collection (compiled from collection.pyx)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from larp.field.geometry.base import MultiRGJGeometry, RGJGeometry
from larp.types import Point, RGeoJSONObject

FloatArray = npt.NDArray[np.float64]

class GeometryCollectionRGJ(MultiRGJGeometry):
    RGJType: ClassVar[str]
    rgjs: list[RGJGeometry]
    inv_repulsions: FloatArray
    grad_matrixes: FloatArray

    def __init__(self, geometries: list[dict[str, Any]], properties: dict | None = ..., **kwargs: Any) -> None: ...
    def set_repulsion(self, new_repulsion: Any) -> None: ...
    def in_bbox(self, x: Point) -> bool: ...
    def get_dist_matrix(self, scaled: bool = ..., inverted: bool = ...) -> FloatArray: ...
    def get_center_point(self) -> FloatArray: ...
    def squared_dist(  # type: ignore[override]
        self, x: FloatArray, scaled: bool = ..., inverted: bool = ..., return_reference: bool = ..., **kwargs: Any
    ) -> Any: ...
    def repulsion_vector(self, x: FloatArray, min_dist_select: bool = ..., **kwargs: Any) -> FloatArray: ...
    def gradient(self, x: FloatArray, **kwargs: Any) -> FloatArray: ...
    def toRGeoJSON(self) -> RGeoJSONObject: ...
