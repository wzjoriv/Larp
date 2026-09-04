"""Type stubs for larp.field.risk_field (compiled from risk_field.pyx)."""

from __future__ import annotations

from types import EllipsisType
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt

from larp.field.geometry import RGJGeometry
from larp.field.quadtree import QuadTree
from larp.types import FieldSize, Point, RGJDict, RGeoJSONCollection, Scaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

class RiskField:
    rgjs: list[RGJGeometry]
    center_point: FloatArray | None
    size: FloatArray | None
    bbox: FloatArray
    extra_info: dict
    quadtree: QuadTree | None

    minimum_cell_size: float | None | EllipsisType
    maximum_cell_size: float
    risk_epsilon: float
    conservative_tol: float
    conservative: bool

    def __init__(
        self,
        rgjs: list[RGJDict] | list[RGJGeometry] | None = ...,
        center_point: Point | None = ...,
        size: FieldSize | float | None = ...,
        properties: list[dict] | None = ...,
        extra_info: dict | None = ...,
        minimum_cell_size: float | None | EllipsisType = ...,
        maximum_cell_size: float = ...,
        risk_epsilon: float = ...,
        conservative_tol: float = ...,
        conservative: bool = ...,
    ) -> None: ...
    def enable_quadtree(
        self,
        minimum_cell_size: float | EllipsisType = ...,
        maximum_cell_size: float = ...,
        risk_epsilon: float = ...,
        conservative_tol: float = ...,
        conservative: bool = ...,
    ) -> None: ...
    def disable_quadtree(self) -> None: ...
    def __getitem__(self, idxs: int | Iterable[int]) -> RGJGeometry | list[RGJGeometry] | None: ...
    def __iter__(self) -> RiskField: ...
    def __next__(self) -> RGJGeometry: ...
    def __len__(self) -> int: ...
    def set_bbox(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None: ...
    def get_extent(self, margin: float = ...) -> list[float]: ...
    def set_all_repulsion(self, new_repulsion: Any) -> None: ...
    def reload_bbox(self) -> FloatArray: ...
    def reload_center_point(self, toggle: bool = ..., recal_size: bool = ...) -> FloatArray: ...
    def addRGJ(
        self,
        rgj: RGJDict | RGJGeometry,
        properties: dict | None = ...,
        reload_bbox: bool = ...,
        _skip_quad: bool = ...,
        **kwargs: Any,
    ) -> list[int]: ...
    def addField(self, new_field: RiskField, reload_bbox: bool = ...) -> IntArray: ...
    def delRGJ(
        self, idxs: int | list[int] | IntArray, reload_bbox: bool = ..., pop_field: bool = ...
    ) -> RiskField | None: ...
    def in_bbox(self, point: Point, filted_idx: list[int] | None = ..., max_depth: int = ...) -> bool: ...
    def find_bbox(self, point: Point, filted_idx: list[int] | None = ..., max_depth: int = ...) -> IntArray: ...
    def repulsion_vectors(
        self,
        points: FloatArray | list[Point],
        filted_idx: list[int] | None = ...,
        min_dist_select: bool = ...,
        return_reference: bool = ...,
        max_depth: int = ...,
    ) -> FloatArray | tuple[FloatArray, IntArray]: ...
    def gradient(self, points: FloatArray | list[Point], min_dist_select: bool = ..., max_depth: int = ...) -> FloatArray: ...
    def contact_points(
        self,
        points: FloatArray | list[Point],
        filted_idx: list[int] | None = ...,
        min_dist_select: bool = ...,
        return_reference: bool = ...,
        max_depth: int = ...,
    ) -> FloatArray | tuple[FloatArray, IntArray]: ...
    def eval(self, points: FloatArray | list[Point], filted_idx: list[int] | None = ..., max_depth: int = ...) -> FloatArray: ...
    def eval_per(self, points: FloatArray | list[Point], idxs: list[int] | None = ...) -> FloatArray: ...
    def squared_dist(
        self,
        points: FloatArray | list[Point],
        filted_idx: list[int] | None = ...,
        scaled: bool = ...,
        inverted: bool = ...,
        return_reference: bool = ...,
        max_depth: int = ...,
    ) -> FloatArray | tuple[FloatArray, IntArray]: ...
    def squared_dist_per(
        self, points: FloatArray | list[Point], idxs: list[int] | None = ..., scaled: bool = ..., inverted: bool = ...
    ) -> FloatArray: ...
    def squared_dist_list(
        self,
        points: FloatArray | list[Point],
        filted_idx: list[int] | None = ...,
        scaled: bool = ...,
        inverted: bool = ...,
        max_depth: int = ...,
    ) -> FloatArray: ...
    def estimate_route_area(
        self, route: list[Point] | FloatArray, step: float = ..., n: int = ..., scale_transform: Scaler = ..., max_depth: int = ...
    ) -> float: ...
    def estimate_route_highest_risk(
        self, route: list[Point] | FloatArray, step: float = ..., n: int = ..., scale_transform: Scaler = ..., max_depth: int = ...
    ) -> float: ...
    def to_image(
        self,
        resolution: int = ...,
        margin: float = ...,
        center_point: Point | None = ...,
        size: FieldSize | None = ...,
        filted_idx: list[int] | None = ...,
        return_extent: bool = ...,
        max_depth: int = ...,
    ) -> FloatArray | tuple[FloatArray, list[float]]: ...
    def toRGeoJSON(self, return_bbox: bool = ...) -> RGeoJSONCollection: ...
    @property
    def __geo_interface__(self) -> RGeoJSONCollection: ...
