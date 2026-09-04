"""Type stubs for larp.field.geometry (re-exports from its submodules)."""

from __future__ import annotations

from larp.field.geometry.base import MultiRGJGeometry as MultiRGJGeometry, RGJGeometry as RGJGeometry
from larp.field.geometry.collection import GeometryCollectionRGJ as GeometryCollectionRGJ
from larp.field.geometry.linestring import (
    LineStringRGJ as LineStringRGJ,
    MultiLineStringRGJ as MultiLineStringRGJ,
)
from larp.field.geometry.point import MultiPointRGJ as MultiPointRGJ, PointRGJ as PointRGJ
from larp.field.geometry.polygon import MultiPolygonRGJ as MultiPolygonRGJ, PolygonRGJ as PolygonRGJ

_RGJ_TYPES: dict[str, type]
