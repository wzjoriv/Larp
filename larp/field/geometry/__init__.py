"""
The RGJGeometry hierarchy: Point, LineString, Polygon, MultiPoint,
MultiLineString, MultiPolygon, GeometryCollection.

Each type (or closely related pair, e.g. LineString/MultiLineString)
lives in its own compiled module; this package re-exports the full set.
"""

from larp.field.geometry.base import RGJGeometry, MultiRGJGeometry
from larp.field.geometry.point import PointRGJ, MultiPointRGJ
from larp.field.geometry.linestring import LineStringRGJ, MultiLineStringRGJ
from larp.field.geometry.polygon import PolygonRGJ, MultiPolygonRGJ
from larp.field.geometry.collection import GeometryCollectionRGJ

_RGJ_TYPES = {
    "Point": PointRGJ,
    "LineString": LineStringRGJ,
    "Polygon": PolygonRGJ,
    "MultiPoint": MultiPointRGJ,
    "MultiLineString": MultiLineStringRGJ,
    "MultiPolygon": MultiPolygonRGJ,
    "GeometryCollection": GeometryCollectionRGJ,
}

__all__ = [
    "RGJGeometry",
    "MultiRGJGeometry",
    "PointRGJ",
    "LineStringRGJ",
    "PolygonRGJ",
    "MultiPointRGJ",
    "MultiLineStringRGJ",
    "MultiPolygonRGJ",
    "GeometryCollectionRGJ",
    "_RGJ_TYPES",
]
