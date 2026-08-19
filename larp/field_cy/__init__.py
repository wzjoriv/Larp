"""
Cython-accelerated risk field: the merged RiskField (with optional automatic
quad decomposition) and its GeoJSON-standard geometries (Point, LineString,
Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection).

Build with (from repo root):
    python setup.py build_ext --inplace
"""

from larp.field_cy.kernels import (
    segment_repulsion_vectors,
    point_in_rings,
    point_obstacles_eval_max,
)
from larp.field_cy.geometry import (
    RGJGeometry,
    MultiRGJGeometry,
    PointRGJ,
    LineStringRGJ,
    PolygonRGJ,
    MultiPointRGJ,
    MultiLineStringRGJ,
    MultiPolygonRGJ,
    GeometryCollectionRGJ,
)
from larp.field_cy.quadtree import QuadTree, QuadNode
from larp.field_cy.risk_field import RiskField

__all__ = [
    "segment_repulsion_vectors",
    "point_in_rings",
    "point_obstacles_eval_max",
    "RGJGeometry",
    "MultiRGJGeometry",
    "PointRGJ",
    "LineStringRGJ",
    "PolygonRGJ",
    "MultiPointRGJ",
    "MultiLineStringRGJ",
    "MultiPolygonRGJ",
    "GeometryCollectionRGJ",
    "QuadTree",
    "QuadNode",
    "RiskField",
]
