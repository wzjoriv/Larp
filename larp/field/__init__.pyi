"""Type stubs for larp.field (compiled from geometry.pyx, quadtree.pyx, risk_field.pyx, kernels.pyx)."""

from larp.field.kernels import (
    point_in_rings as point_in_rings,
    point_obstacles_eval_max as point_obstacles_eval_max,
    quadtree_find_chain as quadtree_find_chain,
    quadtree_find_leaf as quadtree_find_leaf,
    segment_repulsion_vectors as segment_repulsion_vectors,
)
from larp.field.geometry import (
    GeometryCollectionRGJ as GeometryCollectionRGJ,
    LineStringRGJ as LineStringRGJ,
    MultiLineStringRGJ as MultiLineStringRGJ,
    MultiPointRGJ as MultiPointRGJ,
    MultiPolygonRGJ as MultiPolygonRGJ,
    MultiRGJGeometry as MultiRGJGeometry,
    PointRGJ as PointRGJ,
    PolygonRGJ as PolygonRGJ,
    RGJGeometry as RGJGeometry,
)
from larp.field.quadtree import QuadNode as QuadNode, QuadTree as QuadTree
from larp.field.risk_field import RiskField as RiskField
