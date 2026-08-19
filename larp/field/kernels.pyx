# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython kernels for the hottest numeric loops used by larp.field's
GeoJSON-standard geometries (Point, LineString, Polygon, MultiPoint,
MultiLineString, MultiPolygon, GeometryCollection).

Non-standard extension geometries (Rectangle, Ellipse) are not ported here;
the new architecture only carries geometries with a direct GeoJSON
equivalent.
"""

from typing import Tuple

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, INFINITY

cnp.import_array()

ctypedef cnp.float64_t f64
ctypedef cnp.int32_t i32


def segment_repulsion_vectors(const f64[:, :] x,
                               const f64[:, :] p1,
                               const f64[:, :] v,
                               const f64[:] v_dot_v,
                               const f64[:, :] metric) -> cnp.ndarray:
    """
    For every point in `x`, find the closest point (in the metric induced by
    `metric`) among the projections onto every segment [p1, p1+v], and return
    the repulsion vector (x - closest_point).

    Used by LineString / Polygon / MultiPolygon boundary-segment repulsion.

    x:       (N, 2)
    p1, v:   (S, 2) segment origins / direction vectors
    v_dot_v: (S,) precomputed dot(v, v) per segment (denominator, >0)
    metric:  (2, 2) anisotropic distance matrix

    Returns: (N, 2) repulsion vectors
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t s = p1.shape[0]
    cdef Py_ssize_t i, j

    cdef cnp.ndarray[f64, ndim=2] out = np.empty((n, 2), dtype=np.float64)
    cdef f64[:, :] out_v = out

    cdef f64 m00 = metric[0, 0], m01 = metric[0, 1]
    cdef f64 m10 = metric[1, 0], m11 = metric[1, 1]

    cdef f64 wx, wy, t, projx, projy, dx, dy, dist2, best_dist2, best_dx, best_dy
    cdef f64 xi, yi

    if s == 0:
        out_v[:, :] = 0.0
        return out

    with nogil:
        for i in range(n):
            xi = x[i, 0]
            yi = x[i, 1]
            best_dist2 = INFINITY
            best_dx = 0.0
            best_dy = 0.0

            for j in range(s):
                wx = xi - p1[j, 0]
                wy = yi - p1[j, 1]

                t = (wx * v[j, 0] + wy * v[j, 1]) / v_dot_v[j]
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0

                projx = p1[j, 0] + t * v[j, 0]
                projy = p1[j, 1] + t * v[j, 1]

                dx = xi - projx
                dy = yi - projy

                dist2 = dx * (m00 * dx + m01 * dy) + dy * (m10 * dx + m11 * dy)

                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_dx = dx
                    best_dy = dy

            out_v[i, 0] = best_dx
            out_v[i, 1] = best_dy

    return out


cdef inline bint _point_in_ring(f64 x, f64 y, const f64[:, :] ring) noexcept nogil:
    """Standard even-odd ray casting rule against a single closed ring."""
    cdef Py_ssize_t n = ring.shape[0]
    cdef Py_ssize_t i, j
    cdef bint inside = False
    cdef f64 xi, yi, xj, yj

    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi = ring[i, 0]
        yi = ring[i, 1]
        xj = ring[j, 0]
        yj = ring[j, 1]

        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i

    return inside


def point_in_rings(const f64[:, :] pts,
                    const f64[:, :] outer,
                    list holes) -> cnp.ndarray:
    """
    Ray-casting point-in-polygon test with hole support.

    pts:   (N, 2) query points
    outer: (R, 2) outer ring (closed, last point == first point)
    holes: list of (H_k, 2) closed ring arrays (interior holes)

    Returns: (N,) uint8 array, 1 if inside outer ring and outside all holes.
    """
    cdef Py_ssize_t n = pts.shape[0]
    cdef Py_ssize_t i, k
    cdef Py_ssize_t n_holes = len(holes)

    cdef cnp.ndarray[cnp.uint8_t, ndim=1] out = np.zeros(n, dtype=np.uint8)
    cdef cnp.uint8_t[:] out_v = out

    cdef list hole_views = [np.ascontiguousarray(h, dtype=np.float64) for h in holes]

    with nogil:
        for i in range(n):
            out_v[i] = _point_in_ring(pts[i, 0], pts[i, 1], outer)

    for i in range(n):
        if out_v[i]:
            for k in range(n_holes):
                if _point_in_ring(pts[i, 0], pts[i, 1], hole_views[k]):
                    out_v[i] = 0
                    break

    return out


def point_obstacles_eval_max(const f64[:, :] x,
                              const f64[:, :] centers,
                              const f64[:, :, :] inv_repulsions) -> cnp.ndarray:
    """
    Fused kernel for RiskField.eval() over many Point-type obstacles: for
    each query point, compute max_k exp(-((x-c_k)^T M_k (x-c_k))) without
    materializing an (N, K) intermediate array in Python.

    x:               (N, 2) query points
    centers:         (K, 2) obstacle coordinates
    inv_repulsions:  (K, 2, 2) per-obstacle anisotropic repulsion metric

    Returns: (N,) risk values in [0, 1]
    """
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t k = centers.shape[0]
    cdef Py_ssize_t i, c

    cdef cnp.ndarray[f64, ndim=1] out = np.zeros(n, dtype=np.float64)
    cdef f64[:] out_v = out

    cdef f64 dx, dy, d2, val, best

    if k == 0:
        return out

    with nogil:
        for i in range(n):
            best = 0.0
            for c in range(k):
                dx = x[i, 0] - centers[c, 0]
                dy = x[i, 1] - centers[c, 1]
                d2 = (dx * (inv_repulsions[c, 0, 0] * dx + inv_repulsions[c, 0, 1] * dy)
                      + dy * (inv_repulsions[c, 1, 0] * dx + inv_repulsions[c, 1, 1] * dy))
                val = exp(-d2)
                if val > best:
                    best = val
            out_v[i] = best

    return out


def quadtree_find_leaf(const f64[:, :] pts,
                        const f64[:] cx,
                        const f64[:] cy,
                        const cnp.uint8_t[:] is_leaf,
                        const i32[:] child_tl,
                        const i32[:] child_tr,
                        const i32[:] child_bl,
                        const i32[:] child_br,
                        int max_depth) -> cnp.ndarray:
    """
    For every point, walk the flattened tree from the root (node 0) and
    return the node index of the leaf reached, or -1 if the walk falls off
    a missing branch (mirrors the old recursion returning None).
    """
    cdef Py_ssize_t n = pts.shape[0]
    cdef Py_ssize_t i
    cdef i32 node
    cdef int depth
    cdef f64 x, y, dx, dy

    cdef cnp.ndarray[i32, ndim=1] out = np.empty(n, dtype=np.int32)
    cdef i32[:] out_v = out

    with nogil:
        for i in range(n):
            node = 0
            x = pts[i, 0]
            y = pts[i, 1]
            depth = 0

            while node >= 0 and not is_leaf[node] and depth < max_depth:
                dx = x - cx[node]
                dy = y - cy[node]
                if dy >= 0.0:
                    node = child_tr[node] if dx >= 0.0 else child_tl[node]
                else:
                    node = child_br[node] if dx >= 0.0 else child_bl[node]
                depth += 1

            out_v[i] = node

    return out


def quadtree_find_chain(const f64[:, :] pts,
                         const f64[:] cx,
                         const f64[:] cy,
                         const cnp.uint8_t[:] is_leaf,
                         const i32[:] child_tl,
                         const i32[:] child_tr,
                         const i32[:] child_bl,
                         const i32[:] child_br,
                         int max_depth,
                         int max_cols) -> Tuple[cnp.ndarray, cnp.ndarray]:
    """
    Same walk as quadtree_find_leaf, but records every node index visited
    (root -> ... -> terminal). Terminal entry is -1 if the walk fell off a
    missing branch, matching the old recursion's `None` chain entry.

    Returns (chain, length):
      chain:  (N, max_cols) int32, only chain[i, :length[i]] is meaningful.
      length: (N,) int32
    """
    cdef Py_ssize_t n = pts.shape[0]
    cdef Py_ssize_t i
    cdef i32 node
    cdef int depth
    cdef f64 x, y, dx, dy

    cdef cnp.ndarray[i32, ndim=2] chain = np.full((n, max_cols), -2, dtype=np.int32)
    cdef i32[:, :] chain_v = chain
    cdef cnp.ndarray[i32, ndim=1] length = np.zeros(n, dtype=np.int32)
    cdef i32[:] length_v = length

    with nogil:
        for i in range(n):
            node = 0
            x = pts[i, 0]
            y = pts[i, 1]
            depth = 0

            while True:
                chain_v[i, depth] = node
                depth += 1

                if node < 0 or is_leaf[node] or depth >= max_depth + 1:
                    break

                dx = x - cx[node]
                dy = y - cy[node]
                if dy >= 0.0:
                    node = child_tr[node] if dx >= 0.0 else child_tl[node]
                else:
                    node = child_br[node] if dx >= 0.0 else child_bl[node]

            length_v[i] = depth

    return chain, length
