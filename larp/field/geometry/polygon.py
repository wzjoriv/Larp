from __future__ import annotations

import numpy as np
from larp.types import Point

from larp.field.geometry.geometry import RGJGeometry

"""
Author: Josue N Rivera

x are assumed to be a list of point coordinates in euclidean space

"""


__all__ = ["PolygonRGJ"]

class PolygonRGJ(RGJGeometry):
    Type = "Polygon"

    def __init__(
        self,
        coordinates: np.ndarray | list[list | tuple],
        repulsion: np.ndarray | None = None,
        properties: dict | None = None,
        optional_dim: int = 2,
        **kwargs
    ) -> None:
        """
        Highly optimized Polygon RGJ.

        coordinates may be:
          - single ring: [(x,y), (x,y), ...]  or np.array([[x,y],...])
          - list of rings: [outer_ring, hole1, hole2, ...]
        """
        # ---------- normalize input to list of rings ----------
        if isinstance(coordinates, np.ndarray):
            if coordinates.ndim == 2:
                rings = [coordinates.astype(float)]
            else:
                rings = [np.asarray(r, float) for r in coordinates]
        elif isinstance(coordinates, (list, tuple)):
            if len(coordinates) > 0:
                first_elem = coordinates[0]
                # Check if it's a flat list of numbers [x, y, x, y...]
                if isinstance(first_elem, (int, float, np.floating)):
                    rings = [np.asarray(coordinates, float).reshape(-1, 2)]
                # Check if it's a list of points [[x,y], [x,y]...] 
                # We check the first point without using np.shape on the whole list
                elif isinstance(first_elem, (list, tuple, np.ndarray)) and len(first_elem) == 2:
                    rings = [np.asarray(coordinates, float).reshape(-1, 2)]
                # Otherwise, it's nested rings [[[x,y], [x,y]...], [[x,y]...]]
                else:
                    rings = [np.asarray(r, float).reshape(-1, 2) for r in coordinates]
            else:
                rings = []
        else:
            raise TypeError("Coordinates must be ndarray or list/tuple.")

        # ensure each ring is closed
        self.rings: List[np.ndarray] = []
        for r in rings:
            if r.ndim != 2 or r.shape[1] != 2:
                r = r.reshape(-1, 2)
            if r.size > 0 and not np.allclose(r[0], r[-1]):
                r = np.vstack([r, r[0]])
            self.rings.append(r.astype(float))

        # ---------- build segment arrays (concatenate all rings' segments) ----------
        segments = []
        self.ring_segment_counts = []
        for r in self.rings:
            if r.shape[0] >= 2:
                segs = np.stack([r[:-1], r[1:]], axis=1)  # (num_segments, 2, 2)
                segments.append(segs)
                self.ring_segment_counts.append(segs.shape[0])
            else:
                self.ring_segment_counts.append(0)

        self.points_in_line_pair = np.concatenate(segments, axis=0) if segments else np.zeros((0, 2, 2))

        # ---------- cached per-segment arrays ----------
        if self.points_in_line_pair.size:
            self._p1 = self.points_in_line_pair[:, 0]      # (S,2)
            self._p2 = self.points_in_line_pair[:, 1]      # (S,2)
            self._v  = self._p2 - self._p1                 # (S,2)
            self._v_dot_v = (self._v * self._v).sum(axis=1)  # (S,)
            # avoid strict zero denom
            self._denom = np.where(self._v_dot_v <= 0, 1.0, self._v_dot_v)
            self._S = self._p1.shape[0]
        else:
            self._p1 = np.zeros((0,2))
            self._p2 = np.zeros((0,2))
            self._v = np.zeros((0,2))
            self._v_dot_v = np.zeros((0,))
            self._denom = np.zeros((0,))
            self._S = 0

        # ---------- base class attributes ----------
        self.coordinates = np.array([r.tolist() for r in self.rings], dtype=object)
        self.repulsion = np.eye(optional_dim) if repulsion is None else np.array(repulsion)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.properties = {} if properties is None else properties
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T

        # metric cache (anisotropic distance matrix M)
        self.metric = self.get_dist_matrix(scaled=True, inverted=True)

        # ---------- bbox across all rings ----------
        all_pts = np.vstack(self.rings) if self.rings else np.zeros((0,2))
        self.bbox = np.array([all_pts.min(axis=0), all_pts.max(axis=0)]) if all_pts.size else np.array([[0.0,0.0],[0.0,0.0]])

    def set_coordinates(self, new_coords):
        # Rebuild everything by reinitializing with same repulsion/properties
        self.__init__(coordinates=new_coords, repulsion=self.repulsion, properties=self.properties)

    def set_repulsion(self, new_repulsion):
        super().set_repulsion(new_repulsion)
        # update cached metric
        self.metric = self.get_dist_matrix(scaled=True, inverted=True)
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T

    # ---------------- vectorized helpers ----------------

    def _points_on_segments_vectorized(self, pts: np.ndarray, eps: float = ON_EDGE_EPS) -> np.ndarray:
        """
        Vectorized check whether each point is exactly on any segment.
        Returns boolean array shape (N,) True if on boundary.
        """
        if self._S == 0:
            return np.zeros(pts.shape[0], dtype=bool)

        # pts (N,2), _p1 (S,2), _v (S,2)
        w = pts[:, None, :] - self._p1[None, :, :]    # (N,S,2)
        cross = self._v[None, :, 0] * w[..., 1] - self._v[None, :, 1] * w[..., 0]  # (N,S)
        cross_ok = np.abs(cross) <= eps

        dot = (w * self._v[None, :, :]).sum(axis=2)   # (N,S)
        dot_ok = (dot >= -eps) & (dot <= self._v_dot_v[None, :] + eps)

        on_seg = cross_ok & dot_ok
        return on_seg.any(axis=1)

    def _ray_cast_ring_vectorized(self, pts: np.ndarray, ring: np.ndarray) -> np.ndarray:
        N = pts.shape[0]
        if ring.shape[0] < 2:
            return np.zeros(N, dtype=bool)

        xi = ring[:-1, 0]  # (E,)
        yi = ring[:-1, 1]
        xj = ring[1:, 0]
        yj = ring[1:, 1]

        x = pts[:, 0][:, None]  # (N,1)
        y = pts[:, 1][:, None]  # (N,1)

        # For each edge, compute dy and dx
        dy = (yj - yi)[None, :]  # shape (1, E)
        dx = (xj - xi)[None, :]

        # Condition: ray crosses vertical span, and edge is not horizontal (dy != 0)
        cond_y = ((yi > y) != (yj > y))  # shape (N, E)
        non_horiz = (dy != 0)[0, :]      # shape (E,)

        # Combined mask
        mask = cond_y & non_horiz[None, :]

        if not mask.any():
            return np.zeros(N, dtype=bool)

        # Safe compute t = (y - yi) / dy only where mask is True
        # Use np.divide with where=mask to avoid division by zero
        t = np.zeros_like(mask, dtype=float)
        np.divide((y - yi[None, :]), dy, out=t, where=mask)

        x_inter = xi[None, :] + t * dx  # (N, E)
        cond_x = x < x_inter

        intersects = mask & cond_x
        inside = (intersects.sum(axis=1) % 2) == 1
        return inside

    def _point_in_polygon_vectorized(self, pts: np.ndarray) -> np.ndarray:
        """
        Vectorized inside check including holes and on-edge -> True.
        """
        n = pts.shape[0]
        if len(self.rings) == 0:
            return np.zeros(n, dtype=bool)

        # bbox fast rejection
        outside_bbox = np.any((pts < self.bbox[0]) | (pts > self.bbox[1]), axis=1)
        inside = np.zeros(n, dtype=bool)

        # On-edge detection across all segments
        on_edge = self._points_on_segments_vectorized(pts, eps=ON_EDGE_EPS)
        inside[on_edge] = True

        # For points that are not on-edge and inside bbox, do ray cast
        candidates_idx = np.where(~inside & ~outside_bbox)[0]
        if candidates_idx.size == 0:
            return inside

        cand_pts = pts[candidates_idx]

        # outer ring test
        outer = self.rings[0]
        outer_inside = self._ray_cast_ring_vectorized(cand_pts, outer)

        # holes: if any hole contains point -> it's outside
        if len(self.rings) > 1:
            hole_contains = np.zeros(cand_pts.shape[0], dtype=bool)
            for hole in self.rings[1:]:
                hole_inside = self._ray_cast_ring_vectorized(cand_pts, hole)
                hole_contains |= hole_inside
            final = outer_inside & (~hole_contains)
        else:
            final = outer_inside

        inside[candidates_idx] = final
        return inside

    # ---------------- main API ----------------

    def repulsion_vector(self, x: np.ndarray, min_dist_select: bool = True, **kwargs) -> np.ndarray:
        """
        Vectorized and optimized repulsion vector:
         - inside points -> zero vector
         - outside points -> vector from closest point on boundary to x
        """
        x = np.atleast_2d(x).astype(float)
        N = x.shape[0]

        if self._S == 0:
            return np.zeros((N, 2), dtype=float)

        # compute inside flags
        inside = self._point_in_polygon_vectorized(x)

        # if all inside -> zeros
        if inside.all():
            return np.zeros((N, 2), dtype=float)

        # indices of outside points (need repulsion)
        outside_idx = np.where(~inside)[0]
        xo = x[outside_idx]                       # (K,2)
        K = xo.shape[0]

        # broadcasted vectors for projection:
        # w = xo[:, None, :] - p1[None, :, :]  => shape (K, S, 2)
        w = xo[:, None, :] - self._p1[None, :, :]

        # projection scalar t: (K,S) = sum(w * v) / denom
        # einsum 'kmi,mi->km'
        t = np.einsum("kmi,mi->km", w, self._v) / self._denom
        t = np.clip(t, 0.0, 1.0)

        # closest points on each segment: proj = p1 + t[...,None] * v
        proj = self._p1[None, :, :] + t[..., None] * self._v[None, :, :]
        diff = xo[:, None, :] - proj  # (K,S,2)

        # metric distance squared with precomputed metric M:
        # dist2[k,s] = diff[k,s] @ M @ diff[k,s].T
        # single einsum pass:
        dist2 = np.einsum("kmi,ij,kmj->km", diff, self.metric, diff)  # (K,S)

        # choose nearest segment per point
        idx_min = np.argmin(dist2, axis=1)               # (K,)
        chosen = diff[np.arange(K), idx_min]             # (K,2)

        # prepare result and write chosen vectors back
        out = np.zeros((N, 2), dtype=float)
        out[outside_idx] = chosen
        return out

    # ---------------- misc ----------------

    def in_bbox(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return np.all(x >= self.bbox[0]) and np.all(x <= self.bbox[1])

    def toRGeoJSON(self) -> RGeoJSONObject:
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": self.RGJType,
                "coordinates": [r.tolist() for r in self.rings],
                "repulsion": self.repulsion.tolist(),
            },
        }