# cython: language_level=3
"""PolygonRGJ / MultiPolygonRGJ."""

import numpy as np

from larp.field import kernels
from larp.field.geometry.base cimport RGJGeometry, MultiRGJGeometry


cdef class PolygonRGJ(RGJGeometry):
    RGJType = "Polygon"

    cdef public list rings
    cdef public object points_in_line_pair
    cdef public object _p1, _v, _v_dot_v
    cdef public object metric
    cdef public Py_ssize_t _S

    def __init__(self, coordinates, repulsion=None, properties=None, optional_dim=2, **kwargs):
        if isinstance(coordinates, np.ndarray):
            rings = [coordinates.astype(float)] if coordinates.ndim == 2 else [np.asarray(r, float) for r in coordinates]
        elif isinstance(coordinates, (list, tuple)):
            if len(coordinates) > 0:
                first = coordinates[0]
                if isinstance(first, (int, float, np.floating)):
                    rings = [np.asarray(coordinates, float).reshape(-1, 2)]
                elif isinstance(first, (list, tuple, np.ndarray)) and len(first) == 2:
                    rings = [np.asarray(coordinates, float).reshape(-1, 2)]
                else:
                    rings = [np.asarray(r, float).reshape(-1, 2) for r in coordinates]
            else:
                rings = []
        else:
            raise TypeError("Coordinates must be ndarray or list/tuple.")

        self.rings = []
        for r in rings:
            if r.ndim != 2 or r.shape[1] != 2:
                r = r.reshape(-1, 2)
            if r.size > 0 and not np.allclose(r[0], r[-1]):
                r = np.vstack([r, r[0]])
            self.rings.append(np.ascontiguousarray(r, dtype=np.float64))

        segments = []
        for r in self.rings:
            if r.shape[0] >= 2:
                segments.append(np.stack([r[:-1], r[1:]], axis=1))

        self.points_in_line_pair = np.concatenate(segments, axis=0) if segments else np.zeros((0, 2, 2))
        self._build_segment_cache()

        self.coordinates = np.array([r.tolist() for r in self.rings], dtype=object)
        self.repulsion = np.eye(optional_dim) if repulsion is None else np.array(repulsion, dtype=np.float64)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.properties = {} if properties is None else properties
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T
        self.metric = self.get_dist_matrix(scaled=True, inverted=True)

        all_pts = np.vstack(self.rings) if self.rings else np.zeros((0, 2))
        self.bbox = np.array([all_pts.min(axis=0), all_pts.max(axis=0)]) if all_pts.size else np.array([[0.0, 0.0], [0.0, 0.0]])

    def _build_segment_cache(self):
        if self.points_in_line_pair.size:
            self._p1 = np.ascontiguousarray(self.points_in_line_pair[:, 0], dtype=np.float64)
            self._v = np.ascontiguousarray(self.points_in_line_pair[:, 1] - self.points_in_line_pair[:, 0], dtype=np.float64)
            vdv = (self._v * self._v).sum(axis=1)
            self._v_dot_v = np.where(vdv <= 0, 1.0, vdv)
            self._S = self._p1.shape[0]
        else:
            self._p1 = np.zeros((0, 2))
            self._v = np.zeros((0, 2))
            self._v_dot_v = np.zeros((0,))
            self._S = 0

    def set_coordinates(self, new_coords):
        self.__init__(coordinates=new_coords, repulsion=self.repulsion, properties=self.properties)

    def set_repulsion(self, new_repulsion):
        super().set_repulsion(new_repulsion)
        self.metric = self.get_dist_matrix(scaled=True, inverted=True)

    def in_bbox(self, x):
        x = np.asarray(x)
        return bool(np.all(x >= self.bbox[0]) and np.all(x <= self.bbox[1]))

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        x = np.atleast_2d(x).astype(np.float64)
        n = x.shape[0]

        if self._S == 0:
            return np.zeros((n, 2))

        outer = self.rings[0]
        holes = self.rings[1:]
        inside = kernels.point_in_rings(x, outer, holes).astype(bool)

        if inside.all():
            return np.zeros((n, 2))

        outside_idx = np.where(~inside)[0]
        xo = np.ascontiguousarray(x[outside_idx])

        chosen = kernels.segment_repulsion_vectors(xo, self._p1, self._v, self._v_dot_v, self.metric)

        out = np.zeros((n, 2))
        out[outside_idx] = chosen
        return out

    def toRGeoJSON(self):
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": self.RGJType,
                "coordinates": [r.tolist() for r in self.rings],
                "repulsion": self.repulsion.tolist(),
            },
        }


cdef class MultiPolygonRGJ(MultiRGJGeometry):
    RGJType = "MultiPolygon"

    cdef public list polygons
    cdef public object points_in_line_pair
    cdef public object _p1, _v, _v_dot_v
    cdef public Py_ssize_t _S

    def __init__(self, coordinates, repulsion=None, properties=None, optional_dim=2, **kwargs):
        self.polygons = [
            PolygonRGJ(poly_coords, repulsion=repulsion, properties=properties, optional_dim=optional_dim)
            for poly_coords in coordinates
        ]

        self.repulsion = np.eye(optional_dim) if repulsion is None else np.array(repulsion, dtype=np.float64)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T
        self.properties = {} if properties is None else properties

        all_segments = []
        for poly in self.polygons:
            if poly._S > 0:
                all_segments.append(poly.points_in_line_pair)

        self.points_in_line_pair = np.concatenate(all_segments, axis=0) if all_segments else np.zeros((0, 2, 2))

        if self.points_in_line_pair.size:
            self._p1 = np.ascontiguousarray(self.points_in_line_pair[:, 0], dtype=np.float64)
            self._v = np.ascontiguousarray(self.points_in_line_pair[:, 1] - self.points_in_line_pair[:, 0], dtype=np.float64)
            vdv = (self._v * self._v).sum(axis=1)
            self._v_dot_v = np.where(vdv <= 0, 1.0, vdv)
            self._S = self._p1.shape[0]
        else:
            self._p1 = np.zeros((0, 2))
            self._v = np.zeros((0, 2))
            self._v_dot_v = np.zeros((0,))
            self._S = 0

        if len(self.polygons) > 0:
            bboxes = np.array([p.bbox for p in self.polygons]).reshape(-1, 2)
            self.bbox = np.array([bboxes.min(0), bboxes.max(0)])
        else:
            self.bbox = np.array([[0.0, 0.0], [0.0, 0.0]])

    def set_coordinates(self, new_coords):
        self.__init__(coordinates=new_coords, repulsion=self.repulsion, properties=self.properties, optional_dim=len(self.repulsion))

    def in_bbox(self, x):
        return any(poly.in_bbox(x) for poly in self.polygons)

    def get_center_point(self):
        centers = np.array([p.get_center_point() for p in self.polygons])
        return np.array([centers.min(0), centers.max(0)]).mean(0)

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        x = np.atleast_2d(x).astype(np.float64)
        n = x.shape[0]

        if self._S == 0:
            return np.zeros((n, 2))

        if min_dist_select:
            metric = self.get_dist_matrix(scaled=True, inverted=True)
            return kernels.segment_repulsion_vectors(x, self._p1, self._v, self._v_dot_v, metric)

        w = x[None, :, :] - self._p1[:, None, :]
        t = np.clip((w * self._v[:, None, :]).sum(axis=2) / self._v_dot_v[:, None], 0.0, 1.0)
        proj = self._p1[:, None, :] + t[..., None] * self._v[:, None, :]
        return (x[None, :, :] - proj).reshape(-1, 2)

    def toRGeoJSON(self):
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": self.RGJType,
                "coordinates": [poly.rings for poly in self.polygons],
                "repulsion": self.repulsion.tolist(),
            },
        }
