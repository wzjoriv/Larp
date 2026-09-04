# cython: language_level=3
"""LineStringRGJ / MultiLineStringRGJ."""

import numpy as np

from larp.field import kernels
from larp.field.geometry.base cimport MultiRGJGeometry


cdef class LineStringRGJ(MultiRGJGeometry):
    RGJType = "LineString"

    cdef public object points_in_line_pair
    cdef public object _p1, _v, _v_dot_v
    cdef public Py_ssize_t lines_n

    def __init__(self, coordinates, repulsion=None, **kwargs):
        super().__init__(coordinates=coordinates, repulsion=repulsion, **kwargs)
        self._build_segments()

    def _build_segments(self):
        self.lines_n = len(self.coordinates) - 1
        self.points_in_line_pair = np.stack([self.coordinates[:-1], self.coordinates[1:]], axis=1)
        self._p1 = np.ascontiguousarray(self.points_in_line_pair[:, 0], dtype=np.float64)
        self._v = np.ascontiguousarray(self.points_in_line_pair[:, 1] - self.points_in_line_pair[:, 0], dtype=np.float64)
        vdv = (self._v * self._v).sum(axis=1)
        self._v_dot_v = np.where(vdv <= 0, 1.0, vdv)

    def set_coordinates(self, new_coords):
        super().set_coordinates(new_coords)
        self._build_segments()

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        x = np.atleast_2d(x).astype(np.float64)
        if self.lines_n <= 0:
            return np.zeros((len(x), 2))

        if min_dist_select:
            metric = self.get_dist_matrix(scaled=True, inverted=True)
            return kernels.segment_repulsion_vectors(x, self._p1, self._v, self._v_dot_v, metric)

        # unreduced: vector to every segment's closest point, stacked (S*N, 2)
        w = x[None, :, :] - self._p1[:, None, :]
        t = np.clip((w * self._v[:, None, :]).sum(axis=2) / self._v_dot_v[:, None], 0.0, 1.0)
        proj = self._p1[:, None, :] + t[..., None] * self._v[:, None, :]
        return (x[None, :, :] - proj).reshape(-1, 2)


cdef class MultiLineStringRGJ(LineStringRGJ):
    RGJType = "MultiLineString"

    def __init__(self, coordinates, repulsion=None, properties=None, optional_dim=2, **kwargs):
        self.coordinates = [np.array(c, dtype=np.float64) for c in coordinates]
        self.repulsion = np.eye(optional_dim) if repulsion is None else np.array(repulsion, dtype=np.float64)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.properties = {} if properties is None else properties
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T
        self.bbox = np.concatenate([np.array([c.min(0), c.max(0)]) for c in self.coordinates])
        self._build_segments()

    def _build_segments(self):
        self.lines_n = sum(len(c) - 1 for c in self.coordinates)
        self.points_in_line_pair = np.concatenate(
            [[c[:-1], c[1:]] for c in self.coordinates], axis=1
        ).swapaxes(0, 1)
        self._p1 = np.ascontiguousarray(self.points_in_line_pair[:, 0], dtype=np.float64)
        self._v = np.ascontiguousarray(self.points_in_line_pair[:, 1] - self.points_in_line_pair[:, 0], dtype=np.float64)
        vdv = (self._v * self._v).sum(axis=1)
        self._v_dot_v = np.where(vdv <= 0, 1.0, vdv)

    def set_coordinates(self, new_coords):
        self.coordinates = [np.array(c, dtype=np.float64) for c in new_coords]
        self.bbox = np.concatenate([np.array([c.min(0), c.max(0)]) for c in self.coordinates])
        self._build_segments()

    def get_center_point(self):
        coords = np.concatenate([c.reshape((-1, 2)) for c in self.coordinates], axis=0)
        return (coords.min(0) + coords.max(0)) / 2.0
