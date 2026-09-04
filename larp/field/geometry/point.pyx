# cython: language_level=3
"""PointRGJ / MultiPointRGJ."""

import numpy as np

from larp.field.geometry.base cimport RGJGeometry, MultiRGJGeometry


cdef class PointRGJ(RGJGeometry):
    RGJType = "Point"

    def repulsion_vector(self, x, **kwargs):
        return np.atleast_2d(x).astype(np.float64) - self.coordinates


cdef class MultiPointRGJ(MultiRGJGeometry):
    RGJType = "MultiPoint"

    def __init__(self, coordinates, repulsion=None, **kwargs):
        super().__init__(coordinates=coordinates, repulsion=repulsion, **kwargs)
        self.bbox = self.coordinates.copy()

    def in_bbox(self, x):
        return any(self.bbox == x)

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        x = np.atleast_2d(x).astype(np.float64)
        n = x.shape[0]
        m = self.coordinates.shape[0]

        diff = self.coordinates[None, :, :] - x[:, None, :]  # (N, M, 2)

        if min_dist_select:
            matrix = self.get_dist_matrix(scaled=True, inverted=True)
            adiff = diff @ matrix
            dist = (adiff * diff).sum(-1)
            select = dist.argmin(1)
            return diff[np.arange(n), select]

        return diff.swapaxes(0, 1).reshape(-1, 2)
