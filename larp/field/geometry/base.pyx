# cython: language_level=3
"""
RGJGeometry / MultiRGJGeometry: base classes for the RGJ geometry
hierarchy. Declared in base.pxd so other modules in this package can
subclass them across compiled files.
"""

import numpy as np


cdef class RGJGeometry:
    """Base class for all Cython RGJ geometries."""

    RGJType = None

    def __init__(self, coordinates, repulsion=None, properties=None, optional_dim=2, **kwargs):
        self.coordinates = np.array(coordinates, dtype=np.float64)
        self.repulsion = np.eye(optional_dim) if repulsion is None else np.array(repulsion, dtype=np.float64)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.properties = {} if properties is None else properties
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T

        bbox = self.coordinates.reshape(-1, 2)
        self.bbox = np.array([bbox.min(0), bbox.max(0)])

    def set_coordinates(self, new_coords):
        self.coordinates = np.array(new_coords, dtype=np.float64)
        bbox = self.coordinates.reshape(-1, 2)
        self.bbox = np.array([bbox.min(0), bbox.max(0)])

    def set_repulsion(self, new_repulsion):
        self.repulsion = np.array(new_repulsion, dtype=np.float64)
        self.inv_repulsion = np.linalg.inv(self.repulsion)
        self.eye_repulsion = np.eye(len(self.repulsion))
        self.grad_matrix = self.inv_repulsion + self.inv_repulsion.T

    def get_dist_matrix(self, scaled=True, inverted=True):
        if inverted and scaled:
            return self.inv_repulsion
        if not scaled:
            return self.eye_repulsion
        return self.repulsion

    def get_center_point(self):
        if len(self.coordinates.shape) <= 1:
            return self.coordinates
        coords = np.reshape(self.coordinates, (-1, 2))
        return (coords.min(0) + coords.max(0)) / 2.0

    def in_bbox(self, x):
        bboxes = self.bbox.reshape((-1, 2))
        for i in range(0, len(bboxes), 2):
            if all(x >= bboxes[i]) and all(x <= bboxes[i + 1]):
                return True
        return False

    def repulsion_vector(self, x, **kwargs):
        raise NotImplementedError

    def squared_dist(self, x, scaled=True, inverted=True, **kwargs):
        nvector = self.repulsion_vector(x, min_dist_select=True)
        matrix = self.get_dist_matrix(scaled=scaled, inverted=inverted)
        return ((nvector @ matrix) * nvector).sum(axis=1)

    def contact_point(self, x, **kwargs):
        return x - self.repulsion_vector(x, **kwargs)

    def gradient(self, x, **kwargs):
        repulsion_vector = self.repulsion_vector(x, **kwargs)
        return -self.eval(x=x).reshape(-1, 1) * (repulsion_vector @ self.grad_matrix.T)

    def eval(self, x):
        return np.exp(-self.squared_dist(x))

    def toRGeoJSON(self):
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": self.RGJType,
                "coordinates": self.coordinates.tolist() if isinstance(self.coordinates, np.ndarray) else self.coordinates,
                "repulsion": self.repulsion.tolist(),
            },
        }

    @property
    def __geo_interface__(self):
        """Standard geospatial interchange protocol."""
        return self.toRGeoJSON()

    def __repr__(self):
        return str(self.toRGeoJSON())

    def __str__(self):
        return f"{self.__class__.__name__}(coordinates={self.coordinates.tolist()})"


cdef class MultiRGJGeometry(RGJGeometry):
    """Base for geometries composed of multiple sub-parts."""

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        raise NotImplementedError

    def contact_point(self, x, min_dist_select=True, **kwargs):
        vectors = self.repulsion_vector(x=x, min_dist_select=min_dist_select, **kwargs)
        if min_dist_select:
            return x - vectors
        n = len(x)
        points_idx = np.tile(np.arange(n), len(vectors) // n)
        return x[points_idx] - vectors
