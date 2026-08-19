# cython: language_level=3
"""GeometryCollectionRGJ."""

import numpy as np

from larp.field.geometry.base cimport MultiRGJGeometry


cdef class GeometryCollectionRGJ(MultiRGJGeometry):
    RGJType = "GeometryCollection"

    cdef public list rgjs
    cdef public object inv_repulsions
    cdef public object grad_matrixes

    def __init__(self, geometries, properties=None, **kwargs):
        # Imported lazily to avoid a circular import: larp.field.geometry's
        # __init__ builds _RGJ_TYPES from this module (among others), so it
        # can't be imported at this module's top level.
        from larp.field.geometry import _RGJ_TYPES

        self.properties = {} if properties is None else properties
        self.rgjs = [_RGJ_TYPES[rgj_dict["type"]](**rgj_dict) for rgj_dict in geometries]

        self.inv_repulsions = np.array([rgj.get_dist_matrix(scaled=True, inverted=True) for rgj in self.rgjs])
        self.grad_matrixes = np.array([rgj.grad_matrix for rgj in self.rgjs])

        bbox = np.concatenate([rgj.bbox for rgj in self.rgjs], 0).reshape(-1, 2)
        self.bbox = np.array([bbox.min(0), bbox.max(0)])

    def set_coordinates(self, new_coords):
        raise NotImplementedError

    def set_repulsion(self, new_repulsion):
        for rgj in self.rgjs:
            rgj.set_repulsion(new_repulsion)

    def in_bbox(self, x):
        return any(rgj.in_bbox(x) for rgj in self.rgjs)

    def get_dist_matrix(self, scaled=True, inverted=True):
        return np.array([rgj.get_dist_matrix(scaled=scaled, inverted=inverted) for rgj in self.rgjs])

    def get_center_point(self):
        coords = np.reshape(np.array([rgj.get_center_point() for rgj in self.rgjs]), (-1, 2))
        return (coords.min(0) + coords.max(0)) / 2.0

    def squared_dist(self, x, scaled=True, inverted=True, return_reference=False, **kwargs):
        dists = np.stack([rgj.squared_dist(x, scaled=scaled, inverted=inverted) for rgj in self.rgjs], axis=1)
        if return_reference:
            min_idxs = np.argmin(dists, axis=1)
            return dists[np.arange(len(dists)), min_idxs], min_idxs
        return np.min(dists, axis=1)

    def repulsion_vector(self, x, min_dist_select=True, **kwargs):
        vectors = [rgj.repulsion_vector(x, min_dist_select=min_dist_select, **kwargs) for rgj in self.rgjs]
        vectors = np.stack(vectors, axis=0)

        if min_dist_select:
            vectors = vectors.swapaxes(0, 1)
            nvectors = np.einsum('ijk,lik->lij', self.inv_repulsions, vectors)
            dist = (vectors * nvectors).sum(-1)
            select = dist.argmin(1)
            vectors = vectors[np.arange(len(select)), select]

        return vectors.reshape(-1, 2)

    def gradient(self, x, **kwargs):
        _, dist_idxs = self.squared_dist(x, return_reference=True, **kwargs)
        repulsion_vector = self.repulsion_vector(x, min_dist_select=True, **kwargs)
        return -self.eval(x=x).reshape(-1, 1) * (np.einsum('ijk,ik->ij', self.grad_matrixes[dist_idxs], repulsion_vector))

    def toRGeoJSON(self):
        return {
            "type": "Feature",
            "properties": self.properties,
            "geometry": {
                "type": self.RGJType,
                "geometries": [rgj.toRGeoJSON()["geometry"] for rgj in self.rgjs],
            },
        }
