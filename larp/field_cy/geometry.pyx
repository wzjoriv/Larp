# cython: language_level=3
"""
Cython port of larp.field's RGJGeometry hierarchy, restricted to geometries
with a direct GeoJSON equivalent: Point, LineString, Polygon, MultiPoint,
MultiLineString, MultiPolygon, GeometryCollection.

Rectangle and Ellipse (non-standard extensions in larp/field.py) are not
carried over -- the new architecture only supports standard GeoJSON shapes.

Hot numeric loops (segment projection, point-in-polygon) are delegated to
larp.field_cy.kernels; everything else (bookkeeping, bbox, RGeoJSON I/O)
stays as ordinary Cython-compiled Python logic, since that's not where the
measured speedups come from (see benchmark/bench_field_cy.py).
"""

import numpy as np
cimport numpy as cnp

from larp.field_cy import kernels

cnp.import_array()


cdef class RGJGeometry:
    """Base class for all Cython RGJ geometries."""

    RGJType = None

    cdef public object coordinates
    cdef public object repulsion
    cdef public object inv_repulsion
    cdef public object eye_repulsion
    cdef public object grad_matrix
    cdef public object properties
    cdef public object bbox

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


cdef class PointRGJ(RGJGeometry):
    RGJType = "Point"

    def repulsion_vector(self, x, **kwargs):
        return np.atleast_2d(x).astype(np.float64) - self.coordinates


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


cdef class GeometryCollectionRGJ(MultiRGJGeometry):
    RGJType = "GeometryCollection"

    cdef public list rgjs
    cdef public object inv_repulsions
    cdef public object grad_matrixes

    def __init__(self, geometries, properties=None, **kwargs):
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


_RGJ_TYPES = {
    "Point": PointRGJ,
    "LineString": LineStringRGJ,
    "Polygon": PolygonRGJ,
    "MultiPoint": MultiPointRGJ,
    "MultiLineString": MultiLineStringRGJ,
    "MultiPolygon": MultiPolygonRGJ,
    "GeometryCollection": GeometryCollectionRGJ,
}
