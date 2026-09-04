# cython: language_level=3
"""
RiskField: a collection of RGJ geometries with optional automatic quad
decomposition.

    RiskField(rgjs)                                  # no quadtree -- straight loop over self.rgjs
    RiskField(rgjs, minimum_cell_size=2.0)            # quadtree built with an explicit minimum cell size
    RiskField(rgjs, minimum_cell_size=...)            # quadtree built with an auto-picked minimum cell size

When `minimum_cell_size` is None (the default), every method loops
directly over `self.rgjs`. Otherwise a quadtree is built and queries are
routed through it to only consider RGJs relevant to each query
point/region -- pass a float for an explicit minimum cell size, or `...`
(Ellipsis) to have QuadTree pick one from the field's own size.
"""

import warnings
from collections.abc import Iterable

import numpy as np
cimport numpy as cnp

import larp.fn as lpf
from larp.field.geometry import RGJGeometry, _RGJ_TYPES
from larp.field.quadtree import QuadTree, QuadNode

cnp.import_array()


cdef class RiskField:

    cdef public list rgjs
    cdef public object center_point
    cdef public object size
    cdef public object bbox
    cdef public object extra_info
    cdef public object quadtree

    cdef public object minimum_cell_size
    cdef public object maximum_cell_size
    cdef public double risk_epsilon
    cdef public double conservative_tol
    cdef public bint conservative

    cdef bint _reload_center
    cdef Py_ssize_t rgj_idx

    def __init__(self, rgjs=None, center_point=None, size=None, properties=None, extra_info=None,
                 minimum_cell_size=None, maximum_cell_size=np.inf,
                 risk_epsilon=0.01, conservative_tol=0.05, conservative=False):
        rgjs = [] if rgjs is None else rgjs

        self.rgjs = []
        self._reload_center = False
        self.center_point = center_point
        self.extra_info = {} if extra_info is None else extra_info
        self.bbox = np.array([[None, None], [None, None]])

        self.minimum_cell_size = minimum_cell_size
        self.maximum_cell_size = maximum_cell_size
        self.risk_epsilon = risk_epsilon
        self.conservative_tol = conservative_tol
        self.conservative = conservative
        self.quadtree = None

        if size is None:
            self.size = size
        elif np.isscalar(size):
            self.size = np.array([size, size], dtype=float)
        else:
            self.size = np.atleast_1d(np.array(size, dtype=float))

        if properties is None or (len(rgjs) and isinstance(rgjs[0], RGJGeometry)):
            for rgj in rgjs:
                self.addRGJ(rgj=rgj, reload_bbox=False, _skip_quad=True)
        else:
            for rgj, proper in zip(rgjs, properties):
                self.addRGJ(rgj=rgj, properties=proper, reload_bbox=False, _skip_quad=True)

        self.reload_bbox()

        if self.center_point is None:
            self._reload_center = True
            if len(rgjs) > 0:
                self.center_point, suggest_size = self.__calculate_center_point__(suggest_size=True)
                self.size = np.array([max(suggest_size)] * 2, dtype=float) if self.size is None else self.size
        else:
            self._reload_center = False
            if len(rgjs) > 0:
                suggest_size = np.array([max(np.abs(self.bbox - self.center_point).reshape(-1)) * 2] * 2)
                self.size = suggest_size if self.size is None else self.size

        if self.minimum_cell_size is not None and len(self.rgjs) >= 0:
            self._build_quadtree()

    # ---------------- quadtree management ----------------

    def _build_quadtree(self):
        self.quadtree = QuadTree(
            self,
            minimum_length_limit=self.minimum_cell_size,
            maximum_length_limit=self.maximum_cell_size,
            risk_epsilon=self.risk_epsilon,
            conservative_tol=self.conservative_tol,
            conservative=self.conservative,
            build_tree=True,
        )

    def enable_quadtree(self, minimum_cell_size = Ellipsis, maximum_cell_size=np.inf,
                         risk_epsilon=0.01, conservative_tol=0.05, conservative=False):
        """Build (or rebuild) the quadtree for this field."""
        self.minimum_cell_size = minimum_cell_size
        self.maximum_cell_size = maximum_cell_size
        self.risk_epsilon = risk_epsilon
        self.conservative_tol = conservative_tol
        self.conservative = conservative
        self._build_quadtree()

    def disable_quadtree(self):
        self.minimum_cell_size = None
        self.quadtree = None

    def __group_points_by_quads_with_rgjs(self, points, max_depth=3):
        points = np.atleast_2d(points).astype(float)
        quad_chains = self.quadtree.find_quads_chain(points, max_depth=max_depth)

        final_quads = []
        for chain in quad_chains:
            for node in reversed(chain):
                if len(node.rgj_idx):
                    final_quads.append(node)
                    break
            else:
                final_quads.append(chain[0])

        seen = {}
        unique_quads = []
        point_to_quad_idx = []
        for q in final_quads:
            if id(q) not in seen:
                seen[id(q)] = len(unique_quads)
                unique_quads.append(q)
            point_to_quad_idx.append(seen[id(q)])

        from collections import defaultdict
        quad_to_point_indices = defaultdict(list)
        for pt_idx, quad_idx in enumerate(point_to_quad_idx):
            quad_to_point_indices[quad_idx].append(pt_idx)

        return unique_quads, quad_to_point_indices

    # ---------------- container protocol ----------------

    def __getitem__(self, idxs):
        if isinstance(idxs, (int, np.integer)):
            return self.rgjs[int(idxs)]
        elif isinstance(idxs, Iterable) and not isinstance(idxs, (str, bytes)):
            return [self.rgjs[int(i)] for i in idxs]
        warnings.warn(f"Object of type {type(idxs)} not supported")
        return None

    def __iter__(self):
        self.rgj_idx = 0
        return self

    def __next__(self):
        if self.rgj_idx >= len(self):
            raise StopIteration
        out = self.rgjs[self.rgj_idx]
        self.rgj_idx += 1
        return out

    def __len__(self):
        return len(self.rgjs)

    # ---------------- bbox / center ----------------

    def __calculate_center_point__(self, suggest_size=False):
        center = np.sum(self.bbox, 0) / 2.0
        if suggest_size:
            suggest_size = self.bbox[1] - self.bbox[0]
            return center, suggest_size
        return center

    def set_bbox(self, x_min, y_min, x_max, y_max):
        self.bbox = np.array([[x_min, y_min], [x_max, y_max]])
        self.center_point = np.sum(self.bbox, 0) / 2.0
        self.size = self.bbox[1] - self.bbox[0]

    def get_extent(self, margin=0.0):
        n2 = self.size / 2.0
        loc_tl = np.array(self.center_point) + np.array([-n2[0] - margin, n2[1] + margin])
        loc_br = np.array(self.center_point) + np.array([n2[0] + margin, -n2[1] - margin])
        return [loc_tl[0], loc_br[0], loc_br[1], loc_tl[1]]

    def set_all_repulsion(self, new_repulsion):
        new_repulsion = np.array(new_repulsion)
        for rgj in self.rgjs:
            rgj.set_repulsion(new_repulsion)
        if self.quadtree is not None:
            self._build_quadtree()

    def reload_bbox(self):
        if len(self):
            bbox = np.concatenate([rgj.bbox.reshape(-1, 2) for rgj in self.rgjs], 0)
            self.bbox = np.array([bbox.min(0), bbox.max(0)])
        else:
            self.bbox = np.array([[None, None], [None, None]])
        return self.bbox

    def reload_center_point(self, toggle=True, recal_size=False):
        self._reload_center = toggle
        if toggle and len(self.rgjs) > 0:
            if recal_size:
                self.center_point, suggest_size = self.__calculate_center_point__(True)
                self.size = np.array([max(suggest_size)] * 2, dtype=float)
            else:
                self.center_point = self.__calculate_center_point__(False)
        return self.center_point

    # ---------------- mutation ----------------
    #
    # When a quadtree is present, add/remove mutate `self.quadtree` IN PLACE
    # (same object identity) rather than rebuilding it, so external holders
    # of the reference (e.g. QuadNetwork built on `field.quadtree`) observe
    # the update -- this mirrors the old QRiskField's incremental behavior.

    def _new_rgj(self, rgj, properties=None, **kwargs):
        if isinstance(rgj, RGJGeometry):
            return rgj
        if not isinstance(rgj, dict) or "type" not in rgj:
            raise ValueError("RGJ must be an RGJGeometry.")
        cls = _RGJ_TYPES.get(rgj["type"])
        if cls is None:
            raise ValueError(f"No RGJ class found for type {rgj['type']}")
        return cls(properties=properties, **{k: v for k, v in rgj.items() if k != "type"}, **kwargs)

    def addRGJ(self, rgj, properties=None, reload_bbox=True, _skip_quad=False, **kwargs):
        rgj = self._new_rgj(rgj, properties=properties, **kwargs)

        if not _skip_quad and self.quadtree is not None:
            new_field = RiskField(rgjs=[rgj])
            return self.addField(new_field=new_field, reload_bbox=reload_bbox)

        self.rgjs.append(rgj)
        if reload_bbox:
            self.reload_bbox()
        if self._reload_center:
            self.center_point = self.__calculate_center_point__()
        return [len(self.rgjs) - 1]

    def addField(self, new_field, reload_bbox=True):
        if self.quadtree is None:
            og_reload_center = self._reload_center
            self._reload_center = False
            try:
                for rgj in new_field:
                    self.addRGJ(rgj=rgj, reload_bbox=False, _skip_quad=True)
            finally:
                self._reload_center = og_reload_center

            if reload_bbox:
                self.reload_bbox()
            if self._reload_center:
                self.center_point = self.__calculate_center_point__()
            return np.arange(len(self) - len(new_field), len(self))

        if self.quadtree.conservative:
            warnings.warn("Quadtree made not conservative")
        self.quadtree.conservative = False

        n_original = len(self)

        og_reload_center = self._reload_center
        self._reload_center = False
        try:
            for rgj in new_field:
                self.rgjs.append(rgj)
        finally:
            self._reload_center = og_reload_center

        if reload_bbox:
            self.reload_bbox()
        if self._reload_center:
            self.center_point = self.__calculate_center_point__()

        # Align new_field's own geometry with this field, then build a
        # throwaway quadtree over just the new RGJs to merge in.
        new_field.reload_center_point(False)
        new_field.center_point = self.center_point
        new_field.size = self.size

        new_qtree = QuadTree(
            new_field,
            minimum_length_limit=self.quadtree.min_sector_size,
            maximum_length_limit=self.quadtree.max_sector_size,
            risk_epsilon=self.quadtree.risk_epsilon,
            conservative_tol=self.quadtree.conservative_tol,
            size=self.quadtree.size,
            build_tree=True,
            conservative=False,
        )

        def update_idx(quad):
            if quad is None or len(quad.rgj_idx) == 0:
                return
            quad.rgj_idx = quad.rgj_idx + n_original
            for child in quad.children:
                update_idx(child)

        update_idx(new_qtree.root)

        def update_quad(rootquad, newquad):
            if newquad is None or len(newquad.rgj_idx) == 0:
                return

            if newquad.boundary_zone == 0:
                rootquad.boundary_zone = 0
            rootquad.boundary_max_range = max(rootquad.boundary_max_range, newquad.boundary_max_range)

            if len(newquad.rgj_idx) > 0:
                rootquad.rgj_idx = np.concatenate([rootquad.rgj_idx, newquad.rgj_idx])
                rootquad.rgj_risks = np.concatenate([rootquad.rgj_risks, newquad.rgj_risks])

            if rootquad.leaf and not newquad.leaf:
                self.quadtree.leaves.remove(rootquad)
                rootquad.children = [None] * len(rootquad.chdToIdx)
                rootquad.neighbors = [None] * len(rootquad.nghToIdx)
                rootquad.leaf = False

            for child in ['tl', 'tr', 'bl', 'br']:
                nq = newquad[child]
                rq = rootquad[child]
                if rq is None:
                    self.quadtree.replace_branch(rootquad, child, nq)
                else:
                    update_quad(rq, nq)

        update_quad(self.quadtree.root, new_qtree.root)

        return np.arange(n_original, len(self))

    def delRGJ(self, idxs, reload_bbox=True, pop_field=False):
        idxs = np.atleast_1d(idxs).astype(int)
        idxs = np.unique(idxs % len(self))[::-1]

        if self.quadtree is None:
            removed = [self.rgjs[idx] for idx in idxs]
            for idx in idxs:
                del self.rgjs[idx]
            if self._reload_center:
                self.center_point = self.__calculate_center_point__()
            if reload_bbox:
                self.reload_bbox()
            return RiskField(rgjs=removed) if pop_field else None

        if self.quadtree.conservative:
            warnings.warn("Quadtree made non-conservative")
            self.quadtree.conservative = False

        removed = [self.rgjs[idx] for idx in idxs]

        for idx in idxs:
            del self.rgjs[idx]
        if self._reload_center:
            self.center_point = self.__calculate_center_point__()
        if reload_bbox:
            self.reload_bbox()

        total = len(self) + len(idxs)
        shift_map = np.arange(total)
        deleted = np.zeros(total, dtype=bool)
        deleted[idxs] = True
        shift_map = shift_map - np.cumsum(deleted)

        def shift_recursive(quad):
            if quad is None or quad.rgj_idx.size == 0:
                return
            quad.rgj_idx = shift_map[quad.rgj_idx]
            if not quad.leaf:
                for child in ['tl', 'tr', 'bl', 'br']:
                    shift_recursive(quad[child])

        def clean_quad(quad):
            if quad is None:
                return

            keep_mask = ~np.isin(quad.rgj_idx, idxs, assume_unique=True)

            if np.all(keep_mask):
                shift_recursive(quad)
                return

            quad.rgj_idx = shift_map[quad.rgj_idx[keep_mask]]
            quad.rgj_risks = quad.rgj_risks[keep_mask]
            if len(quad.rgj_risks) > 0:
                quad.boundary_max_range = float(quad.rgj_risks.max())
                quad.boundary_zone = 0 if np.isinf(quad.boundary_max_range) else 1
            else:
                quad.boundary_max_range = 0.0
                quad.boundary_zone = 1

            if (not quad.leaf) and quad.size <= self.quadtree.max_sector_size and len(quad.rgj_idx) == 0:
                self.quadtree.leaves -= self.quadtree.search_leaves(quad)
                quad.children = [None] * len(quad.chdToIdx)
                quad.neighbors = [None] * len(quad.nghToIdx)
                self.quadtree.mark_leaf(quad)
                return

            if not quad.leaf and keep_mask.size > 0:
                for child in ['tl', 'tr', 'bl', 'br']:
                    clean_quad(quad[child])

        clean_quad(self.quadtree.root)

        return RiskField(rgjs=removed) if pop_field else None

    # ---------------- spatial queries ----------------

    def in_bbox(self, point, filted_idx=None, max_depth=5):
        point = np.array(point)

        if self.quadtree is not None and filted_idx is None:
            quad = self.quadtree.find_quad([point], max_depth=max_depth)[0]
            if len(quad.rgj_idx):
                return any(self.rgjs[idx].in_bbox(point) for idx in quad.rgj_idx)
            return False

        rgjs = [self.rgjs[idx] for idx in filted_idx] if filted_idx is not None else self.rgjs
        return any(rgj.in_bbox(point) for rgj in rgjs)

    def find_bbox(self, point, filted_idx=None, max_depth=3):
        point = np.array(point)

        if self.quadtree is not None and filted_idx is None:
            quad_chain = self.quadtree.find_quads_chain([point], max_depth=max_depth)[0]
            searched = set()
            for quad in reversed(quad_chain):
                to_search = set(quad.rgj_idx) - searched
                if len(quad.rgj_idx):
                    searched.update(to_search)
                    found = np.array([idx for idx in to_search if self.rgjs[idx].in_bbox(point)])
                    if len(found):
                        return found
            return np.array([idx for idx in range(len(self)) if self.rgjs[idx].in_bbox(point)])

        if filted_idx is not None:
            return np.array([idx for idx in filted_idx if self.rgjs[idx].in_bbox(point)])
        return np.nonzero([rgj.in_bbox(point) for rgj in self.rgjs])[0]

    def repulsion_vectors(self, points, filted_idx=None, min_dist_select=True, return_reference=False, max_depth=3):
        points = np.atleast_2d(points).astype(float)
        if not len(self):
            return points * np.inf

        if self.quadtree is not None and filted_idx is None:
            all_vectors, all_refs = [], []
            unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

            for quad_idx, pt_indices in quad_to_point_indices.items():
                quad = unique_quads[quad_idx]
                rgj_indices = quad.rgj_idx
                if not len(rgj_indices):
                    continue
                res = self._repulsion_vectors_flat(points[pt_indices], filted_idx=rgj_indices,
                                                     min_dist_select=min_dist_select, return_reference=return_reference)
                if return_reference:
                    all_vectors.append(res[0])
                    all_refs.append(res[1])
                else:
                    all_vectors.append(res)

            if not all_vectors:
                return (np.zeros((0, 2)), np.array([], dtype=int)) if return_reference else np.zeros((0, 2))

            final_vectors = np.concatenate(all_vectors, axis=0)
            if return_reference:
                return final_vectors, np.concatenate(all_refs, axis=0)
            return final_vectors

        return self._repulsion_vectors_flat(points, filted_idx=filted_idx, min_dist_select=min_dist_select, return_reference=return_reference)

    def _repulsion_vectors_flat(self, points, filted_idx=None, min_dist_select=True, return_reference=False):
        filted_idx = filted_idx if filted_idx is not None else list(range(len(self)))

        if return_reference:
            idxs = []
            vectors = []
            for idx in filted_idx:
                v = self.rgjs[idx].repulsion_vector(points, min_dist_select=min_dist_select).reshape(-1, 2)
                idxs.extend([idx] * len(v))
                vectors.append(v)
            return np.concatenate(vectors, axis=0), np.array(idxs, dtype=int)

        rgjs = [self.rgjs[idx] for idx in filted_idx]
        return np.concatenate([rgj.repulsion_vector(points, min_dist_select=min_dist_select).reshape(-1, 2) for rgj in rgjs], axis=0)

    def gradient(self, points, min_dist_select=True, max_depth=2):
        points = np.atleast_2d(points).astype(float)
        if not len(self):
            return points * 0.0

        grad = np.zeros((len(points), 2), dtype=float)

        if self.quadtree is not None:
            unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)
            for quad_idx, pt_indices in quad_to_point_indices.items():
                quad = unique_quads[quad_idx]
                pts = points[pt_indices]
                for rgj_idx in quad.rgj_idx:
                    grad[pt_indices] += self.rgjs[rgj_idx].gradient(pts, min_dist_select=min_dist_select)
            return grad

        _, grad_idxs = self.squared_dist(points=points, return_reference=True)
        for idx in set(grad_idxs):
            select = (grad_idxs == idx)
            grad[select] = self.rgjs[idx].gradient(points[select], min_dist_select=min_dist_select)
        return grad

    def contact_points(self, points, filted_idx=None, min_dist_select=True, return_reference=False, max_depth=3):
        points = np.atleast_2d(points).astype(float)
        if not len(self):
            return points * np.inf

        if self.quadtree is not None and filted_idx is None:
            all_points, all_refs = [], []
            unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

            for quad_idx, pt_indices in quad_to_point_indices.items():
                quad = unique_quads[quad_idx]
                rgj_indices = quad.rgj_idx
                if not len(rgj_indices):
                    continue
                res = self._contact_points_flat(points[pt_indices], filted_idx=rgj_indices,
                                                  min_dist_select=min_dist_select, return_reference=return_reference)
                if return_reference:
                    all_points.append(res[0])
                    all_refs.append(res[1])
                else:
                    all_points.append(res)

            if not all_points:
                return (np.zeros((0, 2)), np.array([], dtype=int)) if return_reference else np.zeros((0, 2))

            final_points = np.concatenate(all_points, axis=0)
            if return_reference:
                return final_points, np.concatenate(all_refs, axis=0)
            return final_points

        return self._contact_points_flat(points, filted_idx=filted_idx, min_dist_select=min_dist_select, return_reference=return_reference)

    def _contact_points_flat(self, points, filted_idx=None, min_dist_select=True, return_reference=False):
        filted_idx = range(len(self)) if filted_idx is None else filted_idx

        if return_reference:
            idxs = []
            pts = []
            for idx in filted_idx:
                v = self.rgjs[idx].contact_point(points, min_dist_select=min_dist_select).reshape(-1, 2)
                idxs.extend([idx] * len(v))
                pts.append(v)
            return np.concatenate(pts, axis=0), np.array(idxs, dtype=int)

        rgjs = [self.rgjs[idx] for idx in filted_idx]
        return np.concatenate([rgj.contact_point(points, min_dist_select=min_dist_select).reshape(-1, 2) for rgj in rgjs], axis=0)

    def __call__(self, points, filted_idx=None, max_depth=2):
        return self.eval(points, filted_idx, max_depth)

    def eval(self, points, filted_idx=None, max_depth=2):
        points = np.atleast_2d(points).astype(float)

        if self.quadtree is not None and filted_idx is None:
            n_points = len(points)
            if not len(self.rgjs):
                return np.zeros(n_points, dtype=float)

            unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)
            results = np.zeros(n_points, dtype=float)

            for quad_idx, pt_indices in quad_to_point_indices.items():
                quad = unique_quads[quad_idx]
                rgj_indices = quad.rgj_idx
                group_points = points[pt_indices]
                if len(rgj_indices):
                    results[pt_indices] = self.eval(group_points, filted_idx=rgj_indices)
            return results

        rgjs = [self.rgjs[idx] for idx in filted_idx] if filted_idx is not None else self.rgjs
        if not len(rgjs):
            return points.sum(1) * 0.0
        return np.max(np.stack([rgj.eval(points) for rgj in rgjs], axis=1), axis=1)

    def eval_per(self, points, idxs=None):
        if len(points) != len(idxs):
            raise RuntimeError("The number of points doesn't match the number of indexes")

        points = np.atleast_2d(points).astype(float)
        n = len(points)
        idxs = np.array(idxs, dtype=int)

        evals = np.ones(n, dtype=points[0].dtype)
        for idx in set(idxs):
            select = idx == idxs
            evals[select] = self.rgjs[idx].eval(points[select])
        return evals

    def squared_dist(self, points, filted_idx=None, scaled=True, inverted=True, return_reference=False, max_depth=2):
        points = np.atleast_2d(points).astype(float)
        if not len(self):
            warnings.warn("There are not any RGJs elements in the field")
            if return_reference:
                return points.sum(1) * np.inf, -np.ones_like(points.sum(1))
            return points.sum(1) * np.inf

        dists = self.squared_dist_list(points=points, filted_idx=filted_idx, scaled=scaled, inverted=inverted, max_depth=max_depth)

        if return_reference:
            min_idxs = np.argmin(dists, axis=1)
            filted_idx = filted_idx if filted_idx is not None else np.arange(len(self))
            return dists[np.arange(len(dists)), min_idxs], np.asarray(filted_idx)[min_idxs]

        return np.min(dists, axis=1)

    def squared_dist_per(self, points, idxs=None, scaled=True, inverted=True):
        idxs = [] if idxs is None else idxs
        n = len(points)

        if n != len(idxs):
            if not len(idxs):
                raise RuntimeError("The number of points doesn't match the number of indexes")
            warnings.warn("The number of points doesn't match the number of indexes. Each point matched with each rgj")
            idxs = np.arange(n)

        points = np.array(points)
        idxs = np.array(idxs, dtype=int)

        dists = np.ones(n, dtype=points[0].dtype)
        for idx in set(idxs):
            select = idx == idxs
            dists[select] = self.rgjs[idx].squared_dist(points[select], scaled=scaled, inverted=inverted)
        return dists

    def squared_dist_list(self, points, filted_idx=None, scaled=True, inverted=True, max_depth=3):
        points = np.atleast_2d(points).astype(float)

        if self.quadtree is not None and filted_idx is None:
            n_points = len(points)
            total_rgjs = len(self.rgjs)
            if total_rgjs == 0:
                warnings.warn("There are no RGJs in the field.")
                return np.ones((n_points, 1)) * np.inf

            dist_matrix = np.ones((n_points, total_rgjs), dtype=np.float64) * np.inf
            unique_quads, quad_to_point_indices = self.__group_points_by_quads_with_rgjs(points, max_depth=max_depth)

            for quad_idx, pt_indices in quad_to_point_indices.items():
                quad = unique_quads[quad_idx]
                rgj_indices = quad.rgj_idx
                if not len(rgj_indices):
                    continue
                group_points = points[pt_indices]
                group_dists = np.stack(
                    [self.rgjs[i].squared_dist(group_points, scaled=scaled, inverted=inverted) for i in rgj_indices],
                    axis=1,
                )
                for i, pt_i in enumerate(pt_indices):
                    for j, rgj_i in enumerate(rgj_indices):
                        dist_matrix[pt_i, rgj_i] = group_dists[i, j]

            return dist_matrix

        rgjs = [self.rgjs[idx] for idx in filted_idx] if filted_idx is not None else self.rgjs
        if not len(rgjs):
            warnings.warn("There are not any RGJs elements in the field")
            return np.ones((len(points), len(rgjs))) * np.inf
        return np.stack([rgj.squared_dist(points, scaled=scaled, inverted=inverted) for rgj in rgjs], axis=1)

    # ---------------- route helpers ----------------

    def estimate_route_area(self, route, step=1e-3, n=0, scale_transform=lambda x: x, max_depth=3):
        route = np.array(route)
        points, step, _ = lpf.interpolate_along_route(route=route, step=step, n=n, return_step_n=True)
        points = points if n <= 0 else points[:-1]
        f_eval = scale_transform(self.eval(points=points, max_depth=max_depth))
        return f_eval.sum() * step

    def estimate_route_highest_risk(self, route, step=1e-2, n=0, scale_transform=lambda x: x, max_depth=3):
        route = np.array(route)
        points, step, _ = lpf.interpolate_along_route(route=route, step=step, n=n, return_step_n=True)
        points = points if n <= 0 else points[:-1]
        f_eval = scale_transform(self.eval(points=points, max_depth=max_depth))
        return f_eval.max()

    def to_image(self, resolution=200, margin=0.0, center_point=None, size=None, filted_idx=None, return_extent=True, max_depth=2):
        if center_point is None:
            if self.center_point is None:
                raise RuntimeError('Center point for field has not been defined')
            center_point = self.center_point

        if size is None:
            if self.size is None:
                raise RuntimeError('Size of field has not been defined')
            size = self.size
        else:
            size = np.array(size)

        n2 = size / 2.0
        loc_tl = np.array(center_point) + np.array([-n2[0] - margin, n2[1] + margin])
        loc_br = np.array(center_point) + np.array([n2[0] + margin, -n2[1] - margin])

        y_resolution = int(resolution * abs(loc_tl[1] - loc_br[1]) / abs(loc_br[0] - loc_tl[0]))
        xaxis = np.linspace(loc_tl[0], loc_br[0], resolution)
        yaxis = np.linspace(loc_tl[1], loc_br[1], y_resolution)

        xgrid, ygrid = np.meshgrid(xaxis, yaxis)
        points = np.vstack([xgrid.ravel(), ygrid.ravel()]).T

        image = self.eval(points, filted_idx=filted_idx, max_depth=max_depth).reshape((y_resolution, resolution))

        if return_extent:
            extent = np.reshape([loc_tl[0], loc_br[0], loc_br[1], loc_tl[1]], -1).tolist()
            return image, extent
        return image

    def toRGeoJSON(self, return_bbox=False):
        rgeojson = {
            'type': 'FeatureCollection',
            '_version_': "2D",
            'features': [rgj.toRGeoJSON() for rgj in self.rgjs],
            **self.extra_info,
        }
        if return_bbox:
            extent = self.get_extent()
            rgeojson["bbox"] = extent[::2] + extent[1::2]
        return rgeojson

    @property
    def __geo_interface__(self):
        """Standard geospatial interchange protocol (shapely/geopandas/fiona)."""
        return self.toRGeoJSON()
