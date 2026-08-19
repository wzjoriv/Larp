# cython: language_level=3
"""
Cython port of larp.quad's QuadTree/QuadNode.

`field` is kept as a generic Python object (not a compile-time cimport of
RiskField) so this module and larp.field_cy.risk_field can reference each
other without a circular compile-time dependency -- all cross-calls go
through ordinary Python attribute/method access, exactly as the original
pure-Python code did.
"""
import numpy as np
cimport numpy as cnp

from larp.field_cy import kernels

cnp.import_array()

_CHD_TO_IDX = {'tl': 0, 'tr': 1, 'bl': 2, 'br': 3}
_NGH_TO_IDX = {'tl': 0, 't': 1, 'tr': 2, 'r': 3, 'br': 4, 'b': 5, 'bl': 6, 'l': 7}


cdef class QuadNode:

    chdToIdx = _CHD_TO_IDX
    nghToIdx = _NGH_TO_IDX

    cdef public object center_point
    cdef public double size
    cdef public bint leaf
    cdef public int boundary_zone
    cdef public double boundary_max_range
    cdef public object rgj_idx
    cdef public object rgj_risks
    cdef public list children
    cdef public list neighbors

    def __init__(self, center_point, size):
        self.center_point = np.atleast_1d(center_point).astype(float)
        self.size = size
        self.leaf = False
        self.boundary_zone = 0
        self.boundary_max_range = 1.0

        self.rgj_idx = np.array([], dtype=int)
        self.rgj_risks = np.array([], dtype=float)

        self.children = [None] * len(_CHD_TO_IDX)
        self.neighbors = [None] * len(_NGH_TO_IDX)

    def __getitem__(self, idx):
        """
        If list or tuple given, then neighbors considered. Else, children will be considered.
        """
        if isinstance(idx, (list, tuple)):
            n = len(idx)
            out = [None] * n
            for i in range(n):
                id_ = _NGH_TO_IDX[idx[i]] if not isinstance(idx[i], int) else idx[i]
                out[i] = self.neighbors[id_]
            return out
        else:
            idx = _CHD_TO_IDX[idx] if not isinstance(idx, int) else idx
            return self.children[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, (list, tuple)):
            for id_ in idx:
                id_ = _NGH_TO_IDX[id_] if not isinstance(id_, int) else id_
                self.neighbors[id_] = value
        else:
            idx = _CHD_TO_IDX[idx] if not isinstance(idx, int) else idx
            self.children[idx] = value

    def __lt__(self, other):
        return self.boundary_max_range < other.boundary_max_range

    def __hash__(self):
        # Defining __lt__ makes Cython emit a richcompare slot, which
        # otherwise suppresses the default identity-based hash needed to
        # put QuadNodes in the QuadTree.leaves set.
        return id(self)

    def get_boundaries(self):
        """Returns (xmin, ymin, xmax, ymax) boundaries of the quad."""
        cx, cy = self.center_point
        half = self.size / 2.0
        return np.array([cx - half, cy - half, cx + half, cy + half])

    def in_bbox(self, points):
        points = np.atleast_2d(points)
        xmin, ymin, xmax, ymax = self.get_boundaries()
        return (points[:, 0] >= xmin) & (points[:, 1] >= ymin) & (points[:, 0] <= xmax) & (points[:, 1] <= ymax)

    def get_shared_edge(self, neighbor):
        """
        Computes the shared boundary segment between this quad and a neighboring quad.
        """
        if neighbor is None:
            return None

        b1 = self.get_boundaries()
        b2 = neighbor.get_boundaries()

        x0 = max(b1[0], b2[0])
        y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2])
        y1 = min(b1[3], b2[3])

        if not np.isclose(x0, x1) and not np.isclose(y0, y1):
            return None

        return np.array([[x0, y0], [x1, y1]])

    def to_boundary_lines(self, margin=0.1):
        size2 = self.size / 2.0 - margin
        offset = np.array([
            [-1.0, 1.0],
            [ 1.0, 1.0],
            [ 1.0,-1.0],
            [-1.0,-1.0],
            [-1.0, 1.0],
        ]) * size2
        path = self.center_point + offset
        return path[:, 0], path[:, 1]

    def __str__(self):
        return f"Qd({self.center_point.tolist()}, {self.size})"


cdef class QuadTree:

    MAX_DEPTH = 1000

    cdef public object field
    cdef public double min_sector_size
    cdef public double max_sector_size
    cdef public double size
    cdef public double risk_epsilon
    cdef public double conservative_tol
    cdef public bint conservative
    cdef public object root
    cdef public set leaves
    cdef Py_ssize_t quad_idx
    cdef list leaves_list

    # Flattened array view of the tree, used by find_quad/find_quads_chain
    # for a nogil traversal. Rebuilt lazily whenever _flat_dirty is set by a
    # structural change (build/mark_leaf/replace_branch).
    cdef bint _flat_dirty
    cdef list _flat_nodes
    cdef object _flat_cx
    cdef object _flat_cy
    cdef object _flat_leaf
    cdef object _flat_child_tl
    cdef object _flat_child_tr
    cdef object _flat_child_bl
    cdef object _flat_child_br
    cdef int _flat_max_depth

    def __init__(self, field,
                 minimum_length_limit=None,
                 maximum_length_limit=np.inf,
                 risk_epsilon=0.01,
                 conservative_tol=0.05,
                 size=None,
                 conservative=False,
                 build_tree=True):

        self.field = field
        self.min_sector_size = np.min(field.size) / 128.0 if minimum_length_limit is None else minimum_length_limit
        self.max_sector_size = maximum_length_limit
        self.size = size or np.max(self.field.size)

        self.risk_epsilon = risk_epsilon
        self.conservative_tol = conservative_tol
        self.conservative = conservative

        self.root = None
        self.leaves = set()
        self._flat_dirty = True

        if build_tree:
            self.build()

    def __iter__(self):
        self.quad_idx = 0
        self.leaves_list = list(self.leaves)
        return self

    def __next__(self):
        if self.quad_idx >= len(self):
            raise StopIteration
        out = self.leaves_list[self.quad_idx]
        self.quad_idx += 1
        return out

    def __len__(self):
        return len(self.leaves)

    def mark_leaf(self, quad):
        quad.leaf = True
        self.leaves.add(quad)
        self._flat_dirty = True

    def __approximated_PF_risk__(self, center_point, size, filter_idx=None):
        rep_vectors, refs_idxs = self.field.repulsion_vectors([center_point], filted_idx=filter_idx, min_dist_select=True, return_reference=True)

        dist_sqr = (rep_vectors * rep_vectors).sum(1)
        forbidden_select = dist_sqr <= (size * size) / 2.0

        risks = np.zeros(len(filter_idx), dtype=float)
        risks[forbidden_select] = np.inf

        not_forbidden = ~forbidden_select
        if not_forbidden.any():
            rgjs_idx = filter_idx[not_forbidden]
            vectors = rep_vectors[not_forbidden].reshape(-1, 2)
            uni_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

            worst_dist_sqr = self.field.squared_dist_per(center_point - uni_vectors * (size / np.sqrt(2)), idxs=rgjs_idx).ravel()
            risks[not_forbidden] = np.exp(-worst_dist_sqr)

        return risks, rep_vectors, refs_idxs

    def __build__(self, center_point, size, filter_idx):

        quad = QuadNode(center_point=center_point, size=size)
        filter_n = len(filter_idx)

        if filter_n:
            risks, rep_vectors, refs_idxs = self.__approximated_PF_risk__(center_point=center_point, size=size, filter_idx=filter_idx)

            select = risks >= self.risk_epsilon
            quad.rgj_idx = filter_idx[select]
            quad.rgj_risks = risks[select]
            quad.boundary_max_range = float(quad.rgj_risks.max()) if len(quad.rgj_risks) else 0.0
            quad.boundary_zone = 0 if np.isinf(quad.boundary_max_range) else 1
        else:
            quad.boundary_zone = 1
            quad.boundary_max_range = 0.0

        size2 = size / 2.0
        if size <= self.max_sector_size:
            if size2 < self.min_sector_size or len(quad.rgj_idx) == 0:
                self.mark_leaf(quad)
                return quad
            if self.conservative and quad.boundary_zone != 0:
                governing = risks == quad.boundary_max_range
                vectors = rep_vectors[governing].reshape(-1, 2)
                uni_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

                opposite_dist_sqr = self.field.squared_dist_per(center_point + uni_vectors * (size / np.sqrt(2)), idxs=refs_idxs[governing]).ravel()
                best_case_risk = np.exp(-opposite_dist_sqr).max()

                if (quad.boundary_max_range - best_case_risk) < self.conservative_tol:
                    self.mark_leaf(quad)
                    return quad

        size4 = size2 / 2.0
        quad['tl'] = self.__build__(center_point + np.array([-size4, size4]), size2, quad.rgj_idx)
        quad['tr'] = self.__build__(center_point + np.array([ size4, size4]), size2, quad.rgj_idx)
        quad['bl'] = self.__build__(center_point + np.array([-size4,-size4]), size2, quad.rgj_idx)
        quad['br'] = self.__build__(center_point + np.array([ size4,-size4]), size2, quad.rgj_idx)

        return quad

    def build(self):
        self.leaves = set()
        self.root = self.__build__(self.field.center_point, self.size, np.arange(len(self.field)))
        self._flat_dirty = True
        return self.root

    def to_boundary_lines_collection(self, margin=0.1):
        lines = [quad.to_boundary_lines(margin=margin) for quad in self.leaves]
        return [path for line in lines for path in line]

    def replace_branch(self, rootquad, child, branch):
        if rootquad[child] is not None:
            self.leaves -= self.search_leaves(rootquad[child])

        rootquad[child] = branch
        new_leaves = self.search_leaves(rootquad[child])
        self.leaves.update(new_leaves)
        self._flat_dirty = True

    def get_quad_maximum_range(self):
        return np.array([quad.boundary_max_range for quad in self.leaves])

    def _ensure_flat(self):
        """
        Flatten the QuadNode tree into typed arrays (center_point x/y, leaf
        flag, 4 child indices) so find_quad/find_quads_chain can walk it in
        a nogil C loop (larp.field_cy.kernels.quadtree_find_leaf/_chain)
        instead of Python-level recursion. Rebuilt lazily -- cheap relative
        to the query volume it serves, since mutations are comparatively
        rare (see mark_leaf/build/replace_branch setting _flat_dirty).
        """
        if not self._flat_dirty:
            return

        nodes: list = []
        cx: list = []
        cy: list = []
        leaf: list = []
        child_tl: list = []
        child_tr: list = []
        child_bl: list = []
        child_br: list = []
        max_depth_box = [0]

        def visit(quad, depth):
            idx = len(nodes)
            nodes.append(quad)
            cx.append(float(quad.center_point[0]))
            cy.append(float(quad.center_point[1]))
            leaf.append(1 if quad.leaf else 0)
            child_tl.append(-1)
            child_tr.append(-1)
            child_bl.append(-1)
            child_br.append(-1)
            if depth > max_depth_box[0]:
                max_depth_box[0] = depth

            if not quad.leaf:
                for key, slot in (('tl', child_tl), ('tr', child_tr), ('bl', child_bl), ('br', child_br)):
                    child = quad[key]
                    if child is not None:
                        slot[idx] = visit(child, depth + 1)

            return idx

        if self.root is not None:
            visit(self.root, 0)

        self._flat_nodes = nodes
        self._flat_cx = np.array(cx, dtype=np.float64)
        self._flat_cy = np.array(cy, dtype=np.float64)
        self._flat_leaf = np.array(leaf, dtype=np.uint8)
        self._flat_child_tl = np.array(child_tl, dtype=np.int32)
        self._flat_child_tr = np.array(child_tr, dtype=np.int32)
        self._flat_child_bl = np.array(child_bl, dtype=np.int32)
        self._flat_child_br = np.array(child_br, dtype=np.int32)
        self._flat_max_depth = max_depth_box[0]
        self._flat_dirty = False

    def find_quad(self, x, max_depth=1000):
        """
        Efficiently finds the quad node for each point, minimizing redundant traversal.
        """
        x = np.atleast_2d(x).astype(np.float64)

        if self.root is None:
            return [None] * len(x)

        self._ensure_flat()

        idxs = kernels.quadtree_find_leaf(
            x, self._flat_cx, self._flat_cy, self._flat_leaf,
            self._flat_child_tl, self._flat_child_tr, self._flat_child_bl, self._flat_child_br,
            max_depth,
        )

        nodes = self._flat_nodes
        return [nodes[i] if i >= 0 else None for i in idxs]

    def find_quads_chain(self, x, max_depth=None):
        """
        Efficiently finds the full quad traversal chain (from root to final quad) for each point.
        """
        x = np.atleast_2d(x).astype(np.float64)
        if max_depth is None:
            max_depth = QuadTree.MAX_DEPTH

        if self.root is None:
            return [[None] for _ in range(len(x))]

        self._ensure_flat()

        cols = min(max_depth, self._flat_max_depth) + 1
        chain, lengths = kernels.quadtree_find_chain(
            x, self._flat_cx, self._flat_cy, self._flat_leaf,
            self._flat_child_tl, self._flat_child_tr, self._flat_child_bl, self._flat_child_br,
            max_depth, cols,
        )

        nodes = self._flat_nodes
        results = []
        for i in range(len(x)):
            row = chain[i, :lengths[i]]
            results.append([nodes[j] if j >= 0 else None for j in row])
        return results

    def __search_leaves__(self, quad, depth=0, max_depth=10000):
        if quad is None:
            raise TypeError(f"Branch missing leaf for quad {str(quad)}")
        if quad.leaf or depth >= max_depth:
            return [quad]

        out = []
        for child in quad.children:
            out.extend(self.__search_leaves__(child, depth=depth + 1, max_depth=max_depth))
        return out

    def search_leaves(self, quad=None, max_depth=None):
        quad = self.root if quad is None else quad
        max_depth = QuadTree.MAX_DEPTH if max_depth is None else max_depth
        return set(self.__search_leaves__(quad, depth=0, max_depth=max_depth))

    def get_quad_zones(self):
        return np.array([quad.boundary_zone for quad in self.leaves], dtype=int)

    def to_dict(self):
        def __save_quad__(quad):
            if quad is None:
                return None
            return {
                'center_point': quad.center_point,
                'size': quad.size,
                'leaf': quad.leaf,
                'boundary_zone': quad.boundary_zone,
                'boundary_max_range': quad.boundary_max_range,
                'rgj_idx': quad.rgj_idx,
                'rgj_risks': quad.rgj_risks,
                'children': [__save_quad__(child) for child in quad.children],
            }

        return {
            'field': self.field.toRGeoJSON(),
            'min_sector_size': self.min_sector_size,
            'max_sector_size': self.max_sector_size,
            'size': self.size,
            'risk_epsilon': self.risk_epsilon,
            'conservative_tol': self.conservative_tol,
            'conservative': self.conservative,
            'root': __save_quad__(self.root),
        }

    def from_dict(self, data):
        def __load_quad__(quad_data):
            if quad_data is None:
                return None
            quad = QuadNode(center_point=quad_data['center_point'], size=quad_data['size'])
            quad.leaf = quad_data['leaf']
            quad.boundary_zone = quad_data['boundary_zone']
            quad.boundary_max_range = quad_data['boundary_max_range']
            quad.rgj_idx = quad_data['rgj_idx']
            quad.rgj_risks = quad_data['rgj_risks']
            quad.children = [__load_quad__(child) for child in quad_data['children']]
            return quad

        self.min_sector_size = data['min_sector_size']
        self.max_sector_size = data['max_sector_size']
        self.size = data['size']
        self.risk_epsilon = data['risk_epsilon']
        self.conservative_tol = data['conservative_tol']
        self.conservative = data['conservative']
        self.root = __load_quad__(data['root'])
        self.leaves = self.search_leaves()

    def to_image(self, return_zone=False, return_extent=True, max_depth=None):
        """
        Render a top-down raster image of the quadtree zoning layout.
        """
        resolution = 2 ** int(np.floor(np.log2(self.root.size / self.min_sector_size))) + 1
        pixel_size = self.root.size / resolution

        zone_image = np.ones((resolution, resolution), dtype=np.int8)
        risk_image = np.zeros((resolution, resolution), dtype=float)

        half_size = self.root.size / 2.0
        lower_bound = self.root.center_point - half_size
        upper_bound = self.root.center_point + half_size

        quads = self.leaves if max_depth is None else self.search_leaves(max_depth=max_depth)

        for quad in quads:
            if len(quad.rgj_idx) == 0:
                continue

            quad_half = quad.size / 2.0
            x0 = int((quad.center_point[0] - quad_half - lower_bound[0]) / pixel_size)
            y0 = int((upper_bound[1] - (quad.center_point[1] + quad_half)) / pixel_size)
            block_size = max(1, int(np.floor(quad.size / pixel_size)))

            x1 = min(x0 + block_size, resolution)
            y1 = min(y0 + block_size, resolution)

            zone_image[y0:y1, x0:x1] = quad.boundary_zone
            risk_image[y0:y1, x0:x1] = quad.boundary_max_range

        if return_zone:
            image = zone_image
        else:
            image = risk_image
            image[zone_image == 0] = np.nan

        if return_extent:
            return image, [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1]]
        return image

    def quad_to_image(self, quad=None, resolution=200, margin=0.0):
        if quad is None:
            quad = self.root

        return self.field.to_image(resolution=resolution,
                                    margin=margin,
                                    center_point=quad.center_point,
                                    size=[quad.size] * 2,
                                    filted_idx=quad.rgj_idx)
