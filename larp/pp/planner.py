from collections import defaultdict
import heapq
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
from larp.field import RiskField
from larp.fn import interpolate_along_route
from larp.pp.network import QuadNetwork

from larp.quad import QuadNode, QuadTree, QRiskField
from larp.types import Scaler, Point

"""
Author: Josue N Rivera

Module providing path planning algorithms. They can be over a riskd field or quadtree-based spatial network of the risk field.
Implements classic and multi-resolution planning strategies including A* and Dijkstra.
"""

PathAlg = Callable[[Point, Point, QuadNetwork, dict, Any], Optional[List[Point]]]
PathAlgArg = Union[str, PathAlg]

def __reconstruct_quad_path__(came_from:dict, current:QuadNode):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def __get_scaled_distance__(entry:Point, exit:Point, node_to:QuadNode, scaler:Optional[Scaler]=None, max_scale: float = np.inf):

        if scaler is None:
            multipler = 1.0
        else:
            multipler = np.inf if node_to.boundary_zone == 0 else scaler(node_to.boundary_max_range)
        
        multipler = min(multipler, max_scale)

        diff = exit - entry

        return multipler*np.linalg.norm(diff)

def optimize_path_via_edge_bundling(path: List[Point], quad_path: List[QuadNode], network: QuadNetwork) -> List[Point]:
    """
    Optimizes a path by "pulling" the string tight through the corridor of quads.
    It projects the straight line between segment endpoints onto the shared edges (portals)
    of the quads.
    """

    # 1. Validation
    if path is None or quad_path is None:
        return []
    
    if len(path) != len(quad_path) + 1:
        if len(path) < 2: 
            return path
        limit = min(len(path)-1, len(quad_path))
        quad_path = quad_path[:limit]

    optimized_points = [path[0]]
    current_idx = 0

    while current_idx < len(quad_path):
        
        next_idx = current_idx + 1
        
        # Look ahead loop: Extend segment as long as quads are connected
        while next_idx < len(quad_path):
            curr_q = quad_path[next_idx - 1]
            next_q = quad_path[next_idx]
            
            # Check if they share an edge or corner (are neighbors)
            # We relax the strict "alignment" to allow diagonal transitions
            if curr_q.get_shared_edge(next_q) is None:
                break
            next_idx += 1

        # 3. "Pull the String" logic
        start_point = optimized_points[-1]
        
        # Handle end of path boundary
        end_idx_in_path = next_idx if next_idx < len(path) else len(path) - 1
        end_point = path[end_idx_in_path]

        ideal_trajectory_vec = end_point - start_point
        total_dist = np.linalg.norm(ideal_trajectory_vec)
        
        if total_dist > 1e-6:
            # Iterate through the quads in this specific segment
            segment_quads = quad_path[current_idx:next_idx]
            
            accumulated_dist = 0.0
            
            for k in range(len(segment_quads) - 1):
                q_curr = segment_quads[k]
                q_next = segment_quads[k+1]
                
                dist_step = np.linalg.norm(q_next.center_point - q_curr.center_point)
                accumulated_dist += dist_step
                
                t = accumulated_dist / (total_dist + 1e-6) # Avoid div/0
                t = np.clip(t, 0.01, 0.99)
                
                ideal_point = start_point + t * ideal_trajectory_vec
                
                clamped_point = network.get_shared_entry_point(q_curr, q_next, ideal_point)
                
                if clamped_point is not None:
                    if np.linalg.norm(clamped_point - optimized_points[-1]) > 1e-3:
                         optimized_points.append(clamped_point)

        if np.linalg.norm(end_point - optimized_points[-1]) > 1e-3:
            optimized_points.append(end_point)
            
        # Advance
        current_idx = next_idx

    return optimized_points

def has_quad_zone_sight(
        p1: np.ndarray,
        p2: np.ndarray,
        network: QuadNetwork,
        step: Optional[float] = None,
        min_zone:int = 3
    ) -> Tuple[bool, Optional[int], Optional[QuadNode]]:

    """
    Returns whether the path from p1 to p2 maintains or increases zone (i.e., risk doesn't increase).

    Args:
        p1 (np.ndarray): Start point.
        p2 (np.ndarray): End point.
        network (QuadNetwork): The quad network to check in.
        step (float): Distance between sample points.

    Returns:
        bool: True if zone doesn't decrease, False otherwise.
    """
    if step is None:
        step = network.quadtree.min_sector_size / 4

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    if np.allclose(p1, p2):
        return True

    distance = np.linalg.norm(p2 - p1)
    num_samples = max(2, int(distance / step))
    alphas = np.linspace(0, 1, num_samples)
    points = (1 - alphas[:, None]) * p1 + alphas[:, None] * p2

    quads = network.find_quad(points)
    allowed_zone = max(quads[0].boundary_zone, min_zone)

    # Reduce to unique quads
    used = set()
    quads = [x for x in quads if x not in used and (used.add(x) or True)]

    # Accept quads that have a non-decending quad zone order 
    for quad in quads:
        if quad is None or quad.boundary_zone < allowed_zone:
            return False
        
        allowed_zone = quad.boundary_zone

    return True

def network_path_smoothing(path: List[Point], network: QuadNetwork, step: Optional[float] = None) -> np.ndarray:
    """
    Smooths a path by merging segments that are in the same zone or higher zone.
    """
    if len(path) <= 2:
        return np.array(path, dtype=float)

    optimized = [path[0]]

    i = 1
    while i < len(path) - 1:
        if not has_quad_zone_sight(optimized[-1], path[i+1], network=network, step=step):
            optimized.append(path[i])
        i += 1

    optimized.append(path[-1])
    return np.array(optimized, dtype=float)

class Planner():

    """
    Course path without dynamics included
    """

    def __init__(self, alg: Optional[PathAlgArg] = None):

        self.algs = defaultdict(lambda: None)
        self.alg = self.algs['custom']
        self.memory = {}

        if alg is not None:
            self.select_alg(alg=alg, reset_memory=False)

    def refresh(self):
        """
        Resets planner memory.
        """
        self.reset_memory()

    def reset_memory(self):
        """
        Clears any memory state stored across planning runs.
        """
        self.memory.clear()

    def add_alg(self, name: str, algorithm: PathAlg):
        """
        Adds a custom path planning algorithm.

        Args:
            name (str): Name identifier for the custom algorithm.
            algorithm (PathAlg): The algorithm function.
        """

        self.algs[name.lower()] = algorithm

    def select_alg(self, alg: PathAlgArg, reset_memory = True):
        """
        Selects a planning algorithm by name or function reference.

        Args:
            alg (PathAlgArg): Algorithm name or custom function.
            reset_memory (bool): Whether to reset internal memory state. Defaults to True.
        """

        if isinstance(alg, str):
            if alg not in self.algs.keys():
                raise KeyError("Selected algorithm is not available")

            self.alg = self.algs[alg.lower()]
        else:
            self.add_alg('custom', alg)
            self.alg = self.algs["custom"]

        if reset_memory:
            self.reset_memory()

    def find_path(self, start_point:Point, end_point:Point, reset_memory = False, **kargs) -> Union[List[Point], np.ndarray]:
        """
        Executes the selected path planning algorithm from start to goal.

        Args:
            start_point (Point): Starting position.
            end_point (Point): Target position.
            reset_memory (bool): Whether to reset memory before planning.

        Returns:
            Optional[Union[List[Point], np.ndarray]]: Path as a list of points, or None if no path found.
        """

        raise NotImplementedError
    
    @classmethod
    def get_reference_path(self, path:List[Point] | np.ndarray, pace:float=1.0, dt=0.1) -> np.ndarray:

        return interpolate_along_route(route=path,
                                       step=pace*dt)

class FieldPlanner(Planner):

    """
    Path planner based on field only
    """

    def __init__(self, field:Union[RiskField], alg: Optional[PathAlgArg] = None):
        super().__init__(None)
        
        self.algs = defaultdict(lambda: None)
        self.memory = {}
        
        self.field = field

        # Register Algorithms
        self.add_alg("i-rrt*", self._irrt_star_wrapper)
        self.add_alg("bit*", self._bit_star_wrapper)
        self.add_alg("m-apf", self._m_apf_wrapper)
        
        # Default selection
        if alg is not None:
            self.select_alg(alg=alg, reset_memory=False)
        else:
            self.select_alg("m-apf")

    def _irrt_star_wrapper(self, start_point: Point, end_point: Point, **kwargs) -> Optional[List[Point]]:
        return find_path_irrt_star(start_point, end_point, self.field, **kwargs)
    
    def _bit_star_wrapper(self, start_point: Point, end_point: Point, **kwargs) -> Optional[List[Point]]:
        return find_path_bit_star(start_point, end_point, self.field, **kwargs)

    def _m_apf_wrapper(self, start_point: Point, end_point: Point, **kwargs) -> Optional[List[Point]]:
        return find_path_modified_apf(start_point, end_point, self.field, **kwargs)

    def select_alg(self, alg: PathAlgArg, reset_memory=True):
        if isinstance(alg, str):
            if alg.lower() not in self.algs:
                # Fallback or error
                pass
            else:
                self.alg = self.algs[alg.lower()]
        else:
            super().select_alg(alg, reset_memory)

    def find_path(self, start_point: Point, end_point: Point, reset_memory=False, **kwargs) -> Union[List[Point], np.ndarray]:
        if reset_memory:
            self.reset_memory()
            
        if self.alg is None:
            raise ValueError("No algorithm selected.")
            
        return self.alg(start_point=start_point, end_point=end_point, **kwargs)

class QuadPlanner(Planner):

    """
    Path planner using quad network
    """

    def __init__(self, quadtree:QuadTree, alg: Optional[PathAlgArg] = None):

        super().__init__(None)
        
        self.field = quadtree.field
        self.quadtree = quadtree
        self.network = QuadNetwork(quadtree=quadtree)

        self.algs["a*"] = find_path_A_star
        self.algs["dijkstra"] = find_path_dijkstra
        self.algs["a*-e"] = find_path_astar_e
        self.algs["dijkstra-e"] = find_path_dijkstra_e
        self.alg = self.algs["a*"]

        if alg is not None:
            self.select_alg(alg=alg, reset_memory=False)

    def select_alg(self, alg, reset_memory=True):
        """
        Selects a planning algorithm by name or function reference.

        Args:
            alg (Optional[PathAlgArg]): Algorithm name or custom function.
            reset_memory (bool): Whether to reset internal memory state.

        Options:
        - a*
        - dijkstra
        - a*-e
        - dijkstra-e
        """

        return super().select_alg(alg, reset_memory)

    def refresh(self):
        """
        Resets planner memory and refreshes the network graph.
        """
        super().refresh()
        self.refresh_network()

    def refresh_network(self):
        """
        Refreshes the quad connectivity in the network graph.
        """
        self.network.refresh()

    def reset_memory(self):
        """
        Clears any memory state stored across planning runs.
        """
        self.memory.clear()

    def find_path(self, start_point:Point, end_point:Point, refresh_network = True, reset_memory = False, smooth_path = True, **kargs) -> Optional[List[Point] | np.ndarray]:
        """
        Executes the selected path planning algorithm from start to goal.

        Args:
            start_point (Point): Starting position.
            end_point (Point): Target position.
            refresh_network (bool): Whether to refresh quad connectivity.
            reset_memory (bool): Whether to reset memory before planning.

        Returns:
            Optional[Union[List[Point], np.ndarray]]: Path as a list of points, or None if no path found.
        """
        if refresh_network:
            self.refresh_network()

        if reset_memory:
            self.reset_memory()

        start_point, end_point = np.asarray(start_point, dtype=float), np.asarray(end_point, dtype=float)

        path = self.alg(start_point=start_point, end_point=end_point, network=self.network, memory=self.memory, **kargs)

        if smooth_path and path is not None:
            path = network_path_smoothing(path=path, network=self.network)
            path = network_path_smoothing(path=path[::-1], network=self.network)[::-1]

        return  path

# Path planning algorithms
def smooth_path_line_of_sight(path: np.ndarray, field: RiskField, threshold: float):
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    curr_idx = 0
    
    while curr_idx < len(path) - 1:
        found_next = False
        # Try to find the furthest shortcut
        for next_idx in range(len(path) - 1, curr_idx, -1):
            dist = np.linalg.norm(path[next_idx] - smoothed[-1])
            steps = max(3, int(dist / 2.0))
            check_pts = np.linspace(smoothed[-1], path[next_idx], steps)
            
            if not np.any(field.eval(check_pts) > threshold):
                smoothed.append(path[next_idx])
                curr_idx = next_idx
                found_next = True
                break
        
        # FALLBACK: If no shortcut is found, move to the very next point
        if not found_next:
            curr_idx += 1
            smoothed.append(path[curr_idx])
            
    return np.array(smoothed)

def find_path_irrt_star(
    start_point: Point,
    end_point: Point,
    field: RiskField,
    step_size: float | None = None,     # Increased for better 'wrap-around'
    search_radius: float|None = None,
    max_iter: int = 300,        # Higher iterations for complex buildings
    collision_threshold: float = 0.7,
    collision_step = 2.0,
    goal_bias: float = 0.2,
    **kwargs
) -> Optional[np.ndarray]:
    
    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)
    step_size = max(field.size)/120 if step_size is None else step_size
    search_radius = max(field.size)/30 if search_radius is None else search_radius

    c_best = np.inf
    best_path = None
    
    # Informed RRT* variables
    c_min = np.linalg.norm(end_point - start_point)
    x_center = (start_point + end_point) / 2.0
    # Rotation matrix to align ellipsoid with start-goal axis
    dir_vec = (end_point - start_point) / c_min
    angle = np.arctan2(dir_vec[1], dir_vec[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    nodes = np.zeros((max_iter + 1, 2))
    parents = np.full(max_iter + 1, -1, dtype=int)
    costs = np.zeros(max_iter + 1)
    
    nodes[0] = start_point
    node_count = 1

    for i in range(max_iter):
        if c_best < np.inf:
            r = np.sqrt(np.random.random())
            theta = np.random.uniform(0, 2 * np.pi)
            x_ball = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            a1 = c_best / 2.0
            a2 = np.sqrt(c_best**2 - c_min**2) / 2.0
            x_samp = rot @ np.array([x_ball[0] * a1, x_ball[1] * a2]) + x_center
        else:
            if np.random.random() < goal_bias:
                x_samp = end_point
            else:
                x_samp = np.random.uniform(field.bbox[0], field.bbox[1])

        # Nearest & Steer
        dists = np.linalg.norm(nodes[:node_count] - x_samp, axis=1)
        near_idx = np.argmin(dists)
        
        diff = x_samp - nodes[near_idx]
        mag = np.linalg.norm(diff)
        x_new = nodes[near_idx] + (diff / mag * step_size) if mag > step_size else x_samp

        # Collision Check (Dynamic)
        check_steps = max(3, int(np.linalg.norm(x_new - nodes[near_idx]) / collision_step))
        if np.any(field.eval(np.linspace(nodes[near_idx], x_new, check_steps)) > collision_threshold):
            continue

        # 2. FIND NEARBY (Vectorized Distance)
        dists_all = np.linalg.norm(nodes[:node_count] - x_new, axis=1)
        nearby_mask = dists_all <= search_radius
        nearby_indices = np.where(nearby_mask)[0]
        
        if len(nearby_indices) == 0:
            nearby_indices = np.array([near_idx])

        # 3. VECTORIZED PARENT SELECTION
        candidate_costs = costs[nearby_indices] + dists_all[nearby_indices]
        
        # Sort candidates by cost so we check the cheapest ones first for collisions
        sorted_nearby_idx = nearby_indices[np.argsort(candidate_costs)]
        
        best_near = near_idx
        min_cost = costs[near_idx] + dists_all[near_idx]

        for nb in sorted_nearby_idx:
            # Short-circuiting collision check using your QuadTree awareness
            if is_segment_valid_vectorized(nodes[nb], x_new, field, collision_threshold):
                best_near = nb
                min_cost = costs[nb] + dists_all[nb]
                break 

        # 4. ADD NODE
        nodes[node_count] = x_new
        parents[node_count] = best_near
        costs[node_count] = min_cost

        # 5. VECTORIZED REWIRING
        new_risk_costs = costs[node_count] + dists_all[nearby_indices]
        rewire_mask = new_risk_costs < costs[nearby_indices]
        rewire_indices = nearby_indices[rewire_mask]

        if len(rewire_indices) > 0:
            valid_rewire_mask = batch_collision_check(x_new, nodes[rewire_indices], field, collision_threshold)
            
            final_rewire_targets = rewire_indices[valid_rewire_mask]
            parents[final_rewire_targets] = node_count
            costs[final_rewire_targets] = new_risk_costs[rewire_mask][valid_rewire_mask]

        node_count += 1
        
        # Check if we hit the goal to update c_best
        dist_to_goal = np.linalg.norm(x_new - end_point)
        if dist_to_goal < step_size:
            total_c = costs[node_count-1] + dist_to_goal
            if total_c < c_best:
                c_best = total_c
                # Reconstruct path for c_best tracking
                curr_path = [end_point]
                curr = node_count - 1
                while curr != -1:
                    curr_path.append(nodes[curr])
                    curr = parents[curr]
                best_path = np.array(curr_path[::-1])

    if best_path is not None:
        return smooth_path_line_of_sight(best_path, field, collision_threshold)
    return None

def batch_collision_check(origin: np.ndarray, targets: np.ndarray, field: QRiskField, threshold: float, num_samples: int = 5):
    """
    Checks multiple segments (origin -> targets[i]) for collisions in one go.
    """
    K = targets.shape[0]

    alphas = np.linspace(0, 1, num_samples)
    
    segments = origin[None, None, :] + alphas[None, :, None] * (targets - origin)[:, None, :]
    
    flat_segments = segments.reshape(-1, 2)
    
    risks = field.eval(flat_segments)
    
    risks = risks.reshape(K, num_samples)
    
    # A segment is valid if ALL its samples are below the threshold
    return ~np.any(risks > threshold, axis=1)

def is_segment_valid_vectorized(
    p1: np.ndarray, 
    p2: np.ndarray, 
    field: QRiskField, 
    threshold: float, 
    min_samples: int = 5
) -> bool:
    """
    Checks if a single segment is collision-free using vectorized sampling.
    """
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-7:
        return True
        
    step_limit = field.quadtree.min_sector_size / 2.0
    num_samples = max(min_samples, int(dist / step_limit))
    
    alphas = np.linspace(0, 1, num_samples)[:, None]
    check_pts = p1 + alphas * (p2 - p1)
    
    risks = field.eval(check_pts)
    
    return not np.any(risks > threshold)

def _fast_segment_check(p1: np.ndarray, p2: np.ndarray, field: RiskField, threshold: float) -> Tuple[bool, float]:
    """
    Optimized collision check with Quadtree short-circuiting.
    Returns: (is_safe, avg_risk_cost_multiplier)
    """
    # 1. Quick Endpoints Check
    if np.any(field.eval(np.vstack([p1, p2])) > threshold):
        return False, np.inf

    # 2. Quadtree Awareness (QRiskField integration)
    if hasattr(field, 'quadtree'):
        midpoint = (p1 + p2) / 2.0
        q_node = field.quadtree.find_quad([midpoint], max_depth=3)[0]
        if len(q_node.rgj_idx) == 0:
            return True, 0.0

    # 3. Dense Check
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-6: return True, 0.0
    
    # Adaptive sampling
    steps = max(3, int(dist / 2.0)) 
    check_pts = np.linspace(p1, p2, steps)
    risks = field.eval(check_pts)
    
    if np.any(risks > threshold):
        return False, np.inf
        
    return True, np.mean(risks)

def find_path_bit_star(
    start_point: Point,
    end_point: Point,
    field: RiskField,
    batch_size: int = 150,
    max_batches: int = 20,
    collision_threshold: float = 0.7,
    risk_penalty: float = 5.0,
    **kwargs
) -> Optional[np.ndarray]:
    """
    Faithful BIT* implementation (Gammell et al. 2015).
    Maintains explicit Vertex and Edge queues to ensure optimal search ordering.
    """
    
    start_pt = np.atleast_1d(start_point).astype(float)
    end_pt = np.atleast_1d(end_point).astype(float)
    
    # Heuristic (Euclidean)
    def h(p): return np.linalg.norm(p - end_pt)
    
    points = [start_pt, end_pt]
    
    # Tree properties per index
    g_scores = {0: 0.0, 1: np.inf}
    parents = {0: None, 1: None}
    
    # Sets for logical tracking
    tree_indices = {0}
    sample_indices = {1} 
    
    c_best = np.inf
    c_min = np.linalg.norm(end_pt - start_pt)
    
    # Informed Sampling Setup
    x_center = (start_pt + end_pt) / 2.0
    dir_vec = (end_pt - start_pt) / (c_min + 1e-9)
    angle = np.arctan2(dir_vec[1], dir_vec[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # Queues
    qv = []
    qe = []

    for batch in range(max_batches):
        if c_best < np.inf:
            # Prune unconnected samples that can't improve solution
            to_remove = []
            for idx in sample_indices:
                if idx == 1: continue # Don't delete goal
                pt = points[idx]
                if np.linalg.norm(pt - start_pt) + h(pt) >= c_best:
                    to_remove.append(idx)
            
            for idx in to_remove:
                sample_indices.remove(idx)

        # Generate new samples
        new_indices = []
        count_needed = batch_size
        attempts = 0
        
        while len(new_indices) < count_needed and attempts < count_needed * 10:
            attempts += 1
            if c_best < np.inf:
                # Ellipsoid
                r_rand = np.sqrt(np.random.random())
                theta = np.random.uniform(0, 2 * np.pi)
                x_ball = np.array([r_rand * np.cos(theta), r_rand * np.sin(theta)])
                a1 = c_best / 2.0
                a2 = np.sqrt(max(0, c_best**2 - c_min**2)) / 2.0
                x_samp = rot @ (x_ball * [a1, a2]) + x_center
            else:
                # Global
                x_samp = np.random.uniform(field.bbox[0], field.bbox[1])
            
            # Pruning check
            if np.linalg.norm(x_samp - start_pt) + h(x_samp) < c_best:
                points.append(x_samp)
                idx = len(points) - 1
                sample_indices.add(idx)
                new_indices.append(idx)
                g_scores[idx] = np.inf
                parents[idx] = None

        q = len(points)
        dim = 2
        
        # Calculate R_BIT* (Radius)
        area = (field.bbox[1][0] - field.bbox[0][0]) * (field.bbox[1][1] - field.bbox[0][1])
        zeta_d = np.pi # Unit ball volume in 2D
        gamma = 2 * (1 + 1/dim)**(1/dim) * (area / zeta_d)**(1/dim)
        r_disc = gamma * (np.log(q) / q)**(1/dim)
        r_disc = max(r_disc, 0.5) # Minimum sanity radius

        active_samples_list = list(sample_indices)
        if not active_samples_list: continue
        
        sample_kdtree = cKDTree([points[i] for i in active_samples_list])

        qv = []
        qe = []
        
        for idx in tree_indices:
            # Only expand if it can theoretically improve solution
            if g_scores[idx] + h(points[idx]) < c_best:
                heapq.heappush(qv, (g_scores[idx] + h(points[idx]), idx))

        while qv or qe:
            
            # Stop if the best risk in queue is worse than current solution
            if not qv and not qe: break
            
            min_qv = qv[0][0] if qv else np.inf
            min_qe = qe[0][0] if qe else np.inf
            
            if min_qv >= c_best and min_qe >= c_best:
                break

            # Expand Best Vertex (qv) 
            if min_qv <= min_qe:
                _, v_idx = heapq.heappop(qv)
                
                # Pruning check (just in case g_score changed or bounds updated)
                if g_scores[v_idx] + h(points[v_idx]) >= c_best:
                    continue

                # Spherical Search: Find unconnected samples near v
                v_pt = points[v_idx]
                
                # Query KDT for neighbors within radius
                kd_idxs = sample_kdtree.query_ball_point(v_pt, r_disc)
                
                for k_idx in kd_idxs:
                    x_idx = active_samples_list[k_idx]
                    
                    # Estimate cost
                    dist = np.linalg.norm(points[x_idx] - v_pt)
                    f_hat = g_scores[v_idx] + dist + h(points[x_idx])
                    
                    if f_hat < c_best:
                        # Consistency check: In strict BIT*, we also check if
                        # g_score[v] + dist < g_score[x].
                        if g_scores[v_idx] + dist < g_scores.get(x_idx, np.inf) - 1e-6:
                            heapq.heappush(qe, (f_hat, v_idx, x_idx))

            # Expand Best Edge (qe) 
            else:
                _, v_idx, x_idx = heapq.heappop(qe)
                
                # 1. Check if this edge is still useful
                dist_g = np.linalg.norm(points[x_idx] - points[v_idx])
                
                is_safe, avg_risk = _fast_segment_check(points[v_idx], points[x_idx], field, collision_threshold)
                
                if not is_safe:
                    continue
                    
                edge_cost = dist_g * (1.0 + risk_penalty * avg_risk)
                tentative_g = g_scores[v_idx] + edge_cost
                
                if tentative_g + h(points[x_idx]) >= c_best:
                    continue

                # 2. Update if path improves
                if tentative_g < g_scores.get(x_idx, np.inf):
                    g_scores[x_idx] = tentative_g
                    parents[x_idx] = v_idx
                    
                    if x_idx in sample_indices:
                        sample_indices.remove(x_idx)
                        tree_indices.add(x_idx)
                        
                        heapq.heappush(qv, (tentative_g + h(points[x_idx]), x_idx))
                    
                    if x_idx == 1:
                        c_best = tentative_g

    #  Reconstruct Path 
    if g_scores[1] < np.inf:
        path_idxs = []
        curr = 1
        while curr is not None:
            path_idxs.append(curr)
            curr = parents[curr]
        
        if len(path_idxs) < 2: return None
        
        path_pts = np.array([points[i] for i in reversed(path_idxs)])
        return smooth_path_line_of_sight(path_pts, field, collision_threshold)

    return None

# Algorithm 2: Modified Artificial Potential Field (M-APF)

def find_path_modified_apf(
    start_point: Point,
    end_point: Point,
    field: RiskField,
    step_size: float | None = None,
    max_iter: int = 5000,
    goal_threshold: float | None = None,
    eta: float = 200.0,
    xi: float = 5.0,
    m: float = 2.0, 
    rho_0: float = 1.5, 
    **kwargs
) -> Optional[np.ndarray]:
    """
    Modified Artificial Potential Field (M-APF) using Anisotropic Scaling.
    
    Uses the gradient of the modified risk function to mitigate local minima
    when obstacles are near the goal.
    
    Ref: Adapte from - Rostami et al. / Bounini et al.
    """
    path = [np.array(start_point, dtype=float)]
    current_pos = np.atleast_2d(start_point).astype(float)
    end_point = np.array(end_point, dtype=float)
    step_size = max(field.size)/1000 if step_size is None else step_size
    goal_threshold = max(field.size)/200 if goal_threshold is None else step_size
    
    velocity = np.zeros(2) 
    momentum = 0.1 

    for _ in range(max_iter):
        pos_vec = current_pos[0]
        dist_to_goal = np.linalg.norm(pos_vec - end_point)
        
        if dist_to_goal < goal_threshold:
            path.append(end_point)
            return np.array(path)
        
        # 1. Attractive Force (Standard Euclidean)
        f_att = xi * (end_point - pos_vec)
        
        # 2. Adaptive Repulsive Force
        f_rep = np.zeros(2)
        
        rho_sqr, rgj_idx = field.squared_dist(current_pos, return_reference=True)
        rho = np.sqrt(rho_sqr[0]) + 1e-10 # Scaled distance (rho)
        
        if rho <= rho_0:
            rep_vecs = field.repulsion_vectors(current_pos, filted_idx=[rgj_idx[0]], min_dist_select=True)
            
            if len(rep_vecs) > 0:
                vec_obs_to_robot = rep_vecs[0]
                dist_obs = np.linalg.norm(vec_obs_to_robot) + 1e-10
                
                n_obs = vec_obs_to_robot / dist_obs 
                
                vec_robot_to_goal = end_point - pos_vec
                n_goal = vec_robot_to_goal / (dist_to_goal + 1e-10)
  
                t1 = eta * (1.0/rho - 1.0/rho_0) * (1.0/rho**2) * (dist_to_goal**m)
                f_r1 = t1 * n_obs
                
                t2 = (m/2.0) * eta * ((1.0/rho - 1.0/rho_0)**2) * (dist_to_goal**(m-1))
                f_r2 = t2 * n_goal
                
                f_rep = f_r1 + f_r2
        
        # 3. Motion Integration
        f_total = f_att + f_rep
        f_norm = np.linalg.norm(f_total)
        
        if f_norm > 1e-6:
            direction = f_total / f_norm
            velocity = momentum * velocity + (1 - momentum) * direction
            pos_vec = pos_vec + velocity * step_size
        else:
            pos_vec = pos_vec + np.random.uniform(-0.1, 0.1, 2)

        current_pos[0] = pos_vec
        path.append(pos_vec.copy())

    return None

def find_path_A_star(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf, **kargs) -> Optional[List[Point]]:
    """
    A* pathfinding algorithm over the quad network using quad centers as waypoints.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Goal position.
        network (QuadNetwork): Graph of connected QuadNodes.
        scaler (Optional[Scaler]): Optional scaling function for traversal cost.
        max_scale (float): Upper limit for scaled traversal cost.

    Returns:
        Optional[List[Point]]: Path from start to goal, or None if no path exists.
    """
    start_quad, end_quad = tuple(network.find_quad([start_point, end_point]))

    if scaler is None:
        scaler = lambda p: 1.0 + 2*p

    open_set = []
    heapq.heappush(open_set, (0, start_quad))

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    f_score = defaultdict(lambda: np.inf)
    f_score[start_quad] = network.get_center_distance(start_quad, end_quad)

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_quad:
            break

        for neighbor in network[current]:
            tentative_g_score = g_score[current] + network.get_center_distance(current, neighbor, scaler=scaler, max_scale=max_scale)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + network.get_center_distance(neighbor, end_quad)

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    if current == end_quad:
        quad_path = __reconstruct_quad_path__(came_from, current)
        path = np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

        path = [start_point] + [network.get_shared_entry_point(quad_path[i], quad_path[i+1], path[i+1]) for i in range(len(quad_path)-1)] + [end_point]

        return np.array(path)

    return None

def find_path_dijkstra(start_point:Point, end_point:Point, network:QuadNetwork, scaler:Optional[Scaler]=None, max_scale: float = np.inf, **kargs)-> Optional[List[Point]]:
    """
    Dijkstra's algorithm over the quad network, using quad centers as waypoints.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Goal position.
        network (QuadNetwork): Graph of connected QuadNodes.
        scaler (Optional[Scaler]): Optional scaling function for traversal cost.
        max_scale (float): Upper limit for scaled traversal cost.

    Returns:
        Optional[List[Point]]: Path from start to goal, or None if no path exists.
    """
    start_quad, end_quad = network.find_quad([start_point, end_point])

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    heapq.heappush(open_set, (0, start_quad))

    came_from = {}
    dist = defaultdict(lambda: np.inf)
    dist[start_quad] = 0
    current = None

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_quad:
            break

        for neighbor in network[current]:
            tentative_dist = dist[current] + network.get_center_distance(current, neighbor, scaler=scaler, max_scale=max_scale)

            if tentative_dist < dist[neighbor]:
                came_from[neighbor] = current
                dist[neighbor] = tentative_dist

                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (dist[neighbor], neighbor))

    if current == end_quad:
        quad_path = __reconstruct_quad_path__(came_from, current)
        path = np.array([start_point] + network.get_quad_center(quad_path)[1:-1] + [end_point])

        path = [start_point] + [network.get_shared_entry_point(quad_path[i], quad_path[i+1], path[i+1]) for i in range(len(quad_path)-1)] + [end_point]
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

def find_path_astar_e(
        start_point: Point,
        end_point: Point,
        network: QuadNetwork,
        scaler: Optional[Scaler] = None,
        max_scale: float = np.inf,
        **kargs
    ) -> Optional[List[Point]]:

    """
    A* pathfinding with edge-aware traversal between quad regions.

    This variant of A* seeks to avoid quad center traversal and instead moves between
    shared edges based on entry direction and local quad structure.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Target position.
        network (QuadNetwork): The graph of quad connectivity.
        scaler (Optional[Scaler]): Optional scaling function for local quad cost.
        max_scale (float): Maximum value allowed by the scaler.

    Returns:
        Optional[List[Point]]: Point-to-point path, or None if no path is found.
    """

    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)

    start_quad, end_quad = network.find_quad([start_point, end_point])

    if start_quad is None or end_quad is None:
        return None

    if np.allclose(start_point, end_point):
        return [start_point]
    
    if start_quad == end_quad:
        return [start_point, end_point]

    if scaler is None:
        scaler = lambda p: 1.0 + p

    open_set = []
    open_set_hash = set()
    heapq.heappush(open_set, (0, start_quad))
    open_set_hash.add(start_quad)

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    f_score = defaultdict(lambda: np.inf)
    f_score[start_quad] = np.linalg.norm(start_point - end_point)

    edge_entry = {start_quad: start_point}

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == end_quad:
            break

        for neighbor in network[current]:
            entry = edge_entry[current]
            exit = network.get_shared_entry_point(current, neighbor, entry)

            if exit is None:
                continue

            tentative_g = g_score[current] + __get_scaled_distance__(entry, exit, neighbor, scaler, max_scale)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + np.linalg.norm(exit - end_point)
                edge_entry[neighbor] = exit

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    if current == end_quad:
        # Reconstruct full point-level and quad path
        path = [end_point]
        quad = end_quad
        while quad in came_from:
            path.append(edge_entry[quad])
            quad = came_from[quad]

        path.append(start_point)
        path = path[::-1]

        quad_path = __reconstruct_quad_path__(came_from, end_quad)
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None

def find_path_dijkstra_e(
        start_point: Point,
        end_point: Point,
        network: QuadNetwork,
        scaler: Optional[Scaler] = None,
        max_scale: float = np.inf,
        **kargs
    ) -> Optional[List[Point]]:

    """
    Dijkstra's algorithm with edge-aware traversal between quad regions.

    This variant of Dijkstra seeks to avoid quad center traversal and instead moves between
    shared edges based on entry direction and local quad structure.

    Args:
        start_point (Point): Starting position.
        end_point (Point): Target position.
        network (QuadNetwork): The graph of quad connectivity.
        scaler (Optional[Scaler]): Optional scaling function for local quad cost.
        max_scale (float): Maximum value allowed by the scaler.

    Returns:
        Optional[List[Point]]: Point-to-point path, or None if no path is found.
    """

    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)

    start_quad, end_quad = network.find_quad([start_point, end_point])

    if start_quad is None or end_quad is None:
        return None

    if np.allclose(start_point, end_point):
        return [start_point]
    
    if start_quad == end_quad:
        return [start_point, end_point]

    if scaler is None:
        scaler = lambda p: 1.0 + 4*(p**2)

    open_set = []
    open_set_hash = set()
    heapq.heappush(open_set, (0, start_quad))
    open_set_hash.add(start_quad)

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    g_score[start_quad] = 0

    edge_entry = {start_quad: start_point}

    current = None

    while open_set:
        _, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        if current == end_quad:
            break

        for neighbor in network[current]:
            entry = edge_entry[current]
            exit = network.get_shared_entry_point(current, neighbor, entry)

            if exit is None:
                continue

            tentative_g = g_score[current] + __get_scaled_distance__(entry, exit, neighbor, scaler, max_scale)

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                edge_entry[neighbor] = exit

                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (g_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    if current == end_quad:
        # Reconstruct full point-level path
        path = [end_point]
        quad = end_quad
        while quad in came_from:
            path.append(edge_entry[quad])
            quad = came_from[quad]
        path.append(start_point)
        path = path[::-1]

        quad_path = __reconstruct_quad_path__(came_from, end_quad)
        path = optimize_path_via_edge_bundling(path, quad_path, network)

        return np.array(path)

    return None
