from collections import defaultdict
import heapq
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree
from larp.field import PotentialField
from larp.fn import interpolate_along_route
from larp.pp.network import QuadNetwork

from larp.quad import QuadNode, QuadTree
from larp.types import Scaler, Point

"""
Author: Josue N Rivera

Module providing path planning algorithms. They can be over a potentiald field or quadtree-based spatial network of the potential field.
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
    
    # Allow for path to be len(quad_path) + 1 (Start -> [Q1...Qn] -> End)
    # If sizes match exactly, we assume the last quad corresponds to the last point.
    if len(path) != len(quad_path) + 1:
        # Fallback for when path and quad_path might be misaligned in size
        if len(path) < 2: 
            return path
        # Truncate to safe length
        limit = min(len(path)-1, len(quad_path))
        quad_path = quad_path[:limit]

    optimized_points = [path[0]]
    current_idx = 0

    while current_idx < len(quad_path):
        # 2. Look ahead to find the longest valid segment
        # We look for a sequence where a straight line might be feasible.
        # In this improved version, we simply take the longest chain of neighbors.
        
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
        # Segment goes from path[current_idx] to path[next_idx]
        start_point = optimized_points[-1]
        
        # Handle end of path boundary
        end_idx_in_path = next_idx if next_idx < len(path) else len(path) - 1
        end_point = path[end_idx_in_path]

        # Interpolate through the "Portals"
        # Instead of generic interpolation, we project the ideal line onto the shared edges.
        
        # Vector from start to goal of this segment
        ideal_trajectory_vec = end_point - start_point
        total_dist = np.linalg.norm(ideal_trajectory_vec)
        
        if total_dist > 1e-6:
            # Iterate through the quads in this specific segment
            segment_quads = quad_path[current_idx:next_idx]
            
            accumulated_dist = 0.0
            
            for k in range(len(segment_quads) - 1):
                q_curr = segment_quads[k]
                q_next = segment_quads[k+1]
                
                # Approximate distance progress to find where the straight line *should* be
                # We use the center distances to estimate 't' along the ideal line
                dist_step = np.linalg.norm(q_next.center_point - q_curr.center_point)
                accumulated_dist += dist_step
                
                # t is the ratio of distance traveled along the quad centers
                # (This is an approximation, but sufficient for edge clamping)
                t = accumulated_dist / (total_dist + 1e-6) # Avoid div/0
                t = np.clip(t, 0.01, 0.99)
                
                # Point on the ideal straight line
                ideal_point = start_point + t * ideal_trajectory_vec
                
                # Project this ideal point onto the ACTUAL shared boundary
                # This "clamps" the straight line to the valid corridor
                clamped_point = network.get_shared_entry_point(q_curr, q_next, ideal_point)
                
                if clamped_point is not None:
                    # Deduplication: Only add if sufficiently far from last point
                    if np.linalg.norm(clamped_point - optimized_points[-1]) > 1e-3:
                         optimized_points.append(clamped_point)

        # 4. Add the segment endpoint
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
            alg (Optional[PathAlgArg]): Algorithm name or custom function.
            reset_memory (bool): Whether to reset internal memory state.
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

    def __init__(self, field:Union[PotentialField], alg: Optional[PathAlgArg] = None):
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
def smooth_path_line_of_sight(path: np.ndarray, field: PotentialField, threshold: float):
    """
    State-of-the-art smoothing: Removes redundant waypoints if a 
    direct line-of-sight exists without hitting obstacles.
    """
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    curr_idx = 0
    
    while curr_idx < len(path) - 1:
        # Check from last added point to the furthest possible point in the path
        for next_idx in range(len(path) - 1, curr_idx, -1):
            dist = np.linalg.norm(path[next_idx] - smoothed[-1])
            steps = max(3, int(dist / 2.0))
            check_pts = np.linspace(smoothed[-1], path[next_idx], steps)
            
            if not np.any(field.eval(check_pts) > threshold):
                smoothed.append(path[next_idx])
                curr_idx = next_idx
                break
    return np.array(smoothed)

def find_path_irrt_star(
    start_point: Point,
    end_point: Point,
    field: PotentialField,
    step_size: float = 15.0,     # Increased for better 'wrap-around'
    search_radius: float = 40.0,
    max_iter: int = 3000,        # Higher iterations for complex buildings
    collision_threshold: float = 0.7,
    goal_bias: float = 0.1,
    **kwargs
) -> Optional[np.ndarray]:
    
    start_point = np.array(start_point, dtype=float)
    end_point = np.array(end_point, dtype=float)
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
        # --- Informed Sampling ---
        if c_best < np.inf:
            # Sample in a unit circle, then scale/rotate into the ellipsoid
            r = np.sqrt(np.random.random())
            theta = np.random.uniform(0, 2 * np.pi)
            x_ball = np.array([r * np.cos(theta), r * np.sin(theta)])
            
            # Ellipsoid radii: major axis is c_best/2
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
        check_steps = max(3, int(np.linalg.norm(x_new - nodes[near_idx]) / 2.0))
        if np.any(field.eval(np.linspace(nodes[near_idx], x_new, check_steps)) > collision_threshold):
            continue

        # Choose Parent & Rewire (Vectorized)
        dists_all = np.linalg.norm(nodes[:node_count] - x_new, axis=1)
        nearby = np.where(dists_all <= search_radius)[0]
        
        best_near = near_idx
        min_cost = costs[near_idx] + np.linalg.norm(x_new - nodes[near_idx])

        for nb in nearby:
            c_cand = costs[nb] + dists_all[nb]
            if c_cand < min_cost:
                if not np.any(field.eval(np.linspace(nodes[nb], x_new, 5)) > collision_threshold):
                    min_cost = c_cand
                    best_near = nb

        # Add Node
        nodes[node_count] = x_new
        parents[node_count] = best_near
        costs[node_count] = min_cost
        
        # Rewire Neighbors
        for nb in nearby:
            if costs[node_count] + dists_all[nb] < costs[nb]:
                if not np.any(field.eval(np.linspace(x_new, nodes[nb], 5)) > collision_threshold):
                    parents[nb] = node_count
                    costs[nb] = costs[node_count] + dists_all[nb]

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

def find_path_bit_star(
    start_point: Point,
    end_point: Point,
    field: PotentialField,
    batch_size: int = 150,
    max_batches: int = 20,
    collision_threshold: float = 0.7,
    risk_penalty: float = 5.0,
    **kwargs
) -> Optional[np.ndarray]:
    
    start_pt = np.atleast_1d(start_point).astype(float)
    end_pt = np.atleast_1d(end_point).astype(float)
    start_tuple = tuple(start_pt)
    end_tuple = tuple(end_pt)
    
    # 1. Setup Informed Sampling
    c_best = np.inf
    c_min = np.linalg.norm(end_pt - start_pt)
    x_center = (start_pt + end_pt) / 2.0
    
    dir_vec = (end_pt - start_pt) / c_min
    angle = np.arctan2(dir_vec[1], dir_vec[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    h = lambda p: np.linalg.norm(np.array(p) - end_pt)

    # 2. State Tracking
    samples = {end_tuple} 
    tree = {start_tuple: None}
    g_scores = {start_tuple: 0.0}

    def get_edge_info(u, v):
        u_arr, v_arr = np.array(u), np.array(v)
        dist = np.linalg.norm(v_arr - u_arr)
        # Collision check density: sample every 5 units for performance
        steps = max(3, int(dist / 5.0)) 
        pts = np.linspace(u_arr, v_arr, steps)
        potentials = field.eval(pts)
        
        if np.any(potentials > collision_threshold):
            return np.inf, np.inf
        
        avg_risk = np.mean(potentials)
        weighted_cost = dist * (1.0 + risk_penalty * avg_risk)
        return dist, weighted_cost

    for b in range(max_batches):
        # --- Batch Generation ---
        new_samples = []
        while len(new_samples) < batch_size:
            if c_best < np.inf:
                # Ellipsoidal sampling
                r_val = np.sqrt(np.random.random())
                theta = np.random.uniform(0, 2 * np.pi)
                x_ball = np.array([r_val * np.cos(theta), r_val * np.sin(theta)])
                radii = np.array([c_best, np.sqrt(max(0, c_best**2 - c_min**2))]) / 2.0
                x_samp = rot @ (x_ball * radii) + x_center
            else:
                x_samp = np.random.uniform(field.bbox[0], field.bbox[1])
            
            if np.linalg.norm(x_samp - start_pt) + h(x_samp) < c_best:
                new_samples.append(tuple(x_samp))
        
        samples.update(new_samples)
        
        # --- Efficiency Update: Build KD-Tree for samples ---
        all_samples_list = list(samples)
        sample_tree = cKDTree(all_samples_list)
        
        # Shrinking Radius Formula (Standard RRT* / BIT*)
        # unit_vol = np.pi (for 2D)
        eta = 1.1 # optimality factor
        gamma = 2 * eta * np.sqrt(1.5 * (field.size[0]*field.size[1]) / np.pi)
        n_total = len(samples) + len(tree)
        r_disc = min(gamma * np.sqrt(np.log(n_total) / n_total), c_min)

        # --- Queue Management ---
        edge_queue = []
        # Add edges from tree nodes to nearby samples
        for u_pt in tree.keys():
            u_arr = np.array(u_pt)
            # Find indices of samples within radius
            idxs = sample_tree.query_ball_point(u_arr, r_disc)
            for idx in idxs:
                v_pt = all_samples_list[idx]
                if v_pt == u_pt or v_pt in tree: continue
                
                dist_uv = np.linalg.norm(np.array(v_pt) - u_arr)
                f_val = g_scores[u_pt] + dist_uv + h(v_pt)
                
                if f_val < c_best:
                    heapq.heappush(edge_queue, (f_val, u_pt, v_pt))

        # --- Ordered Search ---
        while edge_queue:
            f_val, u_pt, v_pt = heapq.heappop(edge_queue)
            
            if f_val >= c_best: 
                break # Cannot improve current best

            # Re-check g_score in case a better path to u was found since push
            if g_scores[u_pt] + np.linalg.norm(np.array(v_pt)-np.array(u_pt)) + h(v_pt) >= c_best:
                continue

            dist_uv, actual_cost = get_edge_info(u_pt, v_pt)
            
            if actual_cost == np.inf: continue

            tentative_g = g_scores[u_pt] + actual_cost
            if v_pt not in g_scores or tentative_g < g_scores[v_pt]:
                g_scores[v_pt] = tentative_g
                tree[v_pt] = u_pt
                
                if v_pt == end_tuple:
                    c_best = min(c_best, g_scores[v_pt])
                else:
                    # Expand to neighbors of the newly added node
                    v_arr = np.array(v_pt)
                    idxs = sample_tree.query_ball_point(v_arr, r_disc)
                    for idx in idxs:
                        next_v_pt = all_samples_list[idx]
                        if next_v_pt in tree: continue
                        dist_v_next = np.linalg.norm(np.array(next_v_pt) - v_arr)
                        f_next = g_scores[v_pt] + dist_v_next + h(next_v_pt)
                        if f_next < c_best:
                            heapq.heappush(edge_queue, (f_next, v_pt, next_v_pt))

    # Path Reconstruction
    if end_tuple in tree:
        path = []
        curr = end_tuple
        while curr is not None:
            path.append(curr)
            curr = tree[curr]
        return smooth_path_line_of_sight(np.array(path[::-1]), field, collision_threshold)
    
    return None

# -------------------------------------------------------------------------
# Algorithm 2: Modified Artificial Potential Field (M-APF)
# -------------------------------------------------------------------------

def find_path_modified_apf(
    start_point: Point,
    end_point: Point,
    field: PotentialField,
    step_size: float = 0.5,
    max_iter: int = 5000,
    goal_threshold: float = 2.0,
    eta: float = 200.0, # Increased Repulsion gain
    xi: float = 5.0,    # Attraction gain
    m: float = 2.0,     # M-APF hyper-parameter
    rho_0: float = 1.5,  # Scaling threshold (Influence in 'risk units')
    **kwargs
) -> Optional[np.ndarray]:
    """
    Modified Artificial Potential Field (M-APF) using Anisotropic Scaling.
    
    Uses the gradient of the modified potential function to mitigate local minima
    when obstacles are near the goal.
    
    Ref: Adapte from - Rostami et al. / Bounini et al.
    """
    path = [np.array(start_point, dtype=float)]
    current_pos = np.atleast_2d(start_point).astype(float)
    end_point = np.array(end_point, dtype=float)
    
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
        
        # Get scaled distance (rho^2) and the index of the closest obstacle
        # This automatically uses the repulsion matrix of the nearest RGJ
        rho_sqr, rgj_idx = field.squared_dist(current_pos, return_reference=True)
        rho = np.sqrt(rho_sqr[0]) + 1e-10 # Scaled distance (rho)
        
        if rho <= rho_0:
            # Get the physical repulsion vector for the specific closest obstacle
            # We filter by rgj_idx to be efficient
            rep_vecs = field.repulsion_vectors(current_pos, filted_idx=[rgj_idx[0]], min_dist_select=True)
            
            if len(rep_vecs) > 0:
                vec_obs_to_robot = rep_vecs[0]
                dist_obs = np.linalg.norm(vec_obs_to_robot) + 1e-10
                
                # n_obs: Unit direction away from obstacle
                n_obs = vec_obs_to_robot / dist_obs 
                
                # n_goal: Unit direction toward goal
                vec_robot_to_goal = end_point - pos_vec
                n_goal = vec_robot_to_goal / (dist_to_goal + 1e-10)
                
                # --- Anisotropic M-APF Calculation ---
                # Term 1: Scaled Repulsion
                # Uses (1/rho - 1/rho_0) instead of Euclidean distances
                t1 = eta * (1.0/rho - 1.0/rho_0) * (1.0/rho**2) * (dist_to_goal**m)
                f_r1 = t1 * n_obs
                
                # Term 2: Scaled Alignment (Tangential steering)
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
            # Brownian jitter to escape local minima
            pos_vec = pos_vec + np.random.uniform(-0.1, 0.1, 2)

        current_pos[0] = pos_vec
        path.append(pos_vec.copy())

    return np.array(path)

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

        #path = optimize_path_via_edge_bundling(path, quad_path, network)

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
