from typing import Optional, Tuple, List, Union
import numpy as np
from scipy.interpolate import CubicSpline

from larp.tp.solver import SQPSolver
from larp.types import Point, Trajectory

"""
Author: Josue N Rivera

Module providing motion planning algorithms. They start from a provided reference path.
"""

class Planner:
    """
    Trajectory reference planner using Cubic Spline interpolation.

    Generates a smooth state reference trajectory by projecting the robot's 
    position onto a cached spline geometry and sampling ahead in arc-length.
    """
    def __init__(self, solver: SQPSolver,
                 path: Union[List[Point], np.ndarray],
                 stable_state: np.ndarray,
                 ref_state_indices: Optional[List[int]] = None,
                 lookahead_distance: float = 0.5,
                 search_window: int = 50,
                 max_lat_accel: float = 2.0): # Max lateral accel (m/s^2) for speed limiting
        """
        :param solver: Solver instance with dt and N (horizon).
        :param path: Waypoints (N, 2) or (N, 3).
        :param stable_state: Equilibrium state to fill non-spatial dimensions.
        :param ref_state_indices: Indices for [x, y, theta, v_x, v_y]. 
                                  Pass None if dynamics don't have explicit velocity states.
        :param lookahead_distance: Extra distance ahead of projection to start reference.
        :param search_window: Number of path segments to search around last position (optimization).
        :param max_lat_accel: Maximum allowed lateral acceleration. 
                              Used to slow down reference on curves. 
                              v_max = sqrt(a_max / curvature)
        """
        self.solver = solver
        self.stable_state = np.array(stable_state, dtype=float)
        
        if ref_state_indices is None:
            self.idxs = [0, 1, 2] 
        else:
            self.idxs = ref_state_indices

        self.lookahead = lookahead_distance
        self.window = search_window
        self.max_lat_accel = max_lat_accel
        self.N = self.solver.N
        
        # State tracking
        self.prev_us = None
        self.last_s = 0.0 
        
        self.cs: Optional[CubicSpline] = None
        self.cum_len: Optional[np.ndarray] = None
        self.total_len = 0.0
        self.raw_path: Optional[np.ndarray] = None
        
        self.update_path(path)

    def update_path(self, path: Union[List[Point], np.ndarray]):
        points = np.atleast_2d(np.copy(path))
        if points.shape[0] < 2:
            raise ValueError("Path must contain at least 2 points")

        xy = points[:, :2]

        # Compute Arc Lengths
        deltas = xy[1:] - xy[:-1]
        dists = np.linalg.norm(deltas, axis=1)
        dists = np.maximum(dists, 1e-8)
        
        self.cum_len = np.concatenate(([0.0], np.cumsum(dists)))
        self.total_len = self.cum_len[-1]

        # Natural BC = Zero curvature at ends (linear extrapolation)
        self.cs = CubicSpline(self.cum_len, xy, bc_type='natural')
        self.raw_path = xy
        self.last_s = 0.0
        self.prev_us = None

    def get_projection(self, x_curr: float, y_curr: float, theta_curr: float = None) -> float:
        """
        Projects robot onto path.
        IMPROVEMENT: Penalizes segments that are 'behind' the robot or 
        moving in the opposite direction of theta_curr to prevent jitter.
        """
        if self.cum_len is None: return 0.0

        # 1. Define Search Window
        idx_est = np.searchsorted(self.cum_len, self.last_s, side='right') - 1
        # Search backwards a little, forwards a lot
        start = max(0, idx_est - 5) 
        end = min(len(self.raw_path) - 1, idx_est + self.window)

        if start >= end:
            return self.last_s

        # 2. Vectorized Segment Projection
        P_start = self.raw_path[start:end]
        P_end = self.raw_path[start+1:end+1]
        
        seg_vecs = P_end - P_start
        robot_vecs = np.array([x_curr, y_curr]) - P_start

        seg_lens_sq = np.sum(seg_vecs**2, axis=1)
        t_vals = np.sum(robot_vecs * seg_vecs, axis=1) / np.maximum(seg_lens_sq, 1e-8)
        t_vals = np.clip(t_vals, 0.0, 1.0)

        # Closest points
        closest_points = P_start + t_vals[:, None] * seg_vecs
        diffs = closest_points - np.array([x_curr, y_curr])
        dists_sq = np.sum(diffs**2, axis=1)

        # --- DIRECTIONAL HEURISTIC ---
        # If theta is provided, penalize segments that point opposite to robot
        if theta_curr is not None:
            # Normalized segment directions
            seg_dirs = seg_vecs / np.sqrt(np.maximum(seg_lens_sq, 1e-8))[:, None]
            robot_dir = np.array([np.cos(theta_curr), np.sin(theta_curr)])
            
            # Dot product: 1.0 = aligned, -1.0 = opposite
            alignment = np.dot(seg_dirs, robot_dir)
            
            # Add huge penalty for opposite facing segments
            # This prevents snapping to the wrong part of a loop
            dists_sq[alignment < -0.5] += 1000.0

        # 3. Find Best Segment
        min_idx = np.argmin(dists_sq)
        
        local_s = self.cum_len[start + min_idx] + t_vals[min_idx] * np.sqrt(seg_lens_sq[min_idx])
        
        # --- STRICT MONOTONICITY FIX ---
        # If the new projection is behind the previous one, hold position.
        if local_s < self.last_s:
            return self.last_s
            
        return local_s

    def get_ref(self, x0: np.ndarray, nominal_speed: float = 2.0, stop_at_end: bool = True) -> np.ndarray:
        """
        Generates reference trajectory with Dynamic Velocity Profiling.
        """
        ix, iy, ith = self.idxs[0], self.idxs[1], self.idxs[2]
        
        # 1. Update Progress (Pass theta for better projection)
        current_s = self.get_projection(x0[ix], x0[iy], theta_curr=x0[ith])
        self.last_s = current_s

        # 2. Generate Steps
        dt = self.solver.dt
        N = self.solver.N
        
        # We need to integrate s forward based on variable velocity, 
        # not just s = s0 + i*v*dt
        
        s_future = [current_s + self.lookahead]
        ref_states = []

        # 3. Iterative Reference Generation (Simulate ahead)
        for k in range(N + 1):
            s_curr = s_future[-1]
            
            # Clamp to path length
            s_clamped = np.clip(s_curr, 0, self.total_len)
            
            # --- CURVATURE CALCULATION ---
            # 1st derivative (Tangent)
            der1 = self.cs(s_clamped, 1) 
            # 2nd derivative (Acceleration direction)
            der2 = self.cs(s_clamped, 2)
            
            # Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^1.5
            cross = der1[0]*der2[1] - der1[1]*der2[0]
            norm_sq = der1[0]**2 + der1[1]**2
            curvature = np.abs(cross) / (norm_sq**1.5 + 1e-8)
            
            # --- VELOCITY PROFILING ---
            # Limit speed: v <= sqrt(a_max / k)
            # If curvature is 0, max_speed is infinite (clamped to nominal)
            if curvature > 1e-4:
                max_curve_speed = np.sqrt(self.max_lat_accel / curvature)
                target_v = min(nominal_speed, max_curve_speed)
            else:
                target_v = nominal_speed
                
            # Stop at end logic
            if stop_at_end and s_curr >= self.total_len:
                target_v = 0.0

            # --- CALCULATE STATE ---
            # Position
            pos = self.cs(s_clamped)
            
            # Tangent (Heading)
            # Normalized tangent vector
            tan_norm = np.linalg.norm(der1)
            if tan_norm < 1e-6:
                tan_vec = np.array([np.cos(x0[ith]), np.sin(x0[ith])]) # Fallback
            else:
                tan_vec = der1 / tan_norm

            # Yaw
            yaw = np.arctan2(tan_vec[1], tan_vec[0])
            
            # Unwrap Yaw relative to PREVIOUS reference point (or x0 for first)
            prev_yaw = ref_states[-1][ith] if len(ref_states) > 0 else x0[ith]
            yaw_diff = yaw - prev_yaw
            yaw_diff = (yaw_diff + np.pi) % (2*np.pi) - np.pi
            yaw = prev_yaw + yaw_diff

            # Assemble State
            state = self.stable_state.copy()
            state[ix] = pos[0]
            state[iy] = pos[1]
            state[ith] = yaw
            
            # If we have velocity indices, fill them
            if len(self.idxs) >= 5:
                ivx, ivy = self.idxs[3], self.idxs[4]
                state[ivx] = tan_vec[0] * target_v
                state[ivy] = tan_vec[1] * target_v
            
            ref_states.append(state)
            
            # Integrate s for next step
            s_future.append(s_curr + target_v * dt)

        return np.array(ref_states)
    
    def get_full_ref(self, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Generates the complete reference trajectory from the path start to end.
        Useful for visualization (ground truth) or offline analysis.

        :param nominal_speed: The target speed to traverse the path.
        :return: Array of shape (M, n_states) representing the full ideal trajectory.
        """
        if self.cum_len is None or self.total_len <= 0:
            return np.empty((0, len(self.stable_state)))
            
        dt = self.solver.dt
        
        # 1. Determine Sample Points
        # Total duration = Distance / Speed
        total_duration = self.total_len / max(nominal_speed, 1e-4)
        num_steps = int(np.ceil(total_duration / dt))
        
        # Generate arc-lengths from 0 to Total Length
        s_vals = np.linspace(0, self.total_len, num_steps)
        
        # 2. Evaluate Spline Geometry (Vectorized)
        pos_ref = self.cs(s_vals)        # Position (x, y)
        vel_tan = self.cs(s_vals, 1)     # Tangent (dx/ds, dy/ds)
        
        # 3. Compute Kinematics (Velocity & Heading)
        # Normalize tangent vectors to get unit direction
        norms = np.linalg.norm(vel_tan, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        
        # Scale unit direction by nominal speed
        vel_vec = (vel_tan / norms) * nominal_speed
        
        # Compute Yaw from velocity vector
        headings = np.arctan2(vel_vec[:, 1], vel_vec[:, 0])
        headings = np.unwrap(headings, discont=np.pi) # Continuous yaw (no jumps)
        
        # 4. Assemble Full State Matrix
        ix, iy, ith = self.idxs[0], self.idxs[1], self.idxs[2]
        
        # Initialize with stable state (handles non-spatial dims like arm joint angles)
        full_ref = np.tile(self.stable_state, (num_steps, 1))
        
        # Fill Spatial Dimensions
        full_ref[:, ix] = pos_ref[:, 0]
        full_ref[:, iy] = pos_ref[:, 1]
        full_ref[:, ith] = headings
        
        # Fill Velocity Dimensions (if indices provided)
        if len(self.idxs) >= 5:
            ivx, ivy = self.idxs[3], self.idxs[4]
            full_ref[:, ivx] = vel_vec[:, 0]
            full_ref[:, ivy] = vel_vec[:, 1]
            
        return full_ref

    def find_trajectory(self, x0: np.ndarray, 
                       max_iters: int = 10, 
                       nominal_speed: float = 2.0, 
                       reset: bool = False) -> Trajectory:
        
        if reset:
            self.last_s = 0.0
            self.prev_us = None
            max_iters = max(max_iters, 20) # Boost iters on cold start

        ref_traj = self.get_ref(x0, nominal_speed=nominal_speed)

        xs, us = self.solver.solve(x0, ref_traj, us_init=self.prev_us, max_iters=max_iters)

        self.prev_us = np.vstack([us[1:], us[-1:]]) 
        
        return xs, us
    
    def find_full_trajectory(self, x0: np.ndarray, 
                           max_iters: int = 5, 
                           nominal_speed: float = 2.0,
                           max_steps: int = 50000,
                           goal_tolerance: float = 0.01,
                           stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the closed-loop execution of the planner until the goal is reached.
        
        This method iteratively solves the QP optimization problem, executes a 
        segment of the result (defined by 'stride'), and replans from the new state.

        :param x0: Initial state vector.
        :param max_iters: Maximum SQP iterations per planning step.
        :param nominal_speed: Target progression speed along the path.
        :param max_steps: Safety limit for total simulation steps to prevent infinite loops.
        :param goal_tolerance: Euclidean distance to the final waypoint to consider finished.
        :param stride: Number of control steps to execute before replanning. 
                       - stride=1: Classic Receding Horizon (replan every step).
                       - stride=N: Execute entire horizon before replanning.
        :return: Tuple (all_xs, all_us) containing the complete state and control history.

        Example
        -------

        .. code-block:: python
            planner = Planner(solver, path_points, stable_state)
            xs, us = planner.find_full_trajectory(
                x0=[0, 0, 0], 
                nominal_speed=2.0, 
                stride=5
            )
            import matplotlib.pyplot as plt
            plt.plot(xs[:, 0], xs[:, 1])
            plt.show()
        """
        x_curr = np.array(x0, dtype=float)
        
        full_xs = [x_curr]
        full_us = []
        
        # Reset internal planner state
        self.last_s = 0.0
        self.prev_us = None
        
        ix, iy = self.idxs[0], self.idxs[1]
        
        step_count = 0
        while step_count < max_steps:
            
            # Solve optimization for current horizon
            xs_horizon, us_horizon = self.find_trajectory(
                x_curr, 
                max_iters=max_iters, 
                nominal_speed=nominal_speed,
                reset=False
            )
            
            # Determine execution stride (cannot exceed remaining horizon)
            actual_stride = min(stride, self.N)
            
            # Extract trajectory segment to 'execute'
            next_xs = xs_horizon[1 : actual_stride + 1]
            next_us = us_horizon[0 : actual_stride]
            
            # Update history and current state
            full_xs.extend(next_xs)
            full_us.extend(next_us)
            
            x_curr = next_xs[-1]
            step_count += actual_stride

            # Warm-start Correction:
            # We jumped ahead 'stride' steps, so the previous solution (us_horizon) 
            # is now 'stride' steps behind the new time t. We shift it left to match.
            shift_idx = actual_stride
            remaining_controls = us_horizon[shift_idx:]
            
            # Pad the end with the last known control to maintain horizon length
            padding = np.tile(us_horizon[-1], (shift_idx, 1))
            self.prev_us = np.vstack([remaining_controls, padding])

            # Check Termination Conditions
            dist_to_goal = np.linalg.norm(x_curr[[ix, iy]] - self.raw_path[-1])
            progress_ratio = self.last_s / self.total_len
            
            if dist_to_goal < goal_tolerance and progress_ratio >= 0.95:
                break

        return np.array(full_xs), np.array(full_us)

"""
Linear Interpolation Planner (Robust Fallback)
"""

class LinearPlanner:
    """
    Trajectory reference planner that generates a state reference trajectory 
    by linearly interpolating between waypoints.

    Unlike the Spline planner, this does not attempt to smooth corners.
    It generates a reference that:
    1. Linearly connects the robot's CURRENT position to the current target waypoint.
    2. Then follows the subsequent path segments linearly.
    
    This is often more stable for QP because the reference always starts 
    exactly at the robot's position, reducing initial error terms in the solver.
    """

    def __init__(self, solver: SQPSolver,
                 path: Union[List[Point], np.ndarray],
                 stable_state: np.ndarray,
                 ref_state_indices: Optional[List[int]] = None,
                 waypoint_tol: float = 0.5): # Tolerance to switch to next waypoint
        
        """
        :param solver: MPCSolver instance.
        :param path: Waypoints (N, 2) or (N, 3).
        :param stable_state: Equilibrium state to fill non-spatial dimensions.
        :param ref_state_indices: Indices for [x, y, theta]. Defaults to [0, 1, 2].
        :param waypoint_tol: Distance tolerance to consider a waypoint "reached".
        """
        
        self.solver = solver
        # If solver has field, use it, otherwise ignore
        self.field = getattr(solver, 'field', None)
        
        self.ref_idx = ref_state_indices if ref_state_indices is not None else [0, 1, 2]

        self.waypoint_tol = waypoint_tol
        
        self.way_idx = 0
        self.prev_us = None
        self.path = None
        self.stable_state = np.array(stable_state, dtype=float)

        # Cache containers
        self.cached_seg_lens = None
        self.cached_directions = None
        self.cached_cum_len = None

        self.update_path(path)

    def obtain_waypoint(self, x0: Optional[np.ndarray] = None):
        """
        Checks if the robot (x0) has reached the current waypoint.
        If so, increments the waypoint index.
        """
        if x0 is not None:
            # Only check X,Y distance
            pos = x0[:2]
            target = self.path[self.way_idx, :2]
            
            dist = np.linalg.norm(pos - target)
            
            # While we are close to current waypoint, advance to next
            while dist < self.waypoint_tol:
                if self.way_idx >= len(self.path) - 1:
                    break
                self.way_idx += 1
                target = self.path[self.way_idx, :2]
                dist = np.linalg.norm(pos - target)
                
        return self.path[self.way_idx]

    def reset_path(self):
        self.way_idx = 0
        self.prev_us = None
    
    def update_path(self, path: Union[List[Point], np.ndarray]):
        self.path = np.atleast_2d(np.copy(path))

        # 1. Handle Headings (if only X,Y provided)
        # We compute orientation based on the segment direction
        if self.path.shape[1] == 2:
            deltas = self.path[1:] - self.path[:-1]
            headings = np.arctan2(deltas[:, 1], deltas[:, 0])
            if headings.size > 0:
                headings = np.append(headings, headings[-1])
            else:
                headings = np.array([0.0])
            self.path = np.hstack((self.path, headings[:, None]))
        
        # 2. Pre-compute Segment Math
        # A. Spatial Deltas (X, Y)
        d_xy = self.path[1:, :2] - self.path[:-1, :2]
        
        # B. Segment Lengths (N-1,)
        # Use maximum to avoid division by zero
        self.cached_seg_lens = np.maximum(np.linalg.norm(d_xy, axis=1), 1e-6)
        
        # C. Angular Deltas (Theta) with Wrapping
        d_theta = self.path[1:, 2] - self.path[:-1, 2]
        d_theta = (d_theta + np.pi) % (2*np.pi) - np.pi
        
        # D. Normalized Direction Vectors (N-1, 3) for [x, y, theta]
        # This vector represents the change per meter of arc length
        vecs = np.column_stack((d_xy, d_theta))
        self.cached_directions = vecs / self.cached_seg_lens[:, None]
        
        # E. Cumulative Length from start of path (N,)
        self.cached_cum_len = np.concatenate(([0.0], np.cumsum(self.cached_seg_lens)))

        self.reset_path()
    
    def get_ref(self, x0: np.ndarray, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Generates N reference points.
        
        Strategy:
        1. Calculate vector from Robot -> Current Waypoint.
        2. Generate reference points along this vector until the waypoint is reached.
        3. If horizon extends past the waypoint, continue along the cached path segments.
        """
        step_size = nominal_speed * self.solver.dt
        N = self.solver.N
        
        # 1. Update waypoint index based on current robot position
        # We use indices from ref_state_indices to extract [x, y, theta]
        path_x0 = x0[self.ref_idx]
        self.obtain_waypoint(path_x0)
        
        # 2. Dynamic approach vector (Robot Pos -> Next Waypoint)
        curr_waypoint = self.path[self.way_idx]
        
        # Diff X, Y
        diff_xy = curr_waypoint[:2] - path_x0[:2]
        dist_0 = np.linalg.norm(diff_xy)
        dist_0 = max(dist_0, 1e-6)
        
        # Diff Theta (Wrapped)
        diff_theta = (curr_waypoint[2] - path_x0[2] + np.pi) % (2*np.pi) - np.pi
        
        # Direction vector for the "Approach Phase"
        dir_0 = np.append(diff_xy, diff_theta) / dist_0

        # 3. Generate Reference Points
        # Distances along the prediction horizon
        s_vals = step_size * np.arange(N)
        
        # Container for [x, y, theta]
        ref_path = np.zeros((N, 3))
        
        # A. Approach Phase (Points before reaching the current waypoint)
        mask_head = s_vals <= dist_0
        if np.any(mask_head):
            # Linearly interpolate from robot pos to waypoint
            ref_path[mask_head] = path_x0 + dir_0 * s_vals[mask_head, None]

        # B. Path Following Phase (Points after reaching the current waypoint)
        if np.any(~mask_head):
            if self.way_idx >= len(self.path) - 1:
                # End of path reached: Just stay at the last waypoint
                ref_path[~mask_head] = curr_waypoint
            else:
                # We need to sample from the cached segments starting from the current waypoint
                
                # Distance remaining after reaching the current waypoint
                s_tail = s_vals[~mask_head] - dist_0
                
                # Get lengths of segments remaining in the path
                path_cum_lens = self.cached_cum_len[self.way_idx:] 
                local_cum_len = path_cum_lens - path_cum_lens[0]
                
                # Find which segment each s_tail falls into
                idx = np.searchsorted(local_cum_len, s_tail, side="right") - 1
                
                # Clip to valid segments
                max_seg_idx = len(self.cached_seg_lens) - self.way_idx - 1
                idx = np.clip(idx, 0, max_seg_idx)
                
                # Map local index back to global path index
                global_idx = self.way_idx + idx
                
                # Distance into that specific segment
                ds = s_tail - local_cum_len[idx]
                
                # Interpolate: Start of Segment + Direction * Distance into segment
                ref_path[~mask_head] = (self.path[global_idx] + 
                                        self.cached_directions[global_idx] * ds[:, None])

        # 4. Finalize
        # Wrap theta one last time to be safe
        ref_path[:, 2] = (ref_path[:, 2] + np.pi) % (2 * np.pi) - np.pi

        # Fill the full state vector (e.g. including velocities which we set to 0 or stable_state)
        full_ref = np.empty((N, len(self.stable_state)))
        full_ref[:] = self.stable_state
        
        # Overwrite the [x, y, theta] indices with our computed path
        full_ref[:, self.ref_idx] = ref_path
        
        return full_ref

    def find_trajectory(self, x0: np.ndarray, 
                        max_iters: int = 10, 
                        nominal_speed: float = 2.0, 
                        reset: bool = False) -> Trajectory:
        """
        Solve for a state-control trajectory from x0.
        """

        if reset:
            # On reset, we might want to iterate more to converge
            max_iters = max(max_iters, 20)
            self.reset_path()

        # Generate reference using nominal_speed
        ref_traj = self.get_ref(x0, nominal_speed=nominal_speed)

        # Solve MPC
        xs, us = self.solver.solve(x0, ref_traj, us_init=self.prev_us, max_iters=max_iters)

        # Cache controls for warm start
        self.prev_us = np.vstack([us[1:], us[-1:]])

        return xs, us
    
    def get_full_ref(self, nominal_speed=2.0):
        # Fallback if you need visualization of the full path
        return self.path