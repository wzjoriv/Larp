"""
larp/tp/planner.py
==================
Reference-trajectory generators for the larp trajectory-planning framework.

Planner hierarchy
-----------------
Planner (ABC)           - shared interface: update_path, get_ref, get_full_ref,
                          find_trajectory, get_full_trajectory
├ WaypointPlanner     - arc-length-projection following on the raw piecewise-linear path
└ SplinePlanner       - B-spline path following with curvature-based velocity profiling;
                        degree selectable (default 3 cubic, 5 for quintic smoothness)

Public API summary
------------------
``get_ref(x0, nominal_speed)``
    Dense (N, n) reference for one solver horizon, starting one step ahead of
    the robot's current position (x₁ … xₙ).  Call this inside a real-time loop.

``get_full_ref(nominal_speed)``
    Complete (T, n) reference showing the idealised path the robot will
    follow over its entire journey.  Useful for visualisation and comparing
    actual vs. intended motion.  WaypointPlanner uses piecewise-linear
    interpolation; SplinePlanner samples its fitted curve for a smoother result.

``find_trajectory(x0, ...)``
    Call ``get_ref`` then run the solver.  Returns the optimised predictive
    trajectory ``(xs, us)`` for the current horizon.

``get_full_trajectory(x0, ..., stride=1)``
    Plan the complete trajectory from ``x0`` to the goal **before the robot
    starts moving**, assuming a static environment.  Useful for pre-computing
    a route, inspecting solver behaviour end-to-end, or pre-loading a complete
    trajectory into a tracking controller.  ``stride`` controls how many
    simulated steps are taken between replanning calls.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union

import numpy as np
from scipy.interpolate import make_interp_spline

from larp.ltp.solver.solver import Solver
from larp.types import Point, Trajectory


# Abstract base

class RPlanner(ABC):
    """
    Abstract base class for all larp reference-trajectory planners.

    Subclasses must implement :meth:`get_ref`.  All other public methods
    — :meth:`get_full_ref`, :meth:`find_trajectory`, :meth:`get_full_trajectory`
    — have concrete implementations on this class and are inherited for free.

    Parameters
    ----------
    solver : Solver
        Any concrete ``Solver`` instance; exposes ``dt``, ``N``, and ``solve``.
    path : array-like, shape (M, 2) or (M, 3)
        Ordered waypoints in the robot's spatial coordinates.
        A third column is interpreted as a prescribed heading (radians).
    stable_state : array-like, shape (n,)
        Equilibrium state used to fill non-spatial dimensions of each reference
        point (e.g. zero velocity, nominal altitude).
    ref_state_indices : list of int, optional
        Indices ``[i_x, i_y, i_yaw]`` into the full state vector.
        Defaults to ``[0, 1, 2]``.
    """

    def __init__(
        self,
        solver: Solver,
        path: Union[List[Point], np.ndarray],
        stable_state: np.ndarray,
        ref_state_indices: Optional[List[int]] = None,
    ):
        self.solver = solver
        self.stable_state = np.array(stable_state, dtype=float)
        self.ref_idx: List[int] = (ref_state_indices
                                   if ref_state_indices is not None
                                   else [0, 1, 2])

        # Populated by update_path
        self.path: Optional[np.ndarray] = None
        self.cached_seg_lens: Optional[np.ndarray] = None
        self.cached_directions: Optional[np.ndarray] = None
        self.cached_seg_headings: Optional[np.ndarray] = None
        self.cached_cum_len: Optional[np.ndarray] = None
        self.total_len: float = 0.0
        self._use_custom_heading: bool = False

        # Warm-start control sequence
        self.prev_us: Optional[np.ndarray] = None

        self.update_path(path)

    #  Path management 

    def update_path(self, path: Union[List[Point], np.ndarray]):
        """
        Load a new path and pre-compute cached geometry.

        Parameters
        ----------
        path : array-like, shape (M, 2) or (M, 3)
            ``path[:, :2]`` are XY positions; optional third column is heading.
        """
        points = np.atleast_2d(np.copy(path))
        if points.shape[0] < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        xy   = points[:, :2]
        d_xy = xy[1:] - xy[:-1]

        seg_lens = np.maximum(np.linalg.norm(d_xy, axis=1), 1e-8)

        self.cached_seg_lens  = seg_lens
        self.cached_directions = d_xy / seg_lens[:, None]
        self.cached_cum_len   = np.concatenate(([0.0], np.cumsum(seg_lens)))
        self.total_len        = float(self.cached_cum_len[-1])

        if points.shape[1] >= 3:
            self._use_custom_heading = True
            self.path = points[:, :3].copy()
            self.cached_seg_headings = None
        else:
            self._use_custom_heading = False
            seg_headings = np.arctan2(d_xy[:, 1], d_xy[:, 0])
            final_heading = seg_headings[-1] if seg_headings.size > 0 else 0.0
            all_headings = np.append(seg_headings, final_heading)
            self.path = np.column_stack((xy, all_headings))
            self.cached_seg_headings = seg_headings

        self._reset_state()

    def _reset_state(self):
        """Reset per-path planner state (progress bookmark, warm start)."""
        self.prev_us = None

    #  Abstract interface 

    @abstractmethod
    def get_ref(self, x0: np.ndarray, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Generate a dense reference for the current solver horizon.

        Returns (N, n) states x₁ … xₙ — one step ahead of ``x0`` through N
        steps ahead — ready to pass to ``Solver.solve(x0, ref)``.

        Parameters
        ----------
        x0 : (n,) current state
        nominal_speed : float
            Target progression speed along the path (m/s).

        Returns
        -------
        ref : (N, n) reference states
        """

    #  Concrete shared methods 

    def get_full_ref(self, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Sample the complete reference the robot will follow over its journey.

        Walks the piecewise-linear path geometry from start to end at
        ``solver.dt`` intervals.  This is the *reference path* — the
        idealised line the robot is trying to follow — not an optimised
        trajectory.  Use it to visualise the planned route or to compare
        actual vs. intended motion.

        Subclasses may override for a smoother representation (SplinePlanner
        and QuinticPlanner sample their fitted curve instead).

        Parameters
        ----------
        nominal_speed : float
            Determines the number of steps T:
            ``T = ceil(total_len / (nominal_speed * dt))``.

        Returns
        -------
        full_ref : (T, n) reference states from path start to path end
        """
        if self.total_len <= 0:
            return np.empty((0, len(self.stable_state)))

        dt        = self.solver.dt
        num_steps = int(np.ceil(self.total_len / max(nominal_speed * dt, 1e-8)))
        s_vals    = np.linspace(0.0, self.total_len, num_steps)

        seg_idxs  = np.searchsorted(self.cached_cum_len, s_vals, side='right') - 1
        seg_idxs  = np.clip(seg_idxs, 0, len(self.path) - 2)

        ds        = s_vals - self.cached_cum_len[seg_idxs]
        positions = (self.path[seg_idxs, :2]
                     + self.cached_directions[seg_idxs] * ds[:, None])

        headings = (self.path[seg_idxs, 2] if self._use_custom_heading
                    else self.cached_seg_headings[seg_idxs])

        if not self._use_custom_heading:
            headings = np.unwrap(headings)
        ix, iy, ith = self.ref_idx[0], self.ref_idx[1], self.ref_idx[2]
        full_ref = np.tile(self.stable_state, (num_steps, 1))
        full_ref[:, ix]  = positions[:, 0]
        full_ref[:, iy]  = positions[:, 1]
        full_ref[:, ith] = headings
        return full_ref

    def find_trajectory(
        self,
        x0: np.ndarray,
        max_iters: int = 10,
        nominal_speed: float = 2.0,
        reset: bool = False,
    ) -> Trajectory:
        """
        Generate a reference and solve for the optimised predictive trajectory.

        The intended production use is that a separate high-frequency feedback
        controller tracks this trajectory; during testing the first step of
        ``(xs, us)`` can be applied directly to advance the simulation.

        Parameters
        ----------
        x0 : (n,) current state
        max_iters : int
            Maximum solver iterations.
        nominal_speed : float
            Target progression speed (m/s).
        reset : bool
            If True, resets progress state and bumps max_iters to at least 20.

        Returns
        -------
        xs : (N+1, n) optimised state trajectory over the horizon
        us : (N, m)   corresponding control sequence
        """
        if reset:
            self._reset_state()
            max_iters = max(max_iters, 20)

        ref_traj = self.get_ref(x0, nominal_speed=nominal_speed)
        xs, us   = self.solver.solve(x0, ref_traj,
                                     us_init=self.prev_us,
                                     max_iters=max_iters)
        self.prev_us = us
        return xs, us

    def get_full_trajectory(
        self,
        x0: np.ndarray,
        nominal_speed: float = 2.0,
        goal_tolerance: float = 1.0,
        max_steps: int = 10000,
        max_iters: int = 10,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a complete optimised trajectory from ``x0`` to the path goal.

        Intended for cases where the environment is static and the full
        trajectory can be planned **before the robot starts moving** — for
        example to inspect the planned route, detect potential issues, or
        pre-load a trajectory into a tracking controller.

        Repeatedly calls :meth:`find_trajectory` (receding-horizon), advances
        ``stride`` simulated steps from the returned horizon, and records the
        full history.

        ``stride=1`` (default) replans after every step — most accurate.
        Larger strides replan less frequently — useful for offline analysis
        or reducing compute.  ``stride`` is clamped to ``[1, N]``.

        Parameters
        ----------
        x0 : (n,) initial state
        nominal_speed : float
            Target progression speed (m/s).
        goal_tolerance : float
            Euclidean distance (m) to the final waypoint to consider done.
        max_steps : int
            Hard limit on total simulated steps.
        max_iters : int
            Maximum solver iterations per planning cycle.
        stride : int
            Steps to execute per horizon before replanning.  Default 1.

        Returns
        -------
        xs_full : (T+1, n) state history from x0 through T simulated steps
        us_full : (T, m)   control history
        """
        stride      = int(np.clip(stride, 1, self.solver.N))
        x_cur       = np.array(x0, dtype=float)
        all_xs      = [x_cur.copy()]
        all_us: List[np.ndarray] = []

        ix, iy      = self.ref_idx[0], self.ref_idx[1]
        goal_xy     = self.path[-1, :2]
        steps_taken = 0

        while steps_taken < max_steps:
            xs, us = self.find_trajectory(x_cur,
                                           nominal_speed=nominal_speed,
                                           max_iters=max_iters)
            actual = min(stride, max_steps - steps_taken)

            for i in range(actual):
                all_xs.append(xs[i + 1].copy())
                all_us.append(us[i].copy())

            x_cur       = xs[actual].copy()
            steps_taken += actual

            # Warm-start: shift by `actual` steps
            self.prev_us = np.vstack([us[actual:], np.tile(us[-1], (actual, 1))])

            if np.linalg.norm(x_cur[[ix, iy]] - goal_xy) < goal_tolerance:
                break

        return np.array(all_xs), np.array(all_us)