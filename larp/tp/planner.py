"""
larp/tp/planner.py
==================
Reference-trajectory generators for the larp trajectory-planning framework.

Planner hierarchy
-----------------
Planner (ABC)           - shared interface: update_path, get_ref, get_full_ref,
                          find_trajectory, get_full_trajectory
├── WaypointPlanner     - arc-length-projection following on the raw piecewise-linear path
├── SplinePlanner       - cubic-spline (C2) path following with velocity profiling
└── QuinticPlanner      - quintic B-spline (C4) path following with velocity profiling

Public API summary
------------------
``get_ref(x0, nominal_speed)``
    Dense (N, n) reference for one solver horizon, starting one step ahead of
    the robot's current position (x₁ … xₙ).  Call this inside a real-time loop.

``get_full_ref(nominal_speed)``
    Complete (T, n) reference showing the idealised path the robot will
    follow over its entire journey.  Useful for visualisation and comparing
    actual vs. intended motion.  A generic piecewise-linear implementation
    is provided on the base class; subclasses may override for a smoother
    result (e.g. SplinePlanner and QuinticPlanner sample their fitted curve).

``find_trajectory(x0, ...)``
    Call ``get_ref`` then run the solver.  Returns the optimised predictive
    trajectory ``(xs, us)`` for the current horizon.

``get_full_trajectory(x0, ..., stride=1)``
    Plan the complete trajectory from ``x0`` to the goal **before the robot
    starts moving**, assuming a static environment.  Useful for pre-computing
    a route, inspecting solver behaviour end-to-end, or pre-loading a complete
    trajectory into a tracking controller.  ``stride`` controls how many
    simulated steps are taken between replanning calls.

Both planners accept a pre-computed path from any upstream path-finder
(e.g. ``larp.pp.QuadPlanner``) and produce references that feed directly
into ``Solver.solve()``.

WaypointPlanner strategy
~~~~~~~~~~~~~~~~~~~~~~~~
Progress is determined by arc-length projection: every ``get_ref`` call
projects the robot's XY position onto the nearest point on the
piecewise-linear path (arc-length ``s_robot``) and generates reference
states by walking forward from ``s_robot + lookahead``.  A windowed scan
with automatic global fallback keeps projection O(W) in the common case
and O(M) only when recovery is needed.

SplinePlanner strategy
~~~~~~~~~~~~~~~~~~~~~~
A natural cubic spline (C2) is fitted through the waypoints and progress is
tracked by projecting the robot's XY position onto the piecewise-linear
path skeleton (same windowed + global-fallback projection as WaypointPlanner).
The reference is then sampled from the smooth spline rather than the skeleton,
so corners are naturally rounded.  A curvature-based speed limit
``v <= sqrt(a_lat_max / kappa)`` automatically slows the reference on tight
bends.  A heading-alignment penalty prevents the projection from snapping to
the wrong branch on U-shaped or looping paths.

QuinticPlanner strategy
~~~~~~~~~~~~~~~~~~~~~~~
Same projection and velocity-profiling approach as SplinePlanner, but uses a
quintic B-spline (degree 5, C4 continuous) rather than a cubic spline (C2).
C4 continuity means position, velocity, acceleration, jerk, and snap are all
continuous across waypoints, giving the smoothest possible acceleration profile.
This is particularly beneficial for dynamically sensitive systems where abrupt
changes in acceleration cause vibration or payload disturbance.

For paths with fewer than 6 waypoints the degree is silently reduced so that
a valid spline can always be fitted: 2 pts → linear, 3 → quadratic, 4 → cubic,
5 → quartic, 6+ → quintic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from larp.tp.solver import Solver
from larp.types import Point, Trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class Planner(ABC):
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

    # ── Path management ────────────────────────────────────────────────────

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

    # ── Abstract interface ─────────────────────────────────────────────────

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

    # ── Concrete shared methods ────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# WaypointPlanner
# ══════════════════════════════════════════════════════════════════════════════

class WaypointPlanner(Planner):
    """
    Reference planner using **arc-length projection** to determine path progress.

    Every call to :meth:`get_ref` projects the robot onto the piecewise-linear
    path (O(M) vectorised), then generates reference states by walking forward
    along the arc from the projected position.

    Near-goal behaviour
    ~~~~~~~~~~~~~~~~~~~
    When ``s_robot >= total_len - goal_blend_dist`` the reference saturates
    to the final waypoint state, giving the solver a stationary target and
    preventing oscillations near the goal.

    Parameters
    ----------
    solver : Solver
    path : array-like, shape (M, 2) or (M, 3)
    stable_state : array-like, shape (n,)
    ref_state_indices : list of int, optional
    lookahead : float
        Extra arc-length offset added to the projected position before
        generating the horizon.  Default 0.0.
    goal_blend_dist : float
        Arc-length window near the end where the reference saturates to the
        goal state.  Default 0.0.
    projection_window : int
        Segments to scan ahead during arc-length projection.  The planner
        uses a windowed scan with automatic global fallback: if the best
        match is at the far window edge a full O(M) scan recovers the correct
        position.  A practical lower bound is
        ``nominal_speed * dt * N / min_seg_len``.  Default 15.
    """

    def __init__(
        self,
        solver: Solver,
        path: Union[List[Point], np.ndarray],
        stable_state: np.ndarray,
        ref_state_indices: Optional[List[int]] = None,
        lookahead: float = 0.0,
        goal_blend_dist: float = 0.0,
        projection_window: int = 15,
    ):
        self.lookahead          = lookahead
        self.goal_blend_dist    = goal_blend_dist
        self._seg_idx: int      = 0
        self._projection_window = max(1, int(projection_window))
        self._last_s: float     = 0.0

        super().__init__(solver, path, stable_state, ref_state_indices)

    def _reset_state(self):
        super()._reset_state()
        self._last_s  = 0.0
        self._seg_idx = 0

    # ── Arc-length projection ──────────────────────────────────────────────

    def _project_to_path(self, pos: np.ndarray) -> float:
        """
        Project a 2-D position onto the piecewise-linear path.

        Windowed scan O(W) in the normal case; automatic O(M) global fallback
        when the robot has outrun the window.  Result is monotone: the return
        value is always >= ``_last_s``.
        """
        M = len(self.path) - 1

        def _scan(a: int, b: int):
            P_start     = self.path[a:b,    :2]
            P_end       = self.path[a+1:b+1, :2]
            seg_vecs    = P_end - P_start
            robot_vecs  = pos - P_start
            seg_lens_sq = np.sum(seg_vecs ** 2, axis=1)
            t_vals      = np.clip(
                np.sum(robot_vecs * seg_vecs, axis=1)
                / np.maximum(seg_lens_sq, 1e-10),
                0.0, 1.0,
            )
            closest  = P_start + t_vals[:, None] * seg_vecs
            dists_sq = np.sum((closest - pos) ** 2, axis=1)
            i        = int(np.argmin(dists_sq))
            s = float(self.cached_cum_len[a + i]
                      + t_vals[i] * np.sqrt(max(seg_lens_sq[i], 0.0)))
            return a + i, s

        start    = max(0, self._seg_idx - 2)
        end      = min(M, self._seg_idx + self._projection_window)
        best, s  = _scan(start, end)

        if best == end - 1 and end < M:   # outran window → global fallback
            best, s = _scan(0, M)

        s = max(s, self._last_s)
        self._seg_idx = max(self._seg_idx, best)
        return s

    def _interp_on_path(self, s: float) -> Tuple[np.ndarray, float]:
        """Interpolate (XY position, heading) at arc-length ``s``."""
        s       = float(np.clip(s, 0.0, self.total_len))
        seg_idx = int(np.clip(
            np.searchsorted(self.cached_cum_len, s, side='right') - 1,
            0, len(self.path) - 2,
        ))
        ds      = s - self.cached_cum_len[seg_idx]
        pos     = self.path[seg_idx, :2] + self.cached_directions[seg_idx] * ds
        heading = float(self.path[seg_idx, 2] if self._use_custom_heading
                        else self.cached_seg_headings[seg_idx])
        return pos, heading

    def get_ref(self, x0: np.ndarray, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Generate N reference states by walking ahead from the projected position.

        Near-goal blending: within ``goal_blend_dist`` of the path end the
        reference saturates to the final waypoint so the solver sees a
        stationary target.
        """
        ix, iy, ith = self.ref_idx[0], self.ref_idx[1], self.ref_idx[2]

        s_robot      = self._project_to_path(np.array([x0[ix], x0[iy]]))
        self._last_s = s_robot

        dt  = self.solver.dt
        N   = self.solver.N
        hdg_off = self.solver.dynamics.heading_convention_offset

        goal_pos, goal_hdg = self._interp_on_path(self.total_len)
        goal_state         = self.stable_state.copy()
        goal_state[ix]     = goal_pos[0]
        goal_state[iy]     = goal_pos[1]
        goal_state[ith]    = goal_hdg + hdg_off

        ref_states = []
        for k in range(N):
            s_ref = s_robot + self.lookahead + (k + 1) * nominal_speed * dt

            if s_ref >= self.total_len - self.goal_blend_dist:
                ref_states.append(goal_state)
                continue

            pos_ref, hdg_ref = self._interp_on_path(s_ref)
            state            = self.stable_state.copy()
            state[ix]        = pos_ref[0]
            state[iy]        = pos_ref[1]
            state[ith]       = hdg_ref + hdg_off
            ref_states.append(state)

        return np.array(ref_states)

    # ── Helpers ────────────────────────────────────────────────────────────

    def reset_path(self):
        """Reset waypoint progress to the start of the path."""
        self._reset_state()

    @property
    def seg_idx(self) -> int:
        """Current best-match segment index into ``self.path``."""
        return self._seg_idx


# ══════════════════════════════════════════════════════════════════════════════
# Internal base for curve-based planners
# ══════════════════════════════════════════════════════════════════════════════

class _CurvePlanner(Planner):
    """
    Internal base for planners that maintain a smooth parametric curve over
    the path (SplinePlanner uses a cubic spline; QuinticPlanner uses a quintic
    B-spline).

    Provides:
    - Shared arc-length projection with heading-alignment penalty
    - Shared curvature-aware ``get_ref`` template
    - Shared ``get_full_ref`` that samples the fitted curve

    Subclasses must set ``self._curve`` (callable with the same interface as
    scipy ``CubicSpline``) and ``self.raw_path`` inside their ``update_path``
    implementation, and call ``self._reset_state()`` at the end.
    """

    def __init__(
        self,
        solver: Solver,
        path: Union[List[Point], np.ndarray],
        stable_state: np.ndarray,
        ref_state_indices: Optional[List[int]] = None,
        lookahead: float = 0.5,
        projection_window: int = 15,
        max_lat_accel: float = 2.0,
    ):
        self.lookahead          = lookahead
        self._projection_window = max(1, int(projection_window))
        self.max_lat_accel      = max_lat_accel

        # Set before super().__init__ so update_path can use them
        self._curve             = None   # set by subclass update_path
        self.raw_path: Optional[np.ndarray] = None

        super().__init__(solver, path, stable_state, ref_state_indices)

    def _reset_state(self):
        super()._reset_state()
        self._last_s: float  = 0.0
        self._seg_idx: int   = 0

    # ── Arc-length projection with heading penalty ─────────────────────────

    def _project_to_path(
        self,
        x_curr: float,
        y_curr: float,
        theta_curr: Optional[float] = None,
    ) -> float:
        """
        Project the robot onto the path skeleton; return arc-length ``s``.

        Windowed scan O(W) with automatic global fallback O(M).
        Optional heading-alignment penalty discourages snapping to the wrong
        branch on U-shaped or looping paths.
        """
        if self.cached_cum_len is None:
            return 0.0

        pos = np.array([x_curr, y_curr])
        M   = len(self.raw_path) - 1

        def _scan(a: int, b: int) -> Tuple[int, float]:
            P_start     = self.raw_path[a:b]
            P_end       = self.raw_path[a+1:b+1]
            seg_vecs    = P_end - P_start
            robot_vecs  = pos - P_start
            seg_lens_sq = np.sum(seg_vecs ** 2, axis=1)
            t_vals      = np.clip(
                np.sum(robot_vecs * seg_vecs, axis=1)
                / np.maximum(seg_lens_sq, 1e-10),
                0.0, 1.0,
            )
            closest  = P_start + t_vals[:, None] * seg_vecs
            dists_sq = np.sum((closest - pos) ** 2, axis=1)

            if theta_curr is not None:
                seg_dirs  = seg_vecs / np.sqrt(
                    np.maximum(seg_lens_sq, 1e-10))[:, None]
                robot_dir = np.array([np.cos(theta_curr), np.sin(theta_curr)])
                dists_sq[np.dot(seg_dirs, robot_dir) < -0.5] += 1000.0

            i = int(np.argmin(dists_sq))
            s = float(self.cached_cum_len[a + i]
                      + t_vals[i] * np.sqrt(max(seg_lens_sq[i], 0.0)))
            return a + i, s

        start    = max(0, self._seg_idx - 2)
        end      = min(M, self._seg_idx + self._projection_window)
        best, s  = _scan(start, end)

        if best == end - 1 and end < M:
            best, s = _scan(0, M)

        s = max(s, self._last_s)
        self._seg_idx = max(self._seg_idx, best)
        return s

    # ── Reference generation (shared by SplinePlanner & QuinticPlanner) ───

    def get_ref(self, x0: np.ndarray, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Generate N reference states with curvature-based velocity profiling.

        Returns states x₁ … xₙ — one step ahead of ``x0`` through N steps.
        Samples ``self._curve`` at arc-length positions computed by walking
        forward from the projected robot position.
        """
        ix, iy, ith = self.ref_idx[0], self.ref_idx[1], self.ref_idx[2]

        current_s    = self._project_to_path(x0[ix], x0[iy],
                                              theta_curr=x0[ith])
        self._last_s = current_s

        dt = self.solver.dt
        N  = self.solver.N

        s_future   = [current_s + self.lookahead]
        all_states = []

        for _ in range(N + 1):
            s_curr    = s_future[-1]
            s_clamped = float(np.clip(s_curr, 0.0, self.total_len))

            der1 = self._curve(s_clamped, 1)
            der2 = self._curve(s_clamped, 2)
            cross     = der1[0] * der2[1] - der1[1] * der2[0]
            norm_sq   = der1[0] ** 2 + der1[1] ** 2
            curvature = abs(cross) / (norm_sq ** 1.5 + 1e-8)

            target_v = (min(nominal_speed,
                            np.sqrt(self.max_lat_accel / curvature))
                        if curvature > 1e-4 else nominal_speed)

            if s_curr >= self.total_len:
                target_v = 0.0

            pos      = self._curve(s_clamped)
            tan_norm = np.linalg.norm(der1)
            tan_vec  = (der1 / tan_norm if tan_norm >= 1e-6
                        else np.array([np.cos(x0[ith]), np.sin(x0[ith])]))

            yaw      = np.arctan2(tan_vec[1], tan_vec[0])
            prev_yaw = all_states[-1][ith] if all_states else x0[ith]
            yaw_diff = (yaw - prev_yaw + np.pi) % (2 * np.pi) - np.pi
            yaw      = prev_yaw + yaw_diff

            state        = self.stable_state.copy()
            state[ix]    = pos[0]
            state[iy]    = pos[1]
            state[ith]   = yaw + self.solver.dynamics.heading_convention_offset

            if len(self.ref_idx) >= 5:
                ivx, ivy    = self.ref_idx[3], self.ref_idx[4]
                state[ivx]  = tan_vec[0] * target_v
                state[ivy]  = tan_vec[1] * target_v

            all_states.append(state)
            s_future.append(s_curr + target_v * dt)

        # Discard k=0 (current position) → return x₁ … xₙ
        return np.array(all_states[1:])

    def get_full_ref(self, nominal_speed: float = 2.0) -> np.ndarray:
        """
        Sample the complete reference from path start to end via the fitted curve.

        Overrides the base-class piecewise-linear implementation with smooth
        curve-sampled positions, continuous tangent-derived headings, and
        optionally filled velocity indices.

        Parameters
        ----------
        nominal_speed : float
            Determines T via ``T = ceil(total_len / (nominal_speed * dt))``.

        Returns
        -------
        full_ref : (T, n) reference states
        """
        if self.total_len <= 0:
            return np.empty((0, len(self.stable_state)))

        dt        = self.solver.dt
        num_steps = int(np.ceil(self.total_len
                                / max(nominal_speed * dt, 1e-8)))
        s_vals    = np.linspace(0.0, self.total_len, num_steps)

        pos_ref  = self._curve(s_vals)
        vel_tan  = self._curve(s_vals, 1)
        norms    = np.linalg.norm(vel_tan, axis=1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        vel_vec  = (vel_tan / norms) * nominal_speed
        headings = np.unwrap(np.arctan2(vel_vec[:, 1], vel_vec[:, 0]),
                             discont=np.pi)

        ix, iy, ith = self.ref_idx[0], self.ref_idx[1], self.ref_idx[2]
        full_ref    = np.tile(self.stable_state, (num_steps, 1))
        full_ref[:, ix]  = pos_ref[:, 0]
        full_ref[:, iy]  = pos_ref[:, 1]
        full_ref[:, ith] = headings

        if len(self.ref_idx) >= 5:
            ivx, ivy = self.ref_idx[3], self.ref_idx[4]
            full_ref[:, ivx] = vel_vec[:, 0]
            full_ref[:, ivy] = vel_vec[:, 1]

        return full_ref

    @property
    def seg_idx(self) -> int:
        """Current best-match segment index into the path skeleton."""
        return self._seg_idx


# ══════════════════════════════════════════════════════════════════════════════
# SplinePlanner
# ══════════════════════════════════════════════════════════════════════════════

class SplinePlanner(_CurvePlanner):
    """
    Reference planner using a natural **cubic spline** (C2 continuous).

    Fits a cubic spline through the path waypoints so corners are naturally
    rounded, then samples it to build the reference.  A curvature-based
    speed limit ``v <= sqrt(a_lat_max / kappa)`` slows the reference on
    tight bends.  A heading-alignment penalty keeps projection stable on
    U-shaped paths.

    Parameters
    ----------
    solver : Solver
    path : array-like, shape (M, 2)
        Only XY is used; headings are derived from the spline tangent.
    stable_state : array-like, shape (n,)
    ref_state_indices : list of int, optional
        Must contain at least [i_x, i_y, i_yaw].  Optional 4th and 5th
        entries are velocity indices [i_vx, i_vy] filled from the tangent.
    lookahead : float
        Extra arc length ahead of the projected position.  Default 0.5.
    projection_window : int
        Windowed projection window size.  Default 15.
    max_lat_accel : float
        Maximum lateral acceleration for speed limiting (m/s²).  Default 2.0.
    """

    def __init__(
        self,
        solver: Solver,
        path: Union[List[Point], np.ndarray],
        stable_state: np.ndarray,
        ref_state_indices: Optional[List[int]] = None,
        lookahead: float = 0.5,
        projection_window: int = 15,
        max_lat_accel: float = 2.0,
    ):
        super().__init__(solver, path, stable_state, ref_state_indices,
                         lookahead=lookahead,
                         projection_window=projection_window,
                         max_lat_accel=max_lat_accel)

    def update_path(self, path: Union[List[Point], np.ndarray]):
        points  = np.atleast_2d(np.copy(path))
        if points.shape[0] < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        xy      = points[:, :2]
        deltas  = xy[1:] - xy[:-1]
        dists   = np.maximum(np.linalg.norm(deltas, axis=1), 1e-8)
        cum     = np.concatenate(([0.0], np.cumsum(dists)))

        # Fit cubic spline; expose as both cs (public) and _curve (internal)
        self.cs      = CubicSpline(cum, xy, bc_type='natural')
        self._curve  = self.cs
        self.raw_path = xy

        # Piecewise-linear geometry for projection
        self.cached_cum_len      = cum
        self.total_len           = float(cum[-1])
        self.cached_seg_lens     = dists
        self.cached_directions   = deltas / dists[:, None]
        seg_headings             = np.arctan2(deltas[:, 1], deltas[:, 0])
        final_heading            = seg_headings[-1] if seg_headings.size > 0 else 0.0
        self.path                = np.column_stack(
            (xy, np.append(seg_headings, final_heading)))
        self.cached_seg_headings = seg_headings
        self._use_custom_heading = False

        self._reset_state()


# ══════════════════════════════════════════════════════════════════════════════
# QuinticPlanner
# ══════════════════════════════════════════════════════════════════════════════

class QuinticPlanner(_CurvePlanner):
    """
    Reference planner using a **quintic B-spline** (degree 5, C4 continuous).

    Compared to ``SplinePlanner`` (cubic, C2):

    * Degree 5 polynomial per segment — C4 continuity at waypoints.
    * Position, velocity, acceleration, jerk, and snap are all continuous
      across waypoints, giving the smoothest possible acceleration profile.
    * Particularly beneficial for dynamically sensitive systems where
      abrupt changes in acceleration cause vibration or payload disturbance
      (e.g. cargo drones, surgical robots, high-speed industrial arms).

    For paths with fewer than 6 waypoints the quintic degree is silently
    reduced so a valid spline can always be fitted:

    ===== =====
    M pts Degree
    ===== =====
    2     1 (linear)
    3     2 (quadratic)
    4     3 (cubic)
    5     4 (quartic)
    6+    5 (quintic)
    ===== =====

    All other behaviour (projection, curvature speed limiting, heading
    penalty, full reference sampling) is identical to ``SplinePlanner``.

    Parameters
    ----------
    solver : Solver
    path : array-like, shape (M, 2)
        Only XY is used; headings are derived from the spline tangent.
    stable_state : array-like, shape (n,)
    ref_state_indices : list of int, optional
        Must contain at least [i_x, i_y, i_yaw].  Optional 4th and 5th
        entries are velocity indices [i_vx, i_vy] filled from the tangent.
    lookahead : float
        Extra arc length ahead of the projected position.  Default 0.5.
    projection_window : int
        Windowed projection window size.  Default 15.
    max_lat_accel : float
        Maximum lateral acceleration for speed limiting (m/s²).  Default 2.0.
    """

    def __init__(
        self,
        solver: Solver,
        path: Union[List[Point], np.ndarray],
        stable_state: np.ndarray,
        ref_state_indices: Optional[List[int]] = None,
        lookahead: float = 0.5,
        projection_window: int = 15,
        max_lat_accel: float = 2.0,
    ):
        self._degree: int = 5   # actual degree set in update_path; store for info
        super().__init__(solver, path, stable_state, ref_state_indices,
                         lookahead=lookahead,
                         projection_window=projection_window,
                         max_lat_accel=max_lat_accel)

    def update_path(self, path: Union[List[Point], np.ndarray]):
        points  = np.atleast_2d(np.copy(path))
        if points.shape[0] < 2:
            raise ValueError("Path must contain at least 2 waypoints.")

        xy      = points[:, :2]
        M       = len(xy)
        deltas  = xy[1:] - xy[:-1]
        dists   = np.maximum(np.linalg.norm(deltas, axis=1), 1e-8)
        cum     = np.concatenate(([0.0], np.cumsum(dists)))

        # Quintic if M >= 6; gracefully degrade for shorter paths
        self._degree = min(5, M - 1)
        self.qs      = make_interp_spline(cum, xy, k=self._degree)
        self._curve  = self.qs
        self.raw_path = xy

        # Piecewise-linear geometry for projection
        self.cached_cum_len      = cum
        self.total_len           = float(cum[-1])
        self.cached_seg_lens     = dists
        self.cached_directions   = deltas / dists[:, None]
        seg_headings             = np.arctan2(deltas[:, 1], deltas[:, 0])
        final_heading            = seg_headings[-1] if seg_headings.size > 0 else 0.0
        self.path                = np.column_stack(
            (xy, np.append(seg_headings, final_heading)))
        self.cached_seg_headings = seg_headings
        self._use_custom_heading = False

        self._reset_state()

    @property
    def degree(self) -> int:
        """Actual polynomial degree used (5 for 6+ waypoints, less otherwise)."""
        return self._degree


# ── Alias ─────────────────────────────────────────────────────────────────

LinearPlanner = WaypointPlanner
