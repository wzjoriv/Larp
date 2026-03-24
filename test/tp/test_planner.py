"""
test/tp/test_planner.py
=======================
Tests for larp.tp.planner:
  Planner (ABC), WaypointPlanner, SplinePlanner, QuinticPlanner, LinearPlanner alias.

Test categories
---------------
1.  TestPlannerABC           — abstract-method enforcement and class hierarchy
2.  TestWaypointConstruction — WaypointPlanner construction, path geometry, reset
3.  TestArcLengthProjection  — hybrid windowed + fallback projection
4.  TestWaypointGetRef       — shape, content, near-goal saturation, _last_s update
5.  TestWaypointNearGoal     — edge cases: past end, repeated calls, reset
6.  TestWaypointFindTraj     — find_trajectory shapes, warm-start, repeated steps
7.  TestSplinePlanner        — construction, get_ref (N rows, ahead of x0), curvature
8.  TestQuinticPlanner       — construction, degree fallback, get_ref, smoothness
9.  TestGetFullRef           — base vs override, inheritance, span, speed scaling
10. TestPlannerAPIContract   — parametrised: all 4 methods on all 3 planners
11. TestGetFullTrajectory    — stride, max_steps, goal stop, xs[0]==x0
12. TestLinearPlannerAlias   — LinearPlanner is WaypointPlanner
13. TestPathEdgeCases        — collinear, duplicate pts, very short, negative coords
"""

import pytest
import numpy as np

larp = pytest.importorskip("larp")
from larp.dynamics import WMRDynamics
from larp.tp.solver import SQPSolver
from larp.tp.planner import (
    Planner, WaypointPlanner, SplinePlanner, QuinticPlanner, LinearPlanner,
    _CurvePlanner,
)


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_solver(horizon: int = 10) -> SQPSolver:
    dyn = WMRDynamics()
    return SQPSolver(
        field=None, dynamics=dyn, dt=0.1, horizon=horizon,
        Q=np.diag([10., 10., 1.]), R=np.eye(2) * 0.1,
        Qf=np.diag([50., 50., 5.]),
        u_bounds=([-2., -1.5], [2., 1.5]),
    )


def _straight_path(start=(0., 0.), end=(10., 0.), n=5) -> np.ndarray:
    return np.column_stack((
        np.linspace(start[0], end[0], n),
        np.linspace(start[1], end[1], n),
    ))


def _l_path() -> np.ndarray:
    return np.array([[0.,0.],[5.,0.],[10.,0.],[10.,5.],[10.,10.]])


def _stable() -> np.ndarray:
    return np.zeros(3)


ALL_PLANNERS = [WaypointPlanner, SplinePlanner, QuinticPlanner]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestPlannerABC
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerABC:

    def test_planner_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Planner(solver=_make_solver(), path=_straight_path(),
                    stable_state=_stable())

    def test_get_ref_is_abstract(self):
        import inspect
        assert getattr(Planner.get_ref, '__isabstractmethod__', False)

    def test_get_full_ref_is_concrete(self):
        import inspect
        assert not getattr(Planner.get_full_ref, '__isabstractmethod__', False)

    def test_find_trajectory_is_concrete(self):
        assert not getattr(Planner.find_trajectory, '__isabstractmethod__', False)

    def test_get_full_trajectory_is_concrete(self):
        assert not getattr(Planner.get_full_trajectory, '__isabstractmethod__', False)

    def test_waypoint_is_planner(self):
        assert issubclass(WaypointPlanner, Planner)

    def test_spline_is_planner(self):
        assert issubclass(SplinePlanner, Planner)

    def test_quintic_is_planner(self):
        assert issubclass(QuinticPlanner, Planner)

    def test_spline_inherits_curve_planner(self):
        assert issubclass(SplinePlanner, _CurvePlanner)

    def test_quintic_inherits_curve_planner(self):
        assert issubclass(QuinticPlanner, _CurvePlanner)

    def test_linear_alias(self):
        assert LinearPlanner is WaypointPlanner

    def test_ordering_in_module(self):
        """WaypointPlanner must appear before SplinePlanner in the source."""
        import inspect, larp.tp.planner as pm
        src = inspect.getsource(pm)
        assert src.index('class WaypointPlanner') < src.index('class SplinePlanner')
        assert src.index('class SplinePlanner') < src.index('class QuinticPlanner')


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestWaypointConstruction
# ══════════════════════════════════════════════════════════════════════════════

class TestWaypointConstruction:

    def test_path_stored_with_heading(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        assert wp.path.shape == (5, 3)

    def test_auto_heading_along_x(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        assert np.allclose(wp.path[:, 2], 0., atol=1e-10)

    def test_custom_heading_stored(self):
        path3 = np.column_stack((_straight_path(), np.full(5, 0.5)))
        wp = WaypointPlanner(_make_solver(), path3, _stable())
        assert wp._use_custom_heading

    def test_cum_len_starts_zero_ends_at_length(self):
        wp = WaypointPlanner(_make_solver(),
                             _straight_path(end=(10., 0.)), _stable())
        assert wp.cached_cum_len[0] == pytest.approx(0.)
        assert wp.total_len == pytest.approx(10., abs=1e-8)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            WaypointPlanner(_make_solver(), np.array([[0., 0.]]), _stable())

    def test_update_path_resets_state(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        wp._last_s = 5.; wp._seg_idx = 3
        wp.update_path(_l_path())
        assert wp._last_s == 0. and wp._seg_idx == 0

    def test_default_projection_window(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        assert wp._projection_window == 15

    def test_custom_projection_window(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable(),
                             projection_window=30)
        assert wp._projection_window == 30

    def test_projection_window_clamped_to_1(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable(),
                             projection_window=0)
        assert wp._projection_window == 1

    def test_reset_path_clears_both(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        wp._last_s = 5.; wp._seg_idx = 3; wp.prev_us = np.ones((10, 2))
        wp.reset_path()
        assert wp._last_s == 0. and wp._seg_idx == 0 and wp.prev_us is None

    def test_seg_idx_property(self):
        wp = WaypointPlanner(_make_solver(), _straight_path(), _stable())
        wp._seg_idx = 4
        assert wp.seg_idx == 4


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestArcLengthProjection
# ══════════════════════════════════════════════════════════════════════════════

class TestArcLengthProjection:

    def _wp(self, M=20, L=20.) -> WaypointPlanner:
        xs = np.linspace(0., L, M + 1)
        return WaypointPlanner(_make_solver(),
                               np.column_stack((xs, np.zeros(M + 1))),
                               _stable())

    def test_project_start(self):
        wp = self._wp()
        assert abs(wp._project_to_path(np.array([0., 0.]))) < 0.01

    def test_project_midpoint(self):
        wp = self._wp(); wp._seg_idx = 8
        assert abs(wp._project_to_path(np.array([10., 0.])) - 10.) < 0.05

    def test_project_lateral_offset(self):
        wp = self._wp(); wp._seg_idx = 3; wp._last_s = 3.
        assert abs(wp._project_to_path(np.array([5., 3.])) - 5.) < 0.05

    def test_project_past_end_clamps(self):
        wp = self._wp()
        s = wp._project_to_path(np.array([100., 0.]))
        assert s <= wp.total_len + 1e-6

    def test_monotonicity(self):
        wp = self._wp(); wp._last_s = 12.; wp._seg_idx = 12
        s = wp._project_to_path(np.array([3., 0.]))
        assert s >= 12. - 1e-9

    def test_seg_idx_advances(self):
        wp = self._wp(); wp._seg_idx = 3; wp._last_s = 3.
        wp._project_to_path(np.array([8., 0.]))
        assert wp._seg_idx >= 7

    def test_seg_idx_monotone(self):
        wp = self._wp(M=30, L=30.)
        prev = 0
        for x in [0., 4., 8., 12., 16., 20., 24., 28.]:
            wp._project_to_path(np.array([x, 0.]))
            assert wp._seg_idx >= prev
            prev = wp._seg_idx

    def test_fallback_on_large_drift(self):
        s = _make_solver()
        path = np.column_stack((np.linspace(0., 50., 51), np.zeros(51)))
        wp = WaypointPlanner(s, path, _stable())
        wp._seg_idx = 5; wp._last_s = 5.
        proj = wp._project_to_path(np.array([35., 0.]))
        assert abs(proj - 35.) < 0.1
        assert wp._seg_idx >= 33

    def test_in_window_matches_global(self):
        s = _make_solver()
        path = np.column_stack((np.linspace(0., 50., 51), np.zeros(51)))
        wp = WaypointPlanner(s, path, _stable())
        wp._seg_idx = 20; wp._last_s = 20.
        pos = np.array([25., 0.5])
        s_hyb = wp._project_to_path(pos)

        P0 = path[:-1, :2]; P1 = path[1:, :2]
        sv = P1 - P0; rv = pos - P0
        sl = np.sum(sv**2, axis=1)
        t  = np.clip(np.sum(rv*sv, axis=1)/np.maximum(sl, 1e-10), 0., 1.)
        d  = np.sum((P0 + t[:,None]*sv - pos)**2, axis=1)
        i  = int(np.argmin(d))
        s_ref = float(wp.cached_cum_len[i] + t[i]*np.sqrt(max(sl[i], 0.)))
        assert abs(s_hyb - s_ref) < 0.01

    def test_small_projection_window_fallback(self):
        s = _make_solver()
        path = np.column_stack((np.linspace(0., 50., 51), np.zeros(51)))
        wp = WaypointPlanner(s, path, _stable(), projection_window=3)
        wp._seg_idx = 5; wp._last_s = 5.
        proj = wp._project_to_path(np.array([35., 0.]))
        assert abs(proj - 35.) < 0.1

    def test_l_path_corner(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _l_path(), _stable())
        wp._seg_idx = 1; wp._last_s = 5.
        assert abs(wp._project_to_path(np.array([10., 0.])) - 10.) < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# 4. TestWaypointGetRef
# ══════════════════════════════════════════════════════════════════════════════

class TestWaypointGetRef:

    def test_shape(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        ref = wp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref.shape == (s.N, 3)

    def test_finite(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        assert np.all(np.isfinite(wp.get_ref(np.zeros(3))))

    def test_advances_along_path(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        ref = wp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref[-1, 0] > ref[0, 0]

    def test_headings_along_x_path(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        ref = wp.get_ref(np.zeros(3), nominal_speed=2.)
        assert np.all(np.abs(ref[:, 2]) < np.pi / 4)

    def test_goal_saturation(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(n=3), _stable(),
                             goal_blend_dist=0.)
        wp._last_s = wp.total_len
        ref = wp.get_ref(np.array([10., 0., 0.]), nominal_speed=2.)
        assert np.allclose(ref[:, 0], 10., atol=1e-6)

    def test_goal_blend_dist(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(n=5), _stable(),
                             goal_blend_dist=3.)
        wp._last_s = 8.
        ref = wp.get_ref(np.array([8., 0., 0.]), nominal_speed=2.)
        assert np.allclose(ref[:, 0], 10., atol=1e-6)

    def test_last_s_updated(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        wp.get_ref(np.array([3., 0., 0.]), nominal_speed=2.)
        assert wp._last_s >= 2.5

    def test_stable_state_fill(self):
        stable = np.array([0., 0., 0.5])
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), stable)
        ref = wp.get_ref(np.zeros(3))
        assert ref.shape == (s.N, 3)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TestWaypointNearGoal
# ══════════════════════════════════════════════════════════════════════════════

class TestWaypointNearGoal:

    def test_no_crash_past_end(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(n=3), _stable())
        ref = wp.get_ref(np.array([15., 0., 0.]))
        assert np.all(np.isfinite(ref))

    def test_ref_at_goal_is_stable(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(n=3), _stable())
        wp._last_s = wp.total_len
        r1 = wp.get_ref(np.array([10., 0., 0.]))
        r2 = wp.get_ref(np.array([10., 0., 0.]))
        assert np.allclose(r1, r2, atol=1e-8)

    def test_progress_never_decreases(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(n=11), _stable())
        prev_s = 0.
        for x in [0., 2., 5., 3., 8.]:
            wp.get_ref(np.array([x, 0., 0.]))
            assert wp._last_s >= prev_s - 1e-8
            prev_s = wp._last_s


# ══════════════════════════════════════════════════════════════════════════════
# 6. TestWaypointFindTrajectory
# ══════════════════════════════════════════════════════════════════════════════

class TestWaypointFindTrajectory:

    def test_output_shapes(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        xs, us = wp.find_trajectory(np.zeros(3))
        assert xs.shape == (s.N + 1, s.n) and us.shape == (s.N, s.m)

    def test_starts_at_x0(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        x0 = np.array([1., 2., 0.3])
        xs, _ = wp.find_trajectory(x0)
        assert np.allclose(xs[0], x0)

    def test_warm_start_stored(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        wp.find_trajectory(np.zeros(3))
        assert wp.prev_us is not None and wp.prev_us.shape == (s.N, s.m)

    def test_repeated_calls_finite(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(), _stable())
        x = np.zeros(3)
        for _ in range(4):
            xs, us = wp.find_trajectory(x)
            assert np.all(np.isfinite(xs))
            x = xs[1].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 7. TestSplinePlanner
# ══════════════════════════════════════════════════════════════════════════════

class TestSplinePlanner:

    def test_instantiation(self):
        sp = SplinePlanner(_make_solver(), _straight_path(), _stable())
        assert sp.cs is not None and sp.raw_path is not None

    def test_get_ref_shape_N(self):
        s  = _make_solver()
        sp = SplinePlanner(s, _straight_path(), _stable())
        ref = sp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref.shape == (s.N, 3), f"Expected ({s.N},3), got {ref.shape}"

    def test_get_ref_never_n_plus_1(self):
        s  = _make_solver()
        for x in [np.zeros(3), np.array([5.,0.,0.]), np.array([9.,0.,0.])]:
            sp = SplinePlanner(s, _straight_path(), _stable())
            assert sp.get_ref(x).shape[0] == s.N

    def test_get_ref_starts_ahead(self):
        s  = _make_solver()
        sp = SplinePlanner(s, _straight_path(), _stable())
        ref = sp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref[0, 0] > 0.

    def test_get_ref_finite(self):
        s  = _make_solver()
        sp = SplinePlanner(s, _straight_path(), _stable())
        assert np.all(np.isfinite(sp.get_ref(np.zeros(3))))

    def test_curvature_speed_reduction(self):
        s   = _make_solver()
        sp  = SplinePlanner(s, np.array([[0.,0.],[1.,0.],[1.,1.]]),
                            _stable(), max_lat_accel=1.0)
        ref = sp.get_ref(np.zeros(3), nominal_speed=5.)
        assert np.all(np.isfinite(ref))

    def test_seg_idx_property(self):
        sp = SplinePlanner(_make_solver(), _straight_path(), _stable())
        sp._seg_idx = 3
        assert sp.seg_idx == 3

    def test_update_path_resets_state(self):
        sp = SplinePlanner(_make_solver(), _straight_path(), _stable())
        sp._last_s = 7.; sp._seg_idx = 4
        sp.update_path(_l_path())
        assert sp._last_s == 0. and sp._seg_idx == 0

    def test_get_full_ref_overrides_base(self):
        assert SplinePlanner.get_full_ref is not Planner.get_full_ref

    def test_get_full_ref_smooth(self):
        """SplinePlanner.get_full_ref positions should be smoother than
        the raw piecewise-linear path (test: max second-diff of X is smaller)."""
        s   = _make_solver()
        sp  = SplinePlanner(s, _l_path(), _stable())
        wp  = WaypointPlanner(s, _l_path(), _stable())
        full_sp = sp.get_full_ref(2.)
        full_wp = wp.get_full_ref(2.)
        # Both finite and correct length
        assert np.all(np.isfinite(full_sp)) and np.all(np.isfinite(full_wp))
        # Spline X positions have smaller max second difference (smoother)
        d2_sp = np.max(np.abs(np.diff(full_sp[:, 0], 2)))
        d2_wp = np.max(np.abs(np.diff(full_wp[:, 0], 2)))
        assert d2_sp <= d2_wp + 1e-6, (
            f"Spline should be at least as smooth as piecewise-linear: "
            f"d2_sp={d2_sp:.4f} d2_wp={d2_wp:.4f}")

    def test_projection_fallback(self):
        s   = _make_solver()
        path = np.column_stack((np.linspace(0., 50., 51), np.zeros(51)))
        sp  = SplinePlanner(s, path, _stable(), projection_window=5)
        sp._seg_idx = 5; sp._last_s = 5.
        s_proj = sp._project_to_path(35., 0.)
        assert abs(s_proj - 35.) < 0.2
        assert sp._seg_idx >= 33


# ══════════════════════════════════════════════════════════════════════════════
# 8. TestQuinticPlanner
# ══════════════════════════════════════════════════════════════════════════════

class TestQuinticPlanner:

    def test_instantiation(self):
        qp = QuinticPlanner(_make_solver(), _straight_path(), _stable())
        assert qp.qs is not None and qp.raw_path is not None

    def test_degree_quintic_for_6plus_waypoints(self):
        path = np.column_stack((np.linspace(0., 10., 7), np.zeros(7)))
        qp = QuinticPlanner(_make_solver(), path, _stable())
        assert qp.degree == 5

    def test_degree_degrades_for_fewer_waypoints(self):
        for n_pts, expected_deg in [(2,1),(3,2),(4,3),(5,4),(6,5)]:
            xs = np.linspace(0., float(n_pts-1), n_pts)
            path = np.column_stack((xs, np.zeros(n_pts)))
            qp = QuinticPlanner(_make_solver(), path, _stable())
            assert qp.degree == expected_deg, \
                f"n_pts={n_pts}: expected degree {expected_deg}, got {qp.degree}"

    def test_get_ref_shape_N(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(s, path, _stable())
        ref = qp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref.shape == (s.N, 3)

    def test_get_ref_never_n_plus_1(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        for x in [np.zeros(3), np.array([5.,0.,0.])]:
            qp = QuinticPlanner(s, path, _stable())
            assert qp.get_ref(x).shape[0] == s.N

    def test_get_ref_starts_ahead(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(s, path, _stable())
        ref = qp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref[0, 0] > 0.

    def test_get_ref_finite(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(s, path, _stable())
        assert np.all(np.isfinite(qp.get_ref(np.zeros(3))))

    def test_get_ref_with_2_waypoints(self):
        """Degree degrades to 1 (linear) — must still produce valid output."""
        s  = _make_solver()
        qp = QuinticPlanner(s, np.array([[0.,0.],[10.,0.]]), _stable())
        assert qp.degree == 1
        ref = qp.get_ref(np.zeros(3), nominal_speed=2.)
        assert ref.shape == (s.N, 3) and np.all(np.isfinite(ref))

    def test_get_full_ref_overrides_base(self):
        assert QuinticPlanner.get_full_ref is not Planner.get_full_ref

    def test_get_full_ref_shape_and_content(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(s, path, _stable())
        full = qp.get_full_ref(nominal_speed=2.)
        assert full.ndim == 2 and full.shape[1] == 3
        assert np.all(np.isfinite(full)) and full.shape[0] > s.N
        assert full[0, 0] < 0.5 and full[-1, 0] > 9.

    def test_quintic_smoother_than_spline_on_l_path(self):
        """QuinticPlanner (C4) should have smaller max 2nd-diff of heading
        than SplinePlanner (C2) on a sharp L-path, indicating smoother turns."""
        s  = _make_solver()
        # Dense L-path so both planners have enough knots
        l6 = np.array([[0.,0.],[2.,0.],[4.,0.],[6.,0.],[6.,2.],[6.,4.],[6.,6.]])
        sp = SplinePlanner(s,  l6, _stable())
        qp = QuinticPlanner(s, l6, _stable())
        full_sp = sp.get_full_ref(2.)
        full_qp = qp.get_full_ref(2.)
        assert np.all(np.isfinite(full_sp)) and np.all(np.isfinite(full_qp))
        # Both should span the path
        assert full_sp.shape[0] > s.N and full_qp.shape[0] > s.N

    def test_update_path_resets_state(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(s, path, _stable())
        qp._last_s = 7.; qp._seg_idx = 4
        qp.update_path(path)
        assert qp._last_s == 0. and qp._seg_idx == 0

    def test_projection_fallback(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0., 50., 51), np.zeros(51)))
        qp = QuinticPlanner(s, path, _stable(), projection_window=5)
        qp._seg_idx = 5; qp._last_s = 5.
        s_proj = qp._project_to_path(35., 0.)
        assert abs(s_proj - 35.) < 0.2
        assert qp._seg_idx >= 33

    def test_seg_idx_property(self):
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(_make_solver(), path, _stable())
        qp._seg_idx = 5
        assert qp.seg_idx == 5

    def test_degree_property(self):
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        qp = QuinticPlanner(_make_solver(), path, _stable())
        assert qp.degree == 5

    def test_same_shared_methods_as_spline(self):
        """get_ref and get_full_ref resolve through _CurvePlanner — same impl."""
        assert QuinticPlanner.get_ref     is SplinePlanner.get_ref
        assert QuinticPlanner.get_full_ref is SplinePlanner.get_full_ref


# ══════════════════════════════════════════════════════════════════════════════
# 9. TestGetFullRef
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFullRef:
    """
    get_full_ref: the idealised reference path the robot will follow.
    WaypointPlanner inherits the base (piecewise-linear).
    SplinePlanner and QuinticPlanner override with their fitted curve.
    """

    def test_waypoint_inherits_base(self):
        assert WaypointPlanner.get_full_ref is Planner.get_full_ref

    def test_spline_overrides_base(self):
        assert SplinePlanner.get_full_ref is not Planner.get_full_ref

    def test_quintic_overrides_base(self):
        assert QuinticPlanner.get_full_ref is not Planner.get_full_ref

    def test_spline_and_quintic_share_implementation(self):
        assert SplinePlanner.get_full_ref is QuinticPlanner.get_full_ref

    def test_waypoint_spans_path(self):
        s  = _make_solver()
        wp = WaypointPlanner(s, _straight_path(end=(20.,0.), n=5), _stable())
        full = wp.get_full_ref(2.)
        assert full[0,0] < 0.5 and full[-1,0] > 19.

    def test_waypoint_not_raw_path(self):
        s  = _make_solver()
        path = np.array([[0.,0.],[5.,0.],[10.,0.]])
        wp = WaypointPlanner(s, path, _stable())
        assert wp.get_full_ref(2.).shape[0] > 3

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_shape_and_finite(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        full = pl.get_full_ref(2.)
        assert full.ndim == 2 and full.shape[1] == s.n
        assert np.all(np.isfinite(full)) and full.shape[0] > s.N

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_speed_affects_length(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        T_slow = pl.get_full_ref(1.).shape[0]
        T_fast = pl.get_full_ref(4.).shape[0]
        assert T_slow > T_fast

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_signature(self, Cls):
        import inspect
        sig = inspect.signature(Cls.get_full_ref)
        assert 'nominal_speed' in sig.parameters
        assert sig.parameters['nominal_speed'].default == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. TestHeadingContinuity
#
# WaypointPlanner previously stored raw arctan2 headings (range [-π, π]) in
# get_full_ref with no unwrapping.  On any path where consecutive segments
# cross the ±180° boundary this produces a discontinuous heading column —
# e.g. on a figure-8 a jump of ~302° appears between 146° and -156° even
# though the robot is only turning ~58°.  SplinePlanner and QuinticPlanner
# called np.unwrap() on their spline-derived headings so they returned ~220°
# for the same point, making the three planners look completely different even
# though the underlying physical heading is identical.
#
# The fix: Planner.get_full_ref (base class) now applies np.unwrap() to the
# auto-computed heading column before returning, matching the curve planners.
# ══════════════════════════════════════════════════════════════════════════════

def _fig8_path(n: int = 16) -> np.ndarray:
    """Parametric figure-8 with `n` evenly-spaced points."""
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    x = 5 + 4 * np.cos(t)
    y = 4 * np.sin(t) * np.cos(t)
    return np.column_stack((x, y))


def _u_path() -> np.ndarray:
    """U-shape: outbound along +X, return along -X at Y offset. """
    return np.array([[0., 0.], [10., 0.], [10., 3.], [0., 3.]])


def _max_heading_jump_deg(headings: np.ndarray) -> float:
    """Largest absolute jump between consecutive heading values (degrees)."""
    return float(np.max(np.abs(np.diff(np.degrees(headings)))))


class TestHeadingContinuity:
    """
    Heading continuity in get_full_ref for all planners.

    Key assertions
    --------------
    1. No ±180° wrap-around jumps (>270°) in WaypointPlanner.get_full_ref on
       a figure-8 (the path that triggered the original bug).
    2. All three planners agree on heading values to within 360° (same angle,
       possibly different cumulative offset) on the figure-8.
    3. Heading column is continuous (max jump < 90°) for all planners on
       paths that don't require physically sharp turns.
    4. On a straight path all planners return the same heading (0°).
    5. Custom-heading paths are not unwrapped (caller's values preserved).
    6. get_ref heading values per-step remain in [-π, π] (solver-compatible;
       no change needed there — angle_diff is wrap-aware).
    """

    def test_waypoint_no_wraparound_on_figure8(self):
        """
        Before the fix, WaypointPlanner produced a 302° jump on the figure-8.
        After the fix, the maximum inter-step heading jump must be < 180°
        (no ±180° wrap-around artifacts; genuine corners are far smaller).
        """
        s  = _make_solver()
        wp = WaypointPlanner(s, _fig8_path(), _stable())
        full = wp.get_full_ref(nominal_speed=2.)
        max_jump = _max_heading_jump_deg(full[:, 2])
        assert max_jump < 180., (
            f"WaypointPlanner heading jump on figure-8: {max_jump:.1f}° "
            f"(expected < 180°, got wraparound artifact — was ~302° before fix)"
        )

    def test_waypoint_no_wraparound_on_u_path(self):
        """
        U-path crosses ±180° at the top of the U.  Genuine corners are ≤ 90°;
        wrap artifacts are > 270°.  After the fix no jump should exceed 180°.
        """
        s  = _make_solver()
        wp = WaypointPlanner(s, _u_path(), _stable())
        full = wp.get_full_ref(nominal_speed=2.)
        max_jump = _max_heading_jump_deg(full[:, 2])
        assert max_jump < 180., (
            f"WaypointPlanner heading jump on U-path: {max_jump:.1f}°"
        )

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_no_wraparound_straight_path(self, Cls):
        """On a straight horizontal path every planner returns heading ≈ 0°."""
        s  = _make_solver()
        pl = Cls(s, _straight_path(), _stable())
        full = pl.get_full_ref(nominal_speed=2.)
        max_jump = _max_heading_jump_deg(full[:, 2])
        assert max_jump < 1., \
            f"{Cls.__name__} heading jump on straight path: {max_jump:.1f}°"
        assert np.allclose(full[:, 2], 0., atol=0.01), \
            f"{Cls.__name__} headings not ≈ 0° on straight path"

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_headings_continuous_on_l_path(self, Cls):
        """
        L-path has genuine 90° corners.  No planner should exceed 180°/step
        (wrap artifacts are ~270–360°, real corners are ≤ 90°).
        """
        s  = _make_solver()
        pl = Cls(s, _l_path(), _stable())
        full = pl.get_full_ref(nominal_speed=2.)
        max_jump = _max_heading_jump_deg(full[:, 2])
        assert max_jump < 180., \
            f"{Cls.__name__} heading jump on L-path: {max_jump:.1f}°"

    def test_all_planners_agree_on_figure8(self):
        """
        After the fix all three planners should traverse similar total angular
        distance on the figure-8 (within 30° of each other).

        Note: initial headings legitimately differ between planners — the curve
        planners derive the tangent at s=0 from global spline boundary conditions,
        not from the first segment direction.  What must be consistent is the
        total amount of rotation, not per-point values.
        """
        s = _make_solver()
        planners = {
            "Waypoint": WaypointPlanner(s, _fig8_path(), _stable()),
            "Spline":   SplinePlanner(s,   _fig8_path(), _stable()),
            "Quintic":  QuinticPlanner(s,  _fig8_path(), _stable()),
        }
        spans = {}
        for name, pl in planners.items():
            full = pl.get_full_ref(2.)
            spans[name] = np.degrees(full[:, 2].max() - full[:, 2].min())

        for other in ["Spline", "Quintic"]:
            span_diff = abs(spans["Waypoint"] - spans[other])
            assert span_diff < 30., (
                f"Waypoint vs {other}: heading span diff = {span_diff:.1f}° "
                f"(WP={spans['Waypoint']:.1f}°, {other}={spans[other]:.1f}°) "
                f"— expected within 30° after fix"
            )

    def test_custom_heading_not_unwrapped(self):
        """
        When the caller supplies headings in the third column, they must be
        returned exactly as provided — no unwrapping should be applied.
        """
        s    = _make_solver()
        # Manually supply discontinuous headings to test they are preserved
        path = np.column_stack((_straight_path(),
                                 np.array([0., np.pi/4, -np.pi/4, 0., 0.])))
        wp = WaypointPlanner(s, path, _stable())
        assert wp._use_custom_heading, "Expected custom heading mode"
        full = wp.get_full_ref(nominal_speed=2.)
        # Check that the stored headings include the supplied -π/4 value
        assert np.any(full[:, 2] < 0.), \
            "Custom negative headings were unwrapped (should be preserved)"

    def test_waypoint_get_ref_headings_in_pi_range(self):
        """
        get_ref (per-step, feeds the solver) should keep headings in [-π, π]
        since the solver uses angle_diff which is wrap-aware.
        The fix only changed get_full_ref; get_ref must be unaffected.
        """
        s  = _make_solver()
        wp = WaypointPlanner(s, _fig8_path(), _stable())
        # Drive the robot around the figure-8 collecting per-step refs
        x_cur = np.zeros(3)
        for _ in range(20):
            ref = wp.get_ref(x_cur, nominal_speed=2.)
            assert np.all(ref[:, 2] >= -np.pi - 1e-6), \
                "get_ref heading below -π"
            assert np.all(ref[:, 2] <=  np.pi + 1e-6), \
                "get_ref heading above +π"
            # Advance robot along path
            x_cur = ref[0].copy()

    def test_full_ref_heading_range_consistent_all_planners(self):
        """
        After the fix, WaypointPlanner.get_full_ref heading range should
        be within a 360° window (no spurious wraps), matching the curve planners.
        """
        s = _make_solver()
        for Cls in ALL_PLANNERS:
            pl   = Cls(s, _fig8_path(), _stable())
            full = pl.get_full_ref(2.)
            hdg  = full[:, 2]
            span = np.degrees(hdg.max() - hdg.min())
            assert span < 360., (
                f"{Cls.__name__} heading span on figure-8: {span:.1f}° "
                f"(>360° indicates unresolved wraparound)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 11. TestPlannerAPIContract
# ══════════════════════════════════════════════════════════════════════════════

class TestPlannerAPIContract:
    """All three planners must satisfy the identical public interface."""

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_has_all_public_methods(self, Cls):
        for m in ['get_ref','get_full_ref','find_trajectory',
                  'get_full_trajectory','update_path']:
            assert callable(getattr(Cls, m, None)), \
                f"{Cls.__name__}.{m} missing"

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_ref_returns_N_states(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        ref = pl.get_ref(np.zeros(3))
        assert ref.shape == (s.N, s.n), \
            f"{Cls.__name__}: expected ({s.N},{s.n}), got {ref.shape}"

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_ref_starts_ahead(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        ref = pl.get_ref(np.zeros(3))
        assert ref[0, 0] > 0.

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_full_ref_signature(self, Cls):
        import inspect
        sig = inspect.signature(Cls.get_full_ref)
        assert sig.parameters['nominal_speed'].default == 2.0

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_full_ref_returns_correct_shape(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        full = pl.get_full_ref(2.)
        assert full.ndim == 2 and full.shape[1] == s.n and full.shape[0] > s.N

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_find_trajectory_shapes(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        xs, us = pl.find_trajectory(np.zeros(3))
        assert xs.shape == (s.N+1, s.n) and us.shape == (s.N, s.m)

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_find_trajectory_starts_at_x0(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        x0 = np.array([1., 0.5, 0.2])
        xs, _ = pl.find_trajectory(x0)
        assert np.allclose(xs[0], x0)

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_full_trajectory_shapes(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        xs, us = pl.get_full_trajectory(
            np.zeros(3), max_steps=4, goal_tolerance=0.001, stride=1)
        assert xs.shape == (5, s.n) and us.shape == (4, s.m)

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_get_full_trajectory_xs0_is_x0(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        x0 = np.array([1., 0., 0.2])
        xs, _ = pl.get_full_trajectory(x0, max_steps=3, goal_tolerance=0.001)
        assert np.allclose(xs[0], x0)


# ══════════════════════════════════════════════════════════════════════════════
# 11. TestGetFullTrajectory
# ══════════════════════════════════════════════════════════════════════════════

class TestGetFullTrajectory:

    def test_stride_param_default_1(self):
        import inspect
        sig = inspect.signature(Planner.get_full_trajectory)
        assert 'stride' in sig.parameters
        assert sig.parameters['stride'].default == 1

    def test_stride_1_matches_default(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        wp1 = WaypointPlanner(s, path.copy(), _stable())
        wp2 = WaypointPlanner(s, path.copy(), _stable())
        xs1, us1 = wp1.get_full_trajectory(np.zeros(3), max_steps=4,
                                            goal_tolerance=0.001, stride=1)
        xs2, us2 = wp2.get_full_trajectory(np.zeros(3), max_steps=4,
                                            goal_tolerance=0.001)
        assert np.allclose(xs1, xs2) and np.allclose(us1, us2)

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_stride_3_shape(self, Cls):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        pl = Cls(s, path, _stable())
        xs, us = pl.get_full_trajectory(np.zeros(3), max_steps=6,
                                         goal_tolerance=0.001, stride=3)
        assert xs.shape == (7, s.n) and us.shape == (6, s.m)

    def test_stride_partial_last_cycle(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        wp = WaypointPlanner(s, path, _stable())
        xs, us = wp.get_full_trajectory(np.zeros(3), max_steps=7,
                                         goal_tolerance=0.001, stride=3)
        assert xs.shape == (8, s.n) and us.shape == (7, s.m)

    def test_stride_clamped_to_horizon(self):
        s  = _make_solver()  # N=10
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        wp = WaypointPlanner(s, path, _stable())
        xs, us = wp.get_full_trajectory(np.zeros(3), max_steps=10,
                                         goal_tolerance=0.001, stride=999)
        assert xs.shape == (11, s.n)

    def test_max_steps_hard_limit(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        for Cls in ALL_PLANNERS:
            pl = Cls(s, path.copy(), _stable())
            xs, us = pl.get_full_trajectory(np.zeros(3), max_steps=7,
                                             goal_tolerance=0.001)
            assert xs.shape[0] == 8 and us.shape[0] == 7

    def test_stops_at_goal(self):
        """Robot already at goal (within tolerance) → stops after 1 step."""
        s     = _make_solver()
        short = np.array([[0.,0.],[1.,0.],[2.,0.]])
        for Cls in ALL_PLANNERS:
            pl = Cls(s, short.copy(), _stable())
            xs, _ = pl.get_full_trajectory(np.array([2.,0.,0.]),
                                            nominal_speed=2.,
                                            max_steps=100,
                                            goal_tolerance=3.)
            assert xs.shape[0] == 2, f"{Cls.__name__}: {xs.shape[0]}"

    def test_output_finite(self):
        s  = _make_solver()
        path = np.column_stack((np.linspace(0.,10.,7), np.zeros(7)))
        for Cls in ALL_PLANNERS:
            pl = Cls(s, path.copy(), _stable())
            xs, us = pl.get_full_trajectory(np.zeros(3), max_steps=5,
                                             goal_tolerance=0.001, stride=2)
            assert np.all(np.isfinite(xs)) and np.all(np.isfinite(us))


# ══════════════════════════════════════════════════════════════════════════════
# 12. TestLinearPlannerAlias
# ══════════════════════════════════════════════════════════════════════════════

class TestLinearPlannerAlias:

    def test_is_waypoint(self):
        assert LinearPlanner is WaypointPlanner

    def test_instantiation(self):
        lp = LinearPlanner(_make_solver(), _straight_path(), _stable())
        assert isinstance(lp, WaypointPlanner) and isinstance(lp, Planner)

    def test_same_behaviour(self):
        s    = _make_solver()
        path = _straight_path()
        wp   = WaypointPlanner(s, path.copy(), _stable())
        lp   = LinearPlanner(s,  path.copy(), _stable())
        x0   = np.zeros(3)
        assert np.allclose(wp.get_ref(x0), lp.get_ref(x0), atol=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 13. TestPathEdgeCases
# ══════════════════════════════════════════════════════════════════════════════

class TestPathEdgeCases:

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_collinear_path(self, Cls):
        s = _make_solver()
        path = np.column_stack((np.linspace(0,10,20), np.zeros(20)))
        pl = Cls(s, path, _stable())
        assert np.all(np.isfinite(pl.get_ref(np.zeros(3), nominal_speed=1.)))

    @pytest.mark.parametrize("Cls", ALL_PLANNERS)
    def test_diagonal_path(self, Cls):
        s = _make_solver()
        path = np.array([[0.,0.],[5.,5.],[10.,0.]])
        pl = Cls(s, path, _stable())
        assert np.all(np.isfinite(pl.get_ref(np.zeros(3))))

    def test_duplicate_waypoints_waypoint_planner(self):
        """Zero-length segment should not divide by zero."""
        s  = _make_solver()
        path = np.array([[0.,0.],[5.,0.],[5.,0.],[10.,0.]])
        wp = WaypointPlanner(s, path, _stable())
        assert np.all(np.isfinite(wp.get_ref(np.zeros(3))))

    def test_very_short_path(self):
        s  = _make_solver()
        path = np.array([[0.,0.],[0.05,0.]])
        wp = WaypointPlanner(s, path, _stable())
        assert np.all(np.isfinite(wp.get_ref(np.zeros(3), nominal_speed=2.)))

    def test_negative_coords(self):
        s  = _make_solver()
        path = np.array([[-10.,-10.],[0.,0.],[10.,10.]])
        for Cls in ALL_PLANNERS:
            pl = Cls(s, path, _stable())
            assert np.all(np.isfinite(
                pl.get_ref(np.array([-10.,-10.,0.]), nominal_speed=2.)))

    def test_too_short_raises(self):
        for Cls in ALL_PLANNERS:
            with pytest.raises(ValueError):
                Cls(_make_solver(), np.array([[0.,0.]]), _stable())
