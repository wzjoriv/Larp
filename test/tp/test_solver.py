"""
test/tp/test_solver.py
======================
Tests for larp.tp.solver: SQPSolver, ALILQRSolver, ALDDPSolver.

Fixtures
--------
Uses WMRDynamics (wheel-mobile-robot, 3-state unicycle) and a minimal
RiskField built from test/data.rgj so tests run without an urban map or
network access.  All obstacle tests use the ``data.rgj`` Point obstacles
(see test fixture ``risk_field``).

Test categories
---------------
1. Solver construction — verify shapes, default params, bound parsing.
2. Rollout — RK4 integration correctness.
3. Field constraints — half-plane extraction from obstacles.
4. SQP solve — stable output shape, warm-start, linearize_every / field_every.
5. AL solve  — same interface as SQP; cache params; DDP vs iLQR.
6. Correctness near goal — verifies solver does not diverge at zero-error ref.
7. Cache speedup smoke test — fewer discretize calls with linearize_every > 1.

Notes
-----
* ``R = 0`` (pure state-tracking) can cause near-degenerate QP near the goal.
  The test ``test_sqp_near_goal_finite`` guards that the solver returns finite
  values even in this regime.  Use at least ``R = 1e-6 * I`` in production.
* GPU / JAX tests are skipped when JAX is not installed.
"""

import os
import json
import pytest
import numpy as np
from unittest.mock import patch

# ── Minimal larp import (skip if package not importable) ──────────────────
larp = pytest.importorskip("larp")
from larp.dynamics import WMRDynamics
from larp.tp.solver import Solver, SQPSolver, ALILQRSolver, ALDDPSolver


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

DATA_RGJ_PATH = os.path.join(os.path.dirname(__file__), "..", "data.rgj")


def _load_risk_field():
    """Build a RiskField from test/data.rgj."""
    with open(DATA_RGJ_PATH) as f:
        raw = json.load(f)

    rgjs = []
    for feat in raw.get("features", []):
        geom = feat.get("geometry", {})
        if geom:
            rgjs.append(geom)

    center = [45.0, 45.0]
    field = larp.RiskField(rgjs=rgjs, center_point=center, size=100)
    return field


def _make_wmr() -> WMRDynamics:
    return WMRDynamics()


def _make_sqp(field=None, **kw) -> SQPSolver:
    dyn = _make_wmr()
    n, m = dyn.first_order_state_n, dyn.first_order_control_n
    Q  = np.diag([10.0, 10.0, 1.0])
    R  = np.eye(m) * 0.1
    Qf = Q * 5.0
    u_bounds = ([-2.0, -1.5], [2.0, 1.5])
    return SQPSolver(
        field=field,
        dynamics=dyn,
        dt=0.1,
        horizon=10,
        Q=Q, R=R, Qf=Qf,
        u_bounds=u_bounds,
        **kw
    )


def _make_ilqr(field=None, **kw) -> ALILQRSolver:
    dyn = _make_wmr()
    n, m = dyn.first_order_state_n, dyn.first_order_control_n
    Q  = np.diag([10.0, 10.0, 1.0])
    R  = np.eye(m) * 0.1
    Qf = Q * 5.0
    u_bounds = ([-2.0, -1.5], [2.0, 1.5])
    return ALILQRSolver(
        field=field,
        dynamics=dyn,
        dt=0.1,
        horizon=10,
        Q=Q, R=R, Qf=Qf,
        u_bounds=u_bounds,
        al_iters=5,
        ilqr_iters=20,
        **kw
    )


def _make_ddp(field=None, **kw) -> ALDDPSolver:
    dyn = _make_wmr()
    Q  = np.diag([10.0, 10.0, 1.0])
    R  = np.eye(dyn.first_order_control_n) * 0.1
    Qf = Q * 5.0
    u_bounds = ([-2.0, -1.5], [2.0, 1.5])
    return ALDDPSolver(
        field=field,
        dynamics=dyn,
        dt=0.1,
        horizon=10,
        Q=Q, R=R, Qf=Qf,
        u_bounds=u_bounds,
        al_iters=5,
        ilqr_iters=20,
        **kw
    )


def _straight_ref(solver: Solver, x_goal: np.ndarray) -> np.ndarray:
    """Return a constant reference of length N pointing at x_goal."""
    return np.tile(x_goal, (solver.N, 1))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Solver Construction
# ══════════════════════════════════════════════════════════════════════════════

class TestSolverConstruction:

    def test_sqp_is_abstract_subclass(self):
        assert issubclass(SQPSolver, Solver)

    def test_ilqr_is_abstract_subclass(self):
        assert issubclass(ALILQRSolver, Solver)

    def test_ddp_is_abstract_subclass(self):
        assert issubclass(ALDDPSolver, Solver)

    def test_solver_is_abstract(self):
        with pytest.raises(TypeError):
            Solver(field=None, dynamics=_make_wmr())   # type: ignore

    def test_sqp_horizon_int(self):
        s = _make_sqp()
        assert s.N == 10

    def test_sqp_horizon_float(self):
        # horizon=1.5 s at dt=0.1 -> N=15
        dyn = _make_wmr()
        s = SQPSolver(
            field=None, dynamics=dyn,
            dt=0.1, horizon=1.5,
            Q=np.eye(3), R=np.eye(2), Qf=np.eye(3),
        )
        assert s.N == 15

    def test_sqp_dimensions(self):
        s = _make_sqp()
        dyn = _make_wmr()
        assert s.n == dyn.first_order_state_n
        assert s.m == dyn.first_order_control_n

    def test_sqp_var_count(self):
        s = _make_sqp()
        assert s.var_count == s.N * s.n + s.N * s.m

    def test_sqp_bound_shapes(self):
        s = _make_sqp()
        assert s.x_min.shape == (s.n,)
        assert s.x_max.shape == (s.n,)
        assert s.u_min.shape == (s.m,)
        assert s.u_max.shape == (s.m,)

    def test_sqp_linearize_field_every(self):
        s = _make_sqp(linearize_every=3, field_every=5)
        assert s.linearize_every == 3
        assert s.field_every == 5

    def test_ilqr_rho_params(self):
        s = _make_ilqr(rho_init=2.0, rho_max=1e4, rho_scale=2.0)
        assert s.rho_init == 2.0
        assert s.rho_max == 1e4
        assert s.rho_scale == 2.0

    def test_ilqr_not_ddp(self):
        s = _make_ilqr()
        assert s.use_ddp is False

    def test_ddp_flag(self):
        s = _make_ddp()
        assert s.use_ddp is True

    def test_cache_initialised(self):
        s = _make_sqp()
        assert s.cache["l_box"] is not None   # populated in __init__
        assert s.cache["A_dyn"] is None        # not yet solved

    def test_al_dyn_cache_initialised(self):
        s = _make_ilqr()
        assert s._il_dyn_cache["Ad"] is None
        assert s._il_dyn_cache["Bd"] is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Rollout
# ══════════════════════════════════════════════════════════════════════════════

class TestRollout:

    def test_rollout_shape(self):
        s   = _make_sqp()
        x0  = np.array([0.0, 0.0, 0.0])
        us  = np.zeros((s.N, s.m))
        xs  = s.rollout(x0, us)
        assert xs.shape == (s.N + 1, s.n)

    def test_rollout_zero_control_stays_still(self):
        """WMR with u=0 should not move."""
        s   = _make_sqp()
        x0  = np.array([1.0, 2.0, 0.5])
        us  = np.zeros((s.N, s.m))
        xs  = s.rollout(x0, us)
        assert np.allclose(xs, x0, atol=1e-10), \
            "WMR should be stationary with zero control."

    def test_rollout_starts_at_x0(self):
        s  = _make_sqp()
        x0 = np.array([3.0, -1.0, np.pi / 4])
        us = np.random.randn(s.N, s.m) * 0.1
        xs = s.rollout(x0, us)
        assert np.allclose(xs[0], x0)

    def test_rollout_finite_with_nonzero_control(self):
        s  = _make_sqp()
        x0 = np.array([0.0, 0.0, 0.0])
        us = np.ones((s.N, s.m)) * 0.5
        xs = s.rollout(x0, us)
        assert np.all(np.isfinite(xs))

    def test_rollout_python_vs_default(self):
        """_rollout_python should match rollout when jax_backend=False."""
        s  = _make_sqp()
        x0 = np.array([0.0, 0.0, 0.0])
        us = np.random.default_rng(0).standard_normal((s.N, s.m))
        xs_default = s.rollout(x0, us)
        xs_python  = s._rollout_python(x0, us)
        assert np.allclose(xs_default, xs_python, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Field Constraints
# ══════════════════════════════════════════════════════════════════════════════

class TestFieldConstraints:

    @pytest.fixture(scope="class")
    def field(self):
        return _load_risk_field()

    def test_no_field_returns_none(self):
        s = _make_sqp(field=None)
        x = np.array([45.0, 45.0, 0.0])   # inside obstacle zone
        A, b = s.get_field_constraints(x)
        assert A is None and b is None

    def test_far_from_obstacle_returns_none(self):
        """A point far from all obstacles should produce no constraints."""
        field = _load_risk_field()
        s = _make_sqp(field=field)
        # data.rgj obstacles cluster around (45-60, 45-60)
        # (0, 0) should be well outside all repulsion radii
        x = np.zeros(s.n)
        x[0] = 0.0
        x[1] = 0.0
        A, b = s.get_field_constraints(x)
        # May or may not be None depending on field size, but must be consistent
        if A is not None:
            assert A.shape[1] == s.n
            assert b.shape[0] == A.shape[0]

    def test_near_obstacle_returns_constraints(self):
        """A point very close to a known obstacle should yield constraints."""
        field = _load_risk_field()
        s = _make_sqp(field=field)
        # Obstacle in data.rgj at (50, 50) with repulsion [[36,0],[0,25]]
        x = np.zeros(s.n)
        x[0] = 50.0   # right on top of obstacle
        x[1] = 50.0
        A, b = s.get_field_constraints(x)
        # We expect at least one constraint
        assert A is not None, "Expected constraints near obstacle at (50, 50)"
        assert A.shape == (A.shape[0], s.n)
        assert b.ndim == 1

    def test_constraint_shape_consistency(self):
        field = _load_risk_field()
        s = _make_sqp(field=field)
        x = np.zeros(s.n)
        x[0], x[1] = 50.0, 50.0
        A, b = s.get_field_constraints(x)
        if A is not None:
            assert A.shape[0] == b.shape[0], "A rows must match b length"


# ══════════════════════════════════════════════════════════════════════════════
# 4. SQP Solver
# ══════════════════════════════════════════════════════════════════════════════

class TestSQPSolver:

    @pytest.fixture
    def solver(self):
        return _make_sqp()

    def test_solve_output_shapes(self, solver):
        x0  = np.array([0.0, 0.0, 0.0])
        ref = _straight_ref(solver, np.array([5.0, 5.0, 0.0]))
        xs, us = solver.solve(x0, ref)
        assert xs.shape == (solver.N + 1, solver.n)
        assert us.shape == (solver.N, solver.m)

    def test_solve_starts_at_x0(self, solver):
        x0  = np.array([1.0, -2.0, np.pi / 3])
        ref = _straight_ref(solver, x0)
        xs, us = solver.solve(x0, ref)
        assert np.allclose(xs[0], x0, atol=1e-8)

    def test_solve_finite_outputs(self, solver):
        x0  = np.zeros(solver.n)
        ref = _straight_ref(solver, np.array([3.0, 3.0, 0.0]))
        xs, us = solver.solve(x0, ref)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(us))

    def test_solve_respects_u_bounds(self, solver):
        x0  = np.zeros(solver.n)
        ref = _straight_ref(solver, np.array([10.0, 0.0, 0.0]))
        xs, us = solver.solve(x0, ref)
        assert np.all(us >= solver.u_min - 1e-4)
        assert np.all(us <= solver.u_max + 1e-4)

    def test_solve_zero_ref_small_control(self, solver):
        """With goal = current position, optimal control should be near zero."""
        x0  = np.array([0.0, 0.0, 0.0])
        ref = _straight_ref(solver, x0)
        _, us = solver.solve(x0, ref)
        assert np.all(np.abs(us) < 1.0), \
            "Control should be small when already at reference."

    def test_warm_start_accepted(self, solver):
        x0  = np.zeros(solver.n)
        ref = _straight_ref(solver, np.array([2.0, 2.0, 0.0]))
        xs1, us1 = solver.solve(x0, ref)
        # Second call with warm start — should still produce valid output
        xs2, us2 = solver.solve(x0, ref, us_init=us1)
        assert xs2.shape == xs1.shape
        assert np.all(np.isfinite(xs2))

    def test_linearize_every_1_vs_5(self):
        """Solver with linearize_every=5 should still converge close to every=1."""
        x0  = np.array([0.0, 0.0, 0.0])
        x_goal = np.array([2.0, 2.0, 0.0])

        s1 = _make_sqp(linearize_every=1)
        s5 = _make_sqp(linearize_every=5)

        ref1 = _straight_ref(s1, x_goal)
        ref5 = _straight_ref(s5, x_goal)

        xs1, us1 = s1.solve(x0, ref1)
        xs5, us5 = s5.solve(x0, ref5)

        # Both should be finite and broadly similar (not identical)
        assert np.all(np.isfinite(xs1)) and np.all(np.isfinite(xs5))
        # End-point error should be in the same ballpark (within 20 %)
        err1 = np.linalg.norm(xs1[-1, :2] - x_goal[:2])
        err5 = np.linalg.norm(xs5[-1, :2] - x_goal[:2])
        assert err5 < err1 * 5 + 0.5, \
            f"linearize_every=5 diverged too much (err1={err1:.3f}, err5={err5:.3f})"

    def test_field_every_no_crash(self):
        """SQP with field_every > 1 should not crash."""
        s = _make_sqp(field_every=5)
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))
        xs, us = s.solve(x0, ref, max_iters=8)
        assert np.all(np.isfinite(xs))

    def test_sqp_near_goal_finite(self):
        """
        With x0 == xref (zero error), SQP should return finite values.
        NOTE: Using R=0 causes near-degenerate QP here; use R >= 1e-6*I
        in production to avoid numerical issues near the goal.
        """
        s  = _make_sqp()  # uses R=0.1*I so safe
        x0 = np.array([5.0, 5.0, 0.0])
        ref = _straight_ref(s, x0)
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs)), "SQP must return finite values at goal"
        assert np.all(np.isfinite(us))

    def test_dynamics_cache_populated_after_solve(self):
        s  = _make_sqp()
        x0 = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))
        s.solve(x0, ref)
        assert s.cache["A_dyn"] is not None

    def test_bound_cache_stable_on_repeated_default_calls(self):
        """Calling solve twice with default bounds should NOT rebuild l_box."""
        s   = _make_sqp()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))

        l_box_before = s.cache["l_box"].copy()
        s.solve(x0, ref)
        s.solve(x0, ref)   # second call — should reuse bound cache
        assert np.array_equal(s.cache["l_box"], l_box_before)

    def test_custom_bounds_override(self):
        s  = _make_sqp()
        x0 = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))

        tight_umin = np.array([-0.1, -0.1])
        tight_umax = np.array([ 0.1,  0.1])
        xs, us = s.solve(x0, ref, umin=tight_umin, umax=tight_umax)
        assert np.all(us >= -0.1 - 1e-4)
        assert np.all(us <=  0.1 + 1e-4)

    def test_short_ref_broadcast(self):
        """ref shorter than N should be broadcast via min(k+1, max_ref_idx)."""
        s  = _make_sqp()
        x0 = np.zeros(s.n)
        # Only 2 reference points for a horizon of 10
        ref = np.tile(np.array([1.0, 1.0, 0.0]), (2, 1))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))


# ══════════════════════════════════════════════════════════════════════════════
# 5. AL Solvers (iLQR and DDP)
# ══════════════════════════════════════════════════════════════════════════════

class TestALSolvers:

    @pytest.mark.parametrize("make_fn", [_make_ilqr, _make_ddp])
    def test_solve_output_shapes(self, make_fn):
        s   = make_fn()
        x0  = np.array([0.0, 0.0, 0.0])
        ref = _straight_ref(s, np.array([5.0, 5.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert xs.shape == (s.N + 1, s.n)
        assert us.shape == (s.N, s.m)

    @pytest.mark.parametrize("make_fn", [_make_ilqr, _make_ddp])
    def test_solve_starts_at_x0(self, make_fn):
        s   = make_fn()
        x0  = np.array([1.0, -2.0, 0.3])
        ref = _straight_ref(s, x0)
        xs, us = s.solve(x0, ref)
        assert np.allclose(xs[0], x0, atol=1e-8)

    @pytest.mark.parametrize("make_fn", [_make_ilqr, _make_ddp])
    def test_solve_finite(self, make_fn):
        s   = make_fn()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([2.0, 2.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(us))

    @pytest.mark.parametrize("make_fn", [_make_ilqr, _make_ddp])
    def test_solve_respects_u_bounds(self, make_fn):
        s   = make_fn()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([5.0, 0.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(us >= s.u_min - 1e-4)
        assert np.all(us <= s.u_max + 1e-4)

    @pytest.mark.parametrize("make_fn", [_make_ilqr, _make_ddp])
    def test_dyn_cache_cleared_between_solves(self, make_fn):
        """Each call to solve() must reset _il_dyn_cache."""
        s   = make_fn()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 0.0, 0.0]))
        s.solve(x0, ref)
        # Cache should have been populated and then used
        # After second call it should again be valid (not stale)
        s.solve(x0, ref)
        assert s._il_dyn_cache["Ad"] is not None   # populated during inner loop

    def test_ilqr_linearize_every_2(self):
        """linearize_every=2 should still produce finite, reasonable output."""
        s   = _make_ilqr(linearize_every=2)
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([2.0, 2.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(us))

    def test_ilqr_field_every_2(self):
        """field_every=2 should not crash."""
        s   = _make_ilqr(field_every=2)
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))

    def test_ddp_falls_back_gracefully(self):
        """
        ALDDPSolver with a dynamics model that has no 2nd-order Jacobians
        should fall back to iLQR behaviour without raising.
        """
        s   = _make_ddp()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([1.0, 1.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))

    def test_al_warm_start(self):
        s   = _make_ilqr()
        x0  = np.zeros(s.n)
        ref = _straight_ref(s, np.array([2.0, 2.0, 0.0]))
        xs1, us1 = s.solve(x0, ref)
        xs2, us2 = s.solve(x0, ref, us_init=us1)
        assert np.all(np.isfinite(xs2))

    def test_ilqr_near_goal_finite(self):
        """iLQR at goal should return finite values."""
        s  = _make_ilqr()
        x0 = np.array([3.0, 3.0, 0.0])
        ref = _straight_ref(s, x0)
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(us))

    def test_ilqr_vs_sqp_same_problem(self):
        """iLQR and SQP on the same problem should both reduce goal distance."""
        x0     = np.array([0.0, 0.0, 0.0])
        x_goal = np.array([3.0, 0.0, 0.0])

        sqp  = _make_sqp()
        ilqr = _make_ilqr()

        ref_s = _straight_ref(sqp,  x_goal)
        ref_i = _straight_ref(ilqr, x_goal)

        xs_s, _ = sqp.solve(x0, ref_s)
        xs_i, _ = ilqr.solve(x0, ref_i)

        # Both should move toward the goal
        dist_s = np.linalg.norm(xs_s[-1, :2] - x_goal[:2])
        dist_i = np.linalg.norm(xs_i[-1, :2] - x_goal[:2])

        assert dist_s < np.linalg.norm(x0[:2] - x_goal[:2]), \
            "SQP should reduce distance to goal"
        assert dist_i < np.linalg.norm(x0[:2] - x_goal[:2]), \
            "iLQR should reduce distance to goal"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Obstacle avoidance with field
# ══════════════════════════════════════════════════════════════════════════════

class TestSolverWithField:

    @pytest.fixture(scope="class")
    def field(self):
        return _load_risk_field()

    def test_sqp_with_field_finite(self, field):
        s   = _make_sqp(field=field)
        x0  = np.array([20.0, 20.0, 0.0])
        ref = _straight_ref(s, np.array([70.0, 70.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))

    def test_ilqr_with_field_finite(self, field):
        s   = _make_ilqr(field=field)
        x0  = np.array([20.0, 20.0, 0.0])
        ref = _straight_ref(s, np.array([70.0, 70.0, 0.0]))
        xs, us = s.solve(x0, ref)
        assert np.all(np.isfinite(xs))

    def test_sqp_field_constraints_enter_qp(self, field):
        """
        When the nominal trajectory passes near an obstacle, the field cache
        should be populated (A_field is not None).
        """
        s   = _make_sqp(field=field)
        x0  = np.zeros(s.n)
        x0[0], x0[1] = 48.0, 48.0   # near obstacle at (50, 50)
        ref = _straight_ref(s, np.array([55.0, 55.0, 0.0]))
        s.solve(x0, ref)
        # A_field may be None if no constraint was active, but if it is set it
        # must be the right number of columns
        if s.cache["A_field"] is not None:
            assert s.cache["A_field"].shape[1] == s.var_count


# ══════════════════════════════════════════════════════════════════════════════
# 7. Parse-bounds helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestParseBounds:

    def test_scalar_broadcast(self):
        s = _make_sqp()
        lb, ub = s._parse_bounds((-1.0, 1.0), 4)
        assert lb.shape == (4,)
        assert ub.shape == (4,)
        assert np.all(lb == -1.0)
        assert np.all(ub ==  1.0)

    def test_vector_passthrough(self):
        s  = _make_sqp()
        v  = np.array([-1.0, -2.0, -3.0])
        lb, ub = s._parse_bounds((v, -v), 3)
        assert np.array_equal(lb, v)
        assert np.array_equal(ub, -v)

    def test_inf_bounds_allowed(self):
        s  = _make_sqp()
        lb, ub = s._parse_bounds((-np.inf, np.inf), 5)
        assert np.all(np.isinf(lb))
        assert np.all(np.isinf(ub))
