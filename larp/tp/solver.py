"""
larp/tp/solver.py
=================
Trajectory-planning solvers for the larp framework.

Solver hierarchy
----------------
Solver (ABC)               - shared rollout (RK4), obstacle constraints, bound parsing
├── SQPSolver              - Sequential QP via OSQP with warm-start caching
├── ALILQRSolver           - Augmented-Lagrangian iLQR  (Gauss-Newton, no 2nd-order dynamics)
└── ALDDPSolver            - Augmented-Lagrangian DDP   (full 2nd-order dynamics correction)

All solvers share the same public interface:

    xs, us = solver.solve(x0, ref,
                          us_init=None,
                          xmin=None, xmax=None,
                          umin=None, umax=None,
                          max_iters=<int>)
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import osqp
from scipy import sparse

from larp import QRiskField, RiskField
from larp.dynamics import Dynamics
from larp.fn import angle_diff

from larp.const import JAX_INSTALLED

if JAX_INSTALLED:
    import jax.numpy as jnp
    from jax import Array, jit, lax
else:
    def jit(fn, **kwargs):       # type: ignore[misc]
        return fn
    class Array:                 # type: ignore[no-redef]
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Base Solver
# ══════════════════════════════════════════════════════════════════════════════

class Solver(ABC):
    """
    Abstract base for all larp trajectory-planning solvers.

    Provides:
      * RK4 forward rollout (NumPy or JAX-JIT)
      * Repulsion queries from RiskField / QRiskField
      * Local linearised obstacle half-planes:  A_local @ x  <=  b_local
      * Bound parsing and shared hyper-parameters
    """

    def __init__(
        self,
        field: "RiskField | QRiskField",
        dynamics: Dynamics,
        dt: float = 0.1,
        horizon: int = 40,
        x_bounds: Tuple = (-np.inf, np.inf),
        u_bounds: Tuple = (-np.inf, np.inf),
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Qf: Optional[np.ndarray] = None,
        minimum_dist: float = 0.1,
        statefield_idxs: List[int] = [],
        linearize_every: int = 1,
        field_every: int = 3,
        tol: float = 1e-4,
        verbose: bool = False,
    ):
        self.dynamics = dynamics
        self.field = field
        self.is_qfield = isinstance(field, QRiskField)
        self.dt = dt
        self.N = int(horizon) if isinstance(horizon, int) else int(np.ceil(horizon / dt))
        self.minimum_dist = minimum_dist
        self.tol = tol
        self.verbose = verbose

        # Update frequencies
        self.linearize_every = max(1, int(linearize_every))
        self.field_every = max(1, int(field_every))

        # State / control dimensions
        self.n = dynamics.first_order_state_n
        self.m = dynamics.first_order_control_n
        self.angle_idxs = dynamics.angle_indices
        self.statefield_idxs = (
            np.array(statefield_idxs, dtype=int)
            if statefield_idxs
            else np.array([0, 1], dtype=int)
        )

        # Cost weights
        self.Q  = Q  if Q  is not None else 1e-3 * np.eye(self.n)
        self.R  = R  if R  is not None else 1e-3 * np.eye(self.m)
        self.Qf = Qf if Qf is not None else self.Q.copy()

        # Parsed bounds
        self.x_min, self.x_max = self._parse_bounds(x_bounds, self.n)
        self.u_min, self.u_max = self._parse_bounds(u_bounds, self.m)

        # Variable count bookkeeping
        self.num_x_vars = self.N * self.n
        self.num_u_vars = self.N * self.m
        self.var_count   = self.num_x_vars + self.num_u_vars

    # ── Utilities ──────────────────────────────────────────────────────────

    def _parse_bounds(self, bounds, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        if bounds is None:
            return np.full(dim, -np.inf), np.full(dim, np.inf)
            
        # Use 'is' to avoid triggering NumPy's element-wise equality evaluation
        if isinstance(bounds, (tuple, list)) and bounds[0] is None and bounds[1] is None:
            return np.full(dim, -np.inf), np.full(dim, np.inf)
            
        lb = np.asarray(bounds[0], dtype=float) if bounds[0] is not None else np.full(dim, -np.inf)
        ub = np.asarray(bounds[1], dtype=float) if bounds[1] is not None else np.full(dim, np.inf)
        
        if lb.ndim == 0:
            lb = np.full(dim, float(lb))
        if ub.ndim == 0:
            ub = np.full(dim, float(ub))
            
        return lb, ub

    def _init_controls(self, us_init: Optional[np.ndarray]) -> np.ndarray:
        """Return a valid (N, m) control sequence, zero-padded / resized if needed."""
        if us_init is None:
            return np.zeros((self.N, self.m))
        us = np.asarray(us_init)
        if us.shape[0] != self.N:
            us = np.resize(us, (self.N, self.m))
        return us.copy()

    # ── RK4 Rollout ────────────────────────────────────────────────────────

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        """
        Forward simulate with RK4 integration.  Returns xs of shape (N+1, n).
        Delegates to JAX-JIT when the dynamics backend supports it.
        """
        x0 = np.asarray(x0).reshape(-1)
        if JAX_INSTALLED and self.dynamics.jax_backend:
            return np.array(self._rollout_jax(jnp.array(x0), jnp.array(us)))
        return self._rollout_python(x0, us)

    def _rollout_python(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        xs    = np.zeros((self.N + 1, self.n))
        xs[0] = x0
        curr_x = x0.reshape(1, -1)
        dt = self.dt
        for k in range(self.N):
            u_k = us[k : k + 1]
            k1 = self.dynamics.f(curr_x, u_k)[0]
            k2 = self.dynamics.f(curr_x + 0.5 * dt * k1, u_k)[0]
            k3 = self.dynamics.f(curr_x + 0.5 * dt * k2, u_k)[0]
            k4 = self.dynamics.f(curr_x + dt * k3, u_k)[0]
            curr_x = curr_x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            xs[k + 1] = curr_x
        return xs

    @partial(jit, static_argnums=(0,))
    def _rollout_jax(self, x0: np.ndarray, us: np.ndarray) -> "Array":
        dt = self.dt

        def rk4_step(x_prev, u_curr):
            x_in, u_in = x_prev[None, :], u_curr[None, :]
            k1 = self.dynamics.f(x_in, u_in, np=jnp)[0]
            k2 = self.dynamics.f(x_in + 0.5 * dt * k1, u_in, np=jnp)[0]
            k3 = self.dynamics.f(x_in + 0.5 * dt * k2, u_in, np=jnp)[0]
            k4 = self.dynamics.f(x_in + dt * k3, u_in, np=jnp)[0]
            x_next = x_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next, x_next

        _, xs_traj = lax.scan(rk4_step, x0, us)
        return jnp.concatenate([x0[None, :], xs_traj], axis=0)

    # ── Field / Obstacle Constraints ───────────────────────────────────────

    def get_nearby_repulsion(
        self, x_curr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Query the risk field for repulsion vectors at the spatial position
        encoded in *x_curr* (indexed by ``statefield_idxs``).
        """
        if self.field is None:
            return None, None

        origin = x_curr[self.statefield_idxs]

        if self.is_qfield:
            quad = self.field.quadtree.find_quad(origin, max_depth=20)[0]
            if not len(quad.rgj_idx):
                return None, None
            repulsion_vecs = -self.field.field.repulsion_vectors(
                origin, filted_idx=quad.rgj_idx, min_dist_select=True
            )
        else:
            repulsion_vecs = -self.field.repulsion_vectors(
                origin, min_dist_select=False
            )

        if len(repulsion_vecs) == 0:
            return None, None

        norms = np.linalg.norm(repulsion_vecs, axis=1)
        valid = norms > 1e-8
        if not np.any(valid):
            return None, None

        return repulsion_vecs[valid], norms[valid]

    def get_field_constraints(
        self, x_curr: np.ndarray, live_field=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Linearise nearby obstacles into local half-plane constraints:

            A_local @ x  <=  b_local

        Each row is a supporting hyperplane at the nearest obstacle surface,
        shifted inward by ``minimum_dist``.

        Returns ``(None, None)`` when no obstacles are in range.
        """
        repulsion_vecs, norms = self.get_nearby_repulsion(x_curr)
        if repulsion_vecs is None:
            return None, None

        origin   = x_curr[self.statefield_idxs]
        n_vecs   = repulsion_vecs / norms[..., None]
        contacts = origin + repulsion_vecs

        A_local = np.zeros((len(n_vecs), self.n))
        A_local[:, self.statefield_idxs] = n_vecs
        b_local = np.sum(n_vecs * contacts, axis=1) - self.minimum_dist

        return A_local, b_local

    # ── Abstract interface ─────────────────────────────────────────────────

    @abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError(
            "Use a concrete subclass: SQPSolver, ALILQRSolver, ALDDPSolver"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SQP Solver  (OSQP-based, warm-start cache)
# ══════════════════════════════════════════════════════════════════════════════

class SQPSolver(Solver):
    """
    Sequential Quadratic Programming solver via OSQP.

    Each SQP iteration re-linearises the nonlinear dynamics and obstacle
    constraints around the current nominal trajectory and solves the resulting
    convex QP.  Three separate cache layers minimise redundant recomputation:

    * **Dynamics cache** — rebuilt every ``linearize_every`` iterations.
    * **Field cache**    — rebuilt every ``field_every`` iterations.
    * **Bound cache**    — rebuilt only when per-call xmin/xmax/umin/umax differ
                           from the defaults.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── Constant cost matrix  P  ──────────────────────────────────────
        P_x_blocks = [sparse.csc_matrix(2 * self.Q)] * (self.N - 1)
        P_x_blocks.append(sparse.csc_matrix(2 * self.Qf))
        self.P = sparse.block_diag(
            P_x_blocks
            + [sparse.kron(sparse.eye(self.N), sparse.csc_matrix(2 * self.R))],
            format="csc",
        )

        # Identity box-constraint matrix
        self.A_box = sparse.eye(self.var_count, format="csc")

        # OSQP instance
        self.prob = osqp.OSQP()

        # ── Cache ─────────────────────────────────────────────────────────
        self.cache: Dict[str, Any] = {
            "A_dyn":   None, "l_dyn":   None, "u_dyn":   None,
            "A_field": None, "l_field": None, "u_field": None,
            "l_box":   None, "u_box":   None,
        }
        self._update_bounds_cache(None, None, None, None)

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _update_bounds_cache(self, xmin, xmax, umin, umax):
        """Rebuild l_box / u_box only when the effective bounds have changed."""
        is_default = (xmin is None and xmax is None
                      and umin is None and umax is None)
        if is_default and self.cache["l_box"] is not None:
            return

        xn = xmin if xmin is not None else self.x_min
        xm = xmax if xmax is not None else self.x_max
        un = umin if umin is not None else self.u_min
        um = umax if umax is not None else self.u_max

        xn_vec, xm_vec = self._parse_bounds((xn, xm), self.n)
        un_vec, um_vec = self._parse_bounds((un, um), self.m)

        self.cache["l_box"] = np.hstack([np.tile(xn_vec, self.N),
                                          np.tile(un_vec, self.N)])
        self.cache["u_box"] = np.hstack([np.tile(xm_vec, self.N),
                                          np.tile(um_vec, self.N)])

    def _update_dynamics(self, x0, xs_lin, us_lin):
        """
        Rebuild the equality-constraint block that encodes linearised dynamics.
        """
        Ad_list, Bd_list, Gd_list = self.dynamics.discretize(
            xs_lin[:-1], us_lin, dt=self.dt, estimate=True
        )

        data_list, row_list, col_list = [], [], []
        l_bounds, u_bounds = [], []

        data_list.append(np.ones(self.N * self.n))
        row_list.append(np.arange(self.N * self.n))
        col_list.append(np.arange(self.N * self.n))

        for k in range(self.N):
            row_start = k * self.n

            if k == 0:
                rhs = Ad_list[k] @ x0 + Gd_list[k]
                l_bounds.append(rhs)
                u_bounds.append(rhs)
            else:
                col_start_x = (k - 1) * self.n
                A_coo = sparse.coo_matrix(-Ad_list[k])
                data_list.append(A_coo.data)
                row_list.append(A_coo.row + row_start)
                col_list.append(A_coo.col + col_start_x)
                l_bounds.append(Gd_list[k])
                u_bounds.append(Gd_list[k])

            col_start_u = self.num_x_vars + k * self.m
            B_coo = sparse.coo_matrix(-Bd_list[k])
            data_list.append(B_coo.data)
            row_list.append(B_coo.row + row_start)
            col_list.append(B_coo.col + col_start_u)

        self.cache["A_dyn"] = sparse.coo_matrix(
            (
                np.concatenate(data_list),
                (np.concatenate(row_list), np.concatenate(col_list)),
            ),
            shape=(self.N * self.n, self.var_count),
        )
        self.cache["l_dyn"] = np.hstack(l_bounds)
        self.cache["u_dyn"] = np.hstack(u_bounds)

    def _update_field(self, xs_lin, live_fields=None):
        """
        Rebuild obstacle inequality block from field constraint linearisation.
        """
        if self.field is None:
            return

        data_list, row_list, col_list = [], [], []
        l_bounds, u_bounds = [], []
        current_row = 0

        for k in range(self.N):
            A_obs, b_obs = self.get_field_constraints(
                xs_lin[k + 1], live_field=live_fields
            )
            if A_obs is None:
                continue

            A_coo       = sparse.coo_matrix(A_obs)
            col_start_x = k * self.n

            data_list.append(A_coo.data)
            row_list.append(A_coo.row + current_row)
            col_list.append(A_coo.col + col_start_x)

            l_bounds.append(np.full(A_obs.shape[0], -np.inf))
            u_bounds.append(b_obs)
            current_row += A_obs.shape[0]

        if current_row > 0:
            self.cache["A_field"] = sparse.coo_matrix(
                (
                    np.concatenate(data_list),
                    (np.concatenate(row_list), np.concatenate(col_list)),
                ),
                shape=(current_row, self.var_count),
            )
            self.cache["l_field"] = np.hstack(l_bounds)
            self.cache["u_field"] = np.hstack(u_bounds)
        else:
            self.cache["A_field"] = None

    def _assemble_qp_matrices(self):
        """Stack cached blocks into the final OSQP constraint matrix A, l, u."""
        matrices = [self.cache["A_dyn"]]
        vecs_l   = [self.cache["l_dyn"]]
        vecs_u   = [self.cache["u_dyn"]]

        if self.cache["A_field"] is not None:
            matrices.append(self.cache["A_field"])
            vecs_l.append(self.cache["l_field"])
            vecs_u.append(self.cache["u_field"])

        matrices.append(self.A_box)
        vecs_l.append(self.cache["l_box"])
        vecs_u.append(self.cache["u_box"])

        return (
            sparse.vstack(matrices, format="csc"),
            np.hstack(vecs_l),
            np.hstack(vecs_u),
        )

    # ── Public interface ───────────────────────────────────────────────────

    def solve(
        self,
        x0: np.ndarray,
        ref: np.ndarray,
        us_init: Optional[np.ndarray] = None,
        xmin: Optional[np.ndarray] = None,
        xmax: Optional[np.ndarray] = None,
        umin: Optional[np.ndarray] = None,
        umax: Optional[np.ndarray] = None,
        max_iters: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:

        x0 = np.asarray(x0).reshape(-1)
        us = self._init_controls(us_init)
        xs = self.rollout(x0, us)

        assert not np.any(np.isnan(xs)), f"Initial rollout has a nan value in xs. values = {xs}"
        assert not np.any(np.isnan(us)), f"Initial controls has a nan value in us. values = {us}"

        self._update_bounds_cache(xmin, xmax, umin, umax)

        force_update = True
        max_ref_idx  = len(ref) - 1

        for i in range(max_iters):
            update_dyn   = force_update or (i % self.linearize_every == 0)
            update_field = force_update or (i % self.field_every == 0)

            if update_dyn:
                self._update_dynamics(x0, xs, us)
            if update_field:
                self._update_field(xs)

            force_update = False

            A, l, u = self._assemble_qp_matrices()

            q = np.zeros(self.var_count)
            for k in range(self.N):
                idx  = k * self.n
                Qk   = self.Q if k < self.N - 1 else self.Qf
                xref = ref[min(k + 1, max_ref_idx)].copy()
                if self.angle_idxs:
                    for ai in self.angle_idxs:
                        delta    = angle_diff(xs[k + 1, ai], xref[ai])
                        xref[ai] = xs[k + 1, ai] - delta
                q[idx : idx + self.n] = -2.0 * (Qk @ xref)

            z_warm = np.concatenate([xs[1:].flatten(), us.flatten()])
            self.prob.setup(
                P=self.P, q=q, A=A, l=l, u=u,
                verbose=self.verbose, polish=False, warm_starting=True,
            )
            self.prob.warm_start(x=z_warm)
            res = self.prob.solve()

            if "solved" not in res.info.status:
                if self.verbose:
                    print(f"[SQPSolver] OSQP failed at iter {i}: {res.info.status}")
                return xs, us

            z       = res.x
            xs_new  = np.vstack([x0, z[: self.num_x_vars].reshape(self.N, self.n)])
            us_new  = z[self.num_x_vars :].reshape(self.N, self.m)

            if np.linalg.norm(xs_new - xs) < self.tol and i >= 2:
                xs, us = xs_new, us_new
                break

            xs, us = xs_new, us_new

        if self.angle_idxs:
            aidx = np.array(self.angle_idxs, dtype=int)
            xs[:, aidx] = (xs[:, aidx] + np.pi) % (2 * np.pi) - np.pi

        return xs, us


# ══════════════════════════════════════════════════════════════════════════════
# Shared AL + iLQR / DDP machinery
# ══════════════════════════════════════════════════════════════════════════════

class _ALSolverBase(Solver):
    """
    Internal base for Augmented-Lagrangian trajectory optimisers.

    Outer loop (AL)
    ---------------
    1. Re-linearise field constraints along the current trajectory every
       ``field_every`` AL iterations (always on the first iteration).
    2. Run the inner iLQR / DDP solve with those fixed linearisations.
    3. Dual update:   lam_t  <-  max(0, lam_t + rho * c_t(xs[t]))
    4. Penalty scale: rho    <-  min(rho * rho_scale, rho_max)

    Inner loop (iLQR / DDP)
    -----------------------
    Backward pass
      Computes Q-function gains from discrete Jacobians.  Jacobians are
      recomputed every ``linearize_every`` inner iterations via
      ``_il_dyn_cache``.  DDP mode additionally adds second-order
      corrections from ``dynamics.dfdxx / dfduu / dfdux``.

    Forward pass
      RK4 rollout with exponential backtracking line search.
    """

    def __init__(
        self,
        *args,
        rho_init:   float = 10.0,
        rho_max:    float = 1e6,
        rho_scale:  float = 5.0,
        al_iters:   int   = 20,
        ilqr_iters: int   = 100,
        reg:        float = 1e-6,
        estimate_hessian: bool = True,
        use_ddp:    bool  = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.rho_init  = rho_init
        self.rho_max   = rho_max
        self.rho_scale = rho_scale
        self.al_iters  = al_iters
        self.ilqr_iters = ilqr_iters
        self.reg        = reg
        self.estimate_hessian = estimate_hessian
        self.use_ddp    = use_ddp

        self.alphas = 0.5 ** np.arange(10)

        # Dynamics Jacobian cache for inner iLQR loop
        self._il_dyn_cache: Dict[str, Any] = {"Ad": None, "Bd": None}

    # ── Constraint helpers ─────────────────────────────────────────────────

    def _linearize_field_along_traj(self, xs):
        """Linearise field constraints at every state on the trajectory."""
        return [self.get_field_constraints(xs[t]) for t in range(self.N + 1)]

    @staticmethod
    def _eval_constraint(A_t, b_t, x):
        """Constraint residual  c = A_t @ x - b_t  (feasible when c <= 0)."""
        return A_t @ x - b_t

    # ── AL penalty ────────────────────────────────────────────────────────

    @staticmethod
    def _al_terms(lam, rho, c, c_x):
        """
        Augmented-Lagrangian penalty for inequality constraints c <= 0.
        Returns (cost, grad, hess).
        """
        mu     = np.maximum(0.0, lam + rho * c)
        active = mu > 0

        cost  = float(np.sum(lam[active] * c[active] + 0.5 * rho * c[active] ** 2))
        cost -= float(np.sum(lam[~active] ** 2 / (2.0 * rho)))

        grad  = c_x.T @ mu
        scale = rho * active.astype(float)
        hess  = c_x.T @ (scale[:, None] * c_x)

        return cost, grad, hess

    # ── Per-step cost ──────────────────────────────────────────────────────

    def _stage_cost(self, x, u, x_ref, lam, rho, lin_con):
        diff = x - x_ref
        if self.angle_idxs:
            aidx = np.array(self.angle_idxs, dtype=int)
            diff[aidx] = angle_diff(x[aidx], x_ref[aidx])

        l    = 0.5 * (diff @ self.Q @ diff + u @ self.R @ u)
        l_x  = self.Q @ diff
        l_u  = self.R @ u
        l_xx = self.Q.copy()
        l_uu = self.R.copy()
        l_ux = np.zeros((self.m, self.n))

        A_t, b_t = lin_con
        if A_t is not None and len(lam) > 0:
            c = self._eval_constraint(A_t, b_t, x)
            al_c, al_g, al_h = self._al_terms(lam, rho, c, A_t)
            l    += al_c
            l_x  += al_g
            l_xx += al_h

        return l, l_x, l_u, l_xx, l_uu, l_ux

    def _terminal_cost(self, x, x_ref, lam, rho, lin_con):
        diff = x - x_ref
        if self.angle_idxs:
            aidx = np.array(self.angle_idxs, dtype=int)
            diff[aidx] = angle_diff(x[aidx], x_ref[aidx])

        l    = 0.5 * (diff @ self.Qf @ diff)
        l_x  = self.Qf @ diff
        l_xx = self.Qf.copy()

        A_t, b_t = lin_con
        if A_t is not None and len(lam) > 0:
            c = self._eval_constraint(A_t, b_t, x)
            al_c, al_g, al_h = self._al_terms(lam, rho, c, A_t)
            l    += al_c
            l_x  += al_g
            l_xx += al_h

        return l, l_x, l_xx

    def _total_aug_cost(self, xs, us, ref, lam_list, rho, lin_cons):
        max_ref_idx = len(ref) - 1
        J = 0.0
        for t in range(self.N):
            x_ref = ref[min(t, max_ref_idx)]
            l, *_ = self._stage_cost(xs[t], us[t], x_ref, lam_list[t], rho, lin_cons[t])
            J += l
        x_ref = ref[min(self.N, max_ref_idx)]
        l, *_ = self._terminal_cost(xs[self.N], x_ref, lam_list[self.N], rho, lin_cons[self.N])
        J += l
        return J

    # ── Box-DDP gains ──────────────────────────────────────────────────────

    def _box_ddp_gains(self, Q_uu, Q_u, Q_ux, u_bar, u_lb, u_ub):
        Q_uu_reg  = Q_uu + self.reg * np.eye(self.m)
        Q_uu_inv  = np.linalg.inv(Q_uu_reg)
        k_unc     = -Q_uu_inv @ Q_u
        K_unc     = -Q_uu_inv @ Q_ux
        u_clamped = np.clip(u_bar + k_unc, u_lb, u_ub)
        k         = u_clamped - u_bar
        saturated = (u_clamped <= u_lb + 1e-5) | (u_clamped >= u_ub - 1e-5)
        K         = K_unc.copy()
        K[saturated] = 0.0
        return k, K

    # ── DDP second-order correction ───────────────────────────────────────

    def ddp_correction(self, V_x, xs_t, us_t):
        if JAX_INSTALLED and self.dynamics.jax_backend:
            dQ_xx, dQ_uu, dQ_ux = self._ddp_correction_jax(jnp.asarray(V_x), jnp.asarray(xs_t), jnp.asarray(us_t))
            return np.array(dQ_xx), np.array(dQ_uu), np.array(dQ_ux)
        
        return self._ddp_correction_python(V_x, xs_t, us_t)
    
    def _ddp_correction_python(self, V_x, xs_t, us_t):
        f_xx, f_uu, f_ux = self.dynamics.discretize_hessian(xs_t, us_t, dt=self.dt, estimate=self.estimate_hessian)
        f_xx = f_xx[0]; f_uu = f_uu[0]; f_ux = f_ux[0]

        dQ_xx = np.einsum("i,ijk->jk", V_x, f_xx)
        dQ_uu = np.einsum("i,ijk->jk", V_x, f_uu)
        dQ_ux = np.einsum("i,ijk->jk", V_x, f_ux)

        return dQ_xx, dQ_uu, dQ_ux
    
    @partial(jit, static_argnums=(0,))
    def _ddp_correction_jax(self, V_x, xs_t, us_t):
        dQ_xx = np.zeros((self.n, self.n))
        dQ_uu = np.zeros((self.m, self.m))
        dQ_ux = np.zeros((self.m, self.n))

        if self.estimate_hessian:
            f_xx, f_uu, f_ux = self.dynamics._discretize_hessian_euler_jit(xs_t, us_t, dt=self.dt)
        else:
            f_xx, f_uu, f_ux = self.dynamics._discretize_hessian_zoh_jit(xs_t, us_t, dt=self.dt)

        f_xx = f_xx[0]; f_uu = f_uu[0]; f_ux = f_ux[0]

        dQ_xx = jnp.einsum("i,ijk->jk", V_x, f_xx)
        dQ_uu = jnp.einsum("i,ijk->jk", V_x, f_uu)
        dQ_ux = jnp.einsum("i,ijk->jk", V_x, f_ux)

        return dQ_xx, dQ_uu, dQ_ux

    def _run_ilqr(
        self,
        x0: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        ref: np.ndarray,
        lam_list: List[np.ndarray],
        rho: float,
        lin_cons: List[Tuple],
        u_lb: np.ndarray,
        u_ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple], List[np.ndarray]]:
        """
        Run up to ``ilqr_iters`` backward + forward passes on the augmented cost.

        Both the dynamics Jacobians and field constraint linearisations are
        refreshed inside the inner loop so the optimiser adapts to trajectory
        changes without waiting for the outer AL loop:

        * Dynamics (Ad, Bd) — rebuilt every ``linearize_every`` inner iterations.
        * Field constraints  — rebuilt every ``field_every`` inner iterations
          (skipped on inner iteration 0; lin_cons arrives fresh from the caller).

        Returns
        -------
        xs, us       : updated trajectory and controls
        lin_cons     : freshest field-constraint linearisation (for the dual update)
        lam_list     : multipliers resized to match the final lin_cons
        """
        angle_idxs  = self.angle_idxs          # local ref — avoid repeated attr lookup
        has_angles  = len(angle_idxs) > 0
        aidx        = np.array(angle_idxs, dtype=int) if has_angles else None
        max_ref_idx = len(ref) - 1

        for iteration in range(self.ilqr_iters):

            # ── Dynamics Jacobian (linearize_every) ──────────────────────
            if self._il_dyn_cache["Ad"] is None or iteration % self.linearize_every == 0:
                Ad, Bd, _ = self.dynamics.discretize(
                    xs[:-1], us, dt=self.dt, estimate=False
                )
                self._il_dyn_cache["Ad"] = Ad
                self._il_dyn_cache["Bd"] = Bd
            else:
                Ad = self._il_dyn_cache["Ad"]
                Bd = self._il_dyn_cache["Bd"]

            # ── Field constraints (field_every, skip iteration 0) ─────────
            if iteration > 0 and iteration % self.field_every == 0:
                lin_cons = self._linearize_field_along_traj(xs)
                for t in range(self.N + 1):
                    n_c = lin_cons[t][0].shape[0] if lin_cons[t][0] is not None else 0
                    if len(lam_list[t]) != n_c:
                        lam_list[t] = np.zeros(n_c)

            # ── Backward pass ─────────────────────────────────────────────
            x_ref_N      = ref[min(self.N, max_ref_idx)]
            _, V_x, V_xx = self._terminal_cost(
                xs[self.N], x_ref_N, lam_list[self.N], rho, lin_cons[self.N]
            )

            k_list: List[Optional[np.ndarray]] = [None] * self.N
            K_list: List[Optional[np.ndarray]] = [None] * self.N
            back_failed = False

            for t in reversed(range(self.N)):
                A_t = Ad[t]; B_t = Bd[t]
                x_ref_t = ref[min(t, max_ref_idx)]

                _, l_x, l_u, l_xx, l_uu, l_ux = self._stage_cost(
                    xs[t], us[t], x_ref_t, lam_list[t], rho, lin_cons[t]
                )

                Q_x  = l_x  + A_t.T @ V_x
                Q_u  = l_u  + B_t.T @ V_x
                Q_xx = l_xx + A_t.T @ V_xx @ A_t
                Q_uu = l_uu + B_t.T @ V_xx @ B_t
                Q_ux = l_ux + B_t.T @ V_xx @ A_t

                if self.use_ddp:
                    dQ_xx, dQ_uu, dQ_ux = self.ddp_correction(
                        V_x, xs[t : t + 1], us[t : t + 1]
                    )
                    Q_xx += dQ_xx; Q_uu += dQ_uu; Q_ux += dQ_ux

                try:
                    k, K = self._box_ddp_gains(Q_uu, Q_u, Q_ux, us[t], u_lb, u_ub)
                except np.linalg.LinAlgError:
                    if self.verbose:
                        print(f"[{self.__class__.__name__}] Singular Q_uu at t={t}.")
                    back_failed = True
                    break

                k_list[t] = k; K_list[t] = K

                V_x  = Q_x  + K.T @ Q_uu @ k + K.T @ Q_u  + Q_ux.T @ k
                V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                V_xx = 0.5 * (V_xx + V_xx.T)

            if back_failed:
                break

            # ── Forward pass ──────────────────────────────────────────────
            J_old    = self._total_aug_cost(xs, us, ref, lam_list, rho, lin_cons)
            accepted = False

            for alpha in self.alphas:
                xs_new    = np.zeros_like(xs)
                us_new    = np.zeros_like(us)
                xs_new[0] = x0
                curr_x    = x0.copy().reshape(1, -1)

                for t in range(self.N):
                    dx = xs_new[t] - xs[t]
                    if has_angles:
                        dx[aidx] = angle_diff(xs_new[t, aidx], xs[t, aidx])

                    u_raw     = us[t] + alpha * k_list[t] + K_list[t] @ dx
                    us_new[t] = np.clip(u_raw, u_lb, u_ub)

                    u_in = us_new[t : t + 1]
                    k1   = self.dynamics.f(curr_x, u_in)[0]
                    k2   = self.dynamics.f(curr_x + 0.5 * self.dt * k1, u_in)[0]
                    k3   = self.dynamics.f(curr_x + 0.5 * self.dt * k2, u_in)[0]
                    k4   = self.dynamics.f(curr_x + self.dt       * k3, u_in)[0]
                    curr_x = curr_x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                    if has_angles:
                        curr_x[0, aidx] = (curr_x[0, aidx] + np.pi) % (2*np.pi) - np.pi

                    xs_new[t + 1] = curr_x[0]

                J_new = self._total_aug_cost(xs_new, us_new, ref, lam_list, rho, lin_cons)
                if J_new < J_old:
                    accepted = True
                    break

            if not accepted:
                break

            dJ     = abs(J_old - J_new)
            xs, us = xs_new, us_new
            if dJ < self.tol and iteration > 2:
                break

        return xs, us, lin_cons, lam_list

    # ── Violation check ────────────────────────────────────────────────────

    def _max_field_violation(self, xs):
        max_viol = 0.0
        for t in range(self.N + 1):
            A_t, b_t = self.get_field_constraints(xs[t])
            if A_t is not None:
                c        = self._eval_constraint(A_t, b_t, xs[t])
                max_viol = max(max_viol, float(np.max(c)))
        return max_viol

    # ── Public interface ───────────────────────────────────────────────────

    def solve(
        self,
        x0:        np.ndarray,
        ref:       np.ndarray,
        us_init:   Optional[np.ndarray] = None,
        xmin:      Optional[np.ndarray] = None,
        xmax:      Optional[np.ndarray] = None,
        umin:      Optional[np.ndarray] = None,
        umax:      Optional[np.ndarray] = None,
        max_iters: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the Augmented-Lagrangian iLQR / DDP trajectory optimiser.

        Parameters
        ----------
        x0        : (n,) initial state
        ref       : (T, n) reference trajectory
        us_init   : (N, m) warm-start controls, or None for zeros
        xmin/xmax : per-call state bound overrides (not enforced in cost)
        umin/umax : per-call control bound overrides (enforced via clamping)
        max_iters : AL outer iterations (defaults to ``self.al_iters``)

        Returns
        -------
        xs : (N+1, n) optimised state trajectory
        us : (N,   m) optimised control sequence
        """
        x0 = np.asarray(x0).reshape(-1)

        u_lb = self._parse_bounds(
            (umin if umin is not None else self.u_min, self.u_max), self.m
        )[0]
        u_ub = self._parse_bounds(
            (self.u_min, umax if umax is not None else self.u_max), self.m
        )[1]

        us = self._init_controls(us_init)
        xs = self.rollout(x0, us)

        # Invalidate dynamics Jacobian cache at the start of each outer solve
        self._il_dyn_cache = {"Ad": None, "Bd": None}

        lin_cons = self._linearize_field_along_traj(xs)
        lam_list: List[np.ndarray] = [
            np.zeros(con[0].shape[0]) if con[0] is not None else np.zeros(0)
            for con in lin_cons
        ]
        rho  = self.rho_init
        n_al = max_iters if max_iters is not None else self.al_iters

        for al_iter in range(n_al):
            # _run_ilqr refreshes field linearisation internally every
            # field_every inner iterations and returns the freshest lin_cons
            # and lam_list so the dual update here always uses current data.
            xs, us, lin_cons, lam_list = self._run_ilqr(
                x0, xs, us, ref, lam_list, rho, lin_cons, u_lb, u_ub
            )

            # Dual update with freshest constraints from end of inner loop
            for t in range(self.N + 1):
                A_t, b_t = lin_cons[t]
                if A_t is not None and len(lam_list[t]) > 0:
                    c           = self._eval_constraint(A_t, b_t, xs[t])
                    lam_list[t] = np.maximum(0.0, lam_list[t] + rho * c)

            max_viol = self._max_field_violation(xs)
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}]  AL {al_iter + 1:3d}  "
                    f"max_viol={max_viol:.5f}  rho={rho:.2e}"
                )

            rho = min(rho * self.rho_scale, self.rho_max)

            if max_viol < self.tol:
                if self.verbose:
                    print(f"  Converged at AL iter {al_iter + 1}.")
                break

        # Vectorised angle wrap for all angle indices at once
        if self.angle_idxs:
            aidx = np.array(self.angle_idxs, dtype=int)
            xs[:, aidx] = (xs[:, aidx] + np.pi) % (2 * np.pi) - np.pi

        return xs, us


# ══════════════════════════════════════════════════════════════════════════════
# Public AL solvers
# ══════════════════════════════════════════════════════════════════════════════

class ALILQRSolver(_ALSolverBase):
    """
    Augmented-Lagrangian iLQR solver (Gauss-Newton, first-order dynamics model).

    Extra constructor kwargs (beyond base ``Solver`` parameters)
    ------------------------------------------------------------
    rho_init        : float = 10.0   Initial AL penalty weight rho
    rho_max         : float = 1e6    Maximum AL penalty weight
    rho_scale       : float = 5.0    Growth factor for rho per AL outer iteration
    al_iters        : int   = 20     Maximum outer AL iterations
    ilqr_iters      : int   = 100    Maximum inner iLQR iterations per AL step
    reg             : float = 1e-6   Tikhonov regularisation on Q_uu
    linearize_every : int   = 1      Re-linearise dynamics every N *inner* iterations
    field_every     : int   = 3      Re-linearise obstacles every N *inner* iterations
                                     (skipped on inner iteration 0; lin_cons is fresh
                                     from the outer AL loop at that point)
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_ddp", False)
        super().__init__(*args, **kwargs)


class ALDDPSolver(_ALSolverBase):
    """
    Augmented-Lagrangian DDP solver (full second-order dynamics correction).

    Extends ``ALILQRSolver`` with the Mayne 1966 / Tassa 2012 second-order
    correction to Q_xx, Q_uu, Q_ux.  Falls back gracefully to iLQR when
    the dynamics model does not expose second-order Jacobians.

    Requires ``dynamics.jax_backend = True`` for the correction to activate.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_ddp", True)
        super().__init__(*args, **kwargs)