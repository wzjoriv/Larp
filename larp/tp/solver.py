import importlib
from typing import List, Optional, Tuple, Dict, Any
import warnings
import numpy as np
import osqp
from scipy import sparse

from larp import PotentialField, QPotentailField
from larp.dynamics import Dynamics

JAX_INSTALLED = importlib.util.find_spec("jax") is not None

if JAX_INSTALLED:
    from jax import lax
    import jax.numpy as jnp
    from jax import jit
    from functools import partial

def angle_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference x - y wrapped to [-π, π]."""
    d = x - y
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

class SQPSolver:
    def __init__(self, 
                 field: PotentialField|QPotentailField, 
                 dynamics: Dynamics,
                 dt: float = 0.1,
                 horizon: int = 40,
                 x_bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                 u_bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 Qf: Optional[np.ndarray] = None,
                 minimum_dist: float = 0.1,
                 statefield_idxs: List[int] = [],
                 linearize_every: int = 1,
                 field_every: int = 3,          # Field update frequency
                 tol: float = 1e-4,
                 verbose: bool = False):
        
        self.dynamics = dynamics
        self.field = field
        self.is_qfield = isinstance(field, QPotentailField)
        self.dt = dt
        self.N = int(horizon) if isinstance(horizon, int) else int(np.ceil(horizon/dt))
        self.minimum_dist = minimum_dist
        self.tol = tol
        self.verbose = verbose

        self.angle_idxs = dynamics.angle_indices
        
        # Frequencies
        self.linearize_every = linearize_every
        self.field_every = field_every

        self.n = dynamics.first_order_state_n
        self.m = dynamics.first_order_control_n
        
        self.num_x_vars = self.N * self.n
        self.num_u_vars = self.N * self.m
        self.var_count = self.num_x_vars + self.num_u_vars

        # Weights
        self.Q = Q if Q is not None else 1e-3 * np.eye(self.n)
        self.R = R if R is not None else 1e-3 * np.eye(self.m)
        self.Qf = Qf if Qf is not None else self.Q

        # Default Bounds (Parsed once)
        self.x_min_def, self.x_max_def = self._parse_bounds(x_bounds, self.n)
        self.u_min_def, self.u_max_def = self._parse_bounds(u_bounds, self.m)

        self.statefield_idxs = np.array(statefield_idxs, dtype=int) if statefield_idxs else np.array([0, 1], dtype=int)

        # Matrices
        P_x_blocks = [sparse.csc_matrix(2 * self.Q) for _ in range(self.N - 1)]
        P_x_blocks.append(sparse.csc_matrix(2 * self.Qf))
        self.P = sparse.block_diag(P_x_blocks + [sparse.kron(sparse.eye(self.N), sparse.csc_matrix(2 * self.R))], format='csc')
        self.A_box = sparse.eye(self.var_count, format='csc')
        self.cache: Dict[str, Any] = {
            "A_dyn": None, "l_dyn": None, "u_dyn": None,
            "A_field": None, "l_field": None, "u_field": None,
            "l_box": None, "u_box": None
        }
        self._update_bounds_cache(None, None, None, None)
        self.prob = osqp.OSQP()
    
    def _parse_bounds(self, bounds, dim):
        lb = np.asarray(bounds[0])
        ub = np.asarray(bounds[1])
        if lb.ndim == 0: lb = np.full(dim, lb)
        if ub.ndim == 0: ub = np.full(dim, ub)
        return lb, ub
    
    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        """
        Simulates trajectory using RK4. 
        Optimized to use JAX JIT compilation if available.
        """
        x0 = np.asarray(x0).reshape(-1)
        
        # JAX Optimization: Use lax.scan for compiled loop
        if self.dynamics.jax_backend and JAX_INSTALLED:
            return np.asarray(self._rollout_jax(x0, us))

        # Fallback to Python loop (existing implementation)
        return self._rollout_python(x0, us)

    def _rollout_python(self, x0, us):
        # ... [Your existing loop code here] ...
        xs = np.zeros((self.N + 1, self.n))
        xs[0] = x0
        dt = self.dt
        
        curr_x = x0.reshape(1, -1)
        
        for k in range(self.N):
            u_k = us[k:k+1]
            
            k1 = self.dynamics.f(curr_x, u_k)[0]
            k2 = self.dynamics.f(curr_x + 0.5 * dt * k1, u_k)[0]
            k3 = self.dynamics.f(curr_x + 0.5 * dt * k2, u_k)[0]
            k4 = self.dynamics.f(curr_x + dt * k3, u_k)[0]
            
            curr_x = curr_x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            xs[k+1] = curr_x
            
        return xs
    
    @partial(jit, static_argnums=(0,))
    def _rollout_jax(self, x0, us):
        dt = self.dt
        
        def rk4_step(x_prev, u_curr):
            # Ensure batch dimension (1, n) for dynamics.f
            x_in = x_prev[None, :]
            u_in = u_curr[None, :]
            
            # Pass np=jnp to enforce JAX operations
            k1 = self.dynamics.f(x_in, u_in, np=jnp)[0]
            k2 = self.dynamics.f(x_in + 0.5 * dt * k1, u_in, np=jnp)[0]
            k3 = self.dynamics.f(x_in + 0.5 * dt * k2, u_in, np=jnp)[0]
            k4 = self.dynamics.f(x_in + dt * k3, u_in, np=jnp)[0]
            
            x_next = x_prev + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return x_next, x_next

        # scan carries state, returns stacked outputs
        _, xs_traj = lax.scan(rk4_step, x0, us)
        
        # Concatenate initial state [x0] with trajectory
        return jnp.concatenate([x0[None, :], xs_traj], axis=0)

    def _get_field_constraints(self, x_curr: np.ndarray, live_fields:List[PotentialField]|None = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.field is None: return None, None
        
        origin = x_curr[self.statefield_idxs]

        if self.is_qfield:
            quad = self.field.quadtree.find_quad(origin, max_depth=20)[0]

            if not len(quad.rgj_idx):
                return None, None

            repulsion_vecs = -self.field.field.repulsion_vectors(origin, filted_idx=quad.rgj_idx, min_dist_select=True)
        else:
            repulsion_vecs = -self.field.repulsion_vectors(origin, min_dist_select=False)

        if len(repulsion_vecs) == 0: return None, None

        norms = np.linalg.norm(repulsion_vecs, axis=1, keepdims=True)
        valid_mask = norms.flatten() > 1e-8
        if not np.any(valid_mask): return None, None
            
        repulsion_vecs = repulsion_vecs[valid_mask]
        norms = norms[valid_mask]
        
        n_vecs = repulsion_vecs / norms
        contacts = origin + repulsion_vecs
        
        A_local = np.zeros((len(n_vecs), self.n))
        A_local[:, self.statefield_idxs] = n_vecs
        b_local = np.sum(n_vecs * contacts, axis=1) - self.minimum_dist 

        return A_local, b_local

    def _update_dynamics(self, x0, xs_lin, us_lin):
        """Recomputes dynamics matrices (Ad, Bd, Gd) and constructs A_dyn."""

        # Check Inputs
        #assert not (np.isnan(x0).any() or np.isinf(x0).any()), f"NaN/Inf found in state x0: {x0}"
        #assert not (np.isnan(xs_lin).any() or np.isinf(xs_lin).any()), "NaN/Inf found in linearization trajectory xs_lin" 
        #assert not (np.isnan(us_lin).any() or np.isinf(us_lin).any()), "NaN/Inf found in linearization controls us_lin"

        Ad_list, Bd_list, Gd_list = self.dynamics.discretize(xs_lin[:-1], us_lin, dt=self.dt)

        #assert not np.isnan(Ad_list).any(), "NaN found in Ad_list (Jacobian). Likely a mathematical singularity in dynamics."
        #assert not np.isnan(Bd_list).any(), "NaN found in Bd_list."
        #assert not np.isnan(Gd_list).any(), "NaN found in Gd_list."
        
        data_list, row_list, col_list = [], [], []
        l_bounds, u_bounds = [], []

        # Identity for x_{k+1}
        data_list.append(np.ones(self.N * self.n))
        row_list.append(np.arange(self.N * self.n))
        col_list.append(np.arange(self.N * self.n))

        for k in range(self.N):
            row_start_idx = k * self.n
            
            # -A_k * x_k
            if k == 0:
                rhs = Ad_list[k] @ x0 + Gd_list[k]
                l_bounds.append(rhs); u_bounds.append(rhs)
            else:
                col_start_x = (k - 1) * self.n
                A_coo = sparse.coo_matrix(-Ad_list[k])
                data_list.append(A_coo.data)
                row_list.append(A_coo.row + row_start_idx)
                col_list.append(A_coo.col + col_start_x)
                l_bounds.append(Gd_list[k]); u_bounds.append(Gd_list[k])

            # -B_k * u_k
            col_start_u = self.num_x_vars + k * self.m
            B_coo = sparse.coo_matrix(-Bd_list[k])
            data_list.append(B_coo.data)
            row_list.append(B_coo.row + row_start_idx)
            col_list.append(B_coo.col + col_start_u)
            
        self.cache["A_dyn"] = sparse.coo_matrix(
            (np.concatenate(data_list), (np.concatenate(row_list), np.concatenate(col_list))),
            shape=(self.N * self.n, self.var_count)
        )
        self.cache["l_dyn"] = np.hstack(l_bounds)
        self.cache["u_dyn"] = np.hstack(u_bounds)

        #assert not np.isnan(self.cache["A_dyn"].data).any(), "NaN found in constructed A_dyn sparse matrix data"
        #assert not np.isnan(self.cache["l_dyn"]).any(), "NaN found in l_dyn vector"
        #assert not np.isnan(self.cache["u_dyn"]).any(), "NaN found in u_dyn vector"

    def _update_field(self, xs_lin, live_fields:List[PotentialField]|None=None):
        """Recomputes field constraints and constructs A_field."""
        if self.field is None: return

        data_list, row_list, col_list = [], [], []
        l_bounds, u_bounds = [], []
        current_row_idx = 0

        for k in range(self.N):
            A_obs, b_obs = self._get_field_constraints(xs_lin[k+1], live_fields=live_fields)
            
            if A_obs is not None: # TODO: Remove False when done testing
                A_obs_coo = sparse.coo_matrix(A_obs)
                col_start_x = k * self.n 
                
                data_list.append(A_obs_coo.data)
                row_list.append(A_obs_coo.row + current_row_idx)
                col_list.append(A_obs_coo.col + col_start_x)
                
                l_bounds.append(np.full(A_obs.shape[0], -np.inf))
                u_bounds.append(b_obs)
                current_row_idx += A_obs.shape[0]

        if current_row_idx > 0:
            self.cache["A_field"] = sparse.coo_matrix(
                (np.concatenate(data_list), (np.concatenate(row_list), np.concatenate(col_list))),
                shape=(current_row_idx, self.var_count)
            )
            self.cache["l_field"] = np.hstack(l_bounds)
            self.cache["u_field"] = np.hstack(u_bounds)

            # Verify Construction
            #assert not np.isnan(self.cache["A_field"].data).any(), "NaN found in constructed A_field data"
            #assert not np.isnan(self.cache["l_field"]).any(), "NaN found in l_field"
            #assert not np.isnan(self.cache["u_field"]).any(), "NaN found in u_field"
        else:
            self.cache["A_field"] = None


    def _update_bounds_cache(self, xmin, xmax, umin, umax):
        """
        Updates l_box/u_box in cache ONLY if input bounds differ from defaults.
        Uses a simple tuple key check to avoid recomputing large arrays.
        """
        # Determine effective bounds
        xn = xmin if xmin is not None else self.x_min_def
        xm = xmax if xmax is not None else self.x_max_def
        un = umin if umin is not None else self.u_min_def
        um = umax if umax is not None else self.u_max_def

        is_default = (xmin is None and xmax is None and umin is None and umax is None)
        
        if is_default and self.cache["l_box"] is not None:
            return # Already cached defaults

        # Regenerate vectors
        xn_vec, xm_vec = self._parse_bounds((xn, xm), self.n)
        un_vec, um_vec = self._parse_bounds((un, um), self.m)

        x_lo, x_up = np.tile(xn_vec, self.N), np.tile(xm_vec, self.N)
        u_lo, u_up = np.tile(un_vec, self.N), np.tile(um_vec, self.N)
        
        self.cache["l_box"] = np.hstack([x_lo, u_lo])
        self.cache["u_box"] = np.hstack([x_up, u_up])

    def _assemble_qp_matrices(self) -> Tuple[sparse.csc_matrix, np.ndarray, np.ndarray]:
        """
        Assembles final A, l, u from cached components: Dynamics, Field, and Box.
        """
        matrices = [self.cache["A_dyn"]]
        vecs_l = [self.cache["l_dyn"]]
        vecs_u = [self.cache["u_dyn"]]
        
        if self.cache["A_field"] is not None:
            matrices.append(self.cache["A_field"])
            vecs_l.append(self.cache["l_field"])
            vecs_u.append(self.cache["u_field"])
        
        matrices.append(self.A_box)
        vecs_l.append(self.cache["l_box"])
        vecs_u.append(self.cache["u_box"])
        
        A = sparse.vstack(matrices, format='csc')
        l = np.hstack(vecs_l)
        u = np.hstack(vecs_u)
        
        return A, l, u

    def solve(self, x0: np.ndarray, ref: np.ndarray,
              us_init: Optional[np.ndarray] = None,
              xmin: Optional[np.ndarray] = None, 
              xmax: Optional[np.ndarray] = None,
              umin: Optional[np.ndarray] = None, 
              umax: Optional[np.ndarray] = None,
              max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        
        x0 = np.asarray(x0).reshape(-1)

        #assert not (np.isnan(x0).any() or np.isinf(x0).any()), "Initial x0 contains NaN/Inf values"
        #assert us_init is not None or not (np.isnan(us_init).any() or np.isinf(x0).any()), "Initial us_init contains NaN/Inf values"
        
        if us_init is None:
            us = np.zeros((self.N, self.m))
        else:
            us = np.asarray(us_init)
            if us.shape[0] != self.N: us = np.resize(us, (self.N, self.m))

        xs = self.rollout(x0, us)

        #assert not (np.isnan(xs).any() or np.isinf(xs).any()), f"Rollout produced NaN/Inf values\n x0: \n{x0}, \n us_init: \n{us_init}, \n rollout xs: \n{xs}"

        # 1. Update Bounds Cache (only if changed)
        self._update_bounds_cache(xmin, xmax, umin, umax)

        # Reuse cache flags
        force_update = True 

        for i in range(max_iters):
            
            # 2. Determine updates
            update_dyn = force_update or (i % self.linearize_every == 0)
            update_field = force_update or (i % self.field_every == 0)

            # 3. Update Cache
            if update_dyn:
                self._update_dynamics(x0, xs, us)
            
            if update_field:
                self._update_field(xs)

            force_update = False 

            # 4. Assemble A, l, u
            A, l, u = self._assemble_qp_matrices()


            # --- DEBUG ASSERTIONS ---
            #assert not np.isnan(A.data).any(), f"NaN found in assembled A matrix data at iter {i}"
            #assert not np.isnan(l).any(), f"NaN found in assembled l vector at iter {i}"
            #assert not np.isnan(u).any(), f"NaN found in assembled u vector at iter {i}"
            # ------------------------

            # 5. Update Cost q (Always update for angles)
            q = np.zeros(self.var_count)
            max_ref_idx = len(ref) - 1
            for k in range(self.N):
                idx = k * self.n
                Qk = self.Q if (k < self.N - 1) else self.Qf
                xref = ref[min(k + 1, max_ref_idx)].copy()
                if self.angle_idxs:
                    for ai in self.angle_idxs:
                        delta = angle_diff(xs[k+1, ai], xref[ai])
                        xref[ai] = xs[k+1, ai] - delta
                q[idx : idx + self.n] = -2.0 * (Qk @ xref)

            #assert not np.isnan(q).any(), f"NaN found in cost vector q at iter {i}"

            # 6. Solve
            self.prob.setup(P=self.P, q=q, A=A, l=l, u=u, verbose=self.verbose, polish=False)
            res = self.prob.solve()

            if res.info.status not in ['solved', 'solved_inaccurate']:
                if self.verbose: print(f"OSQP failed: {res.info.status}")
                return xs, us

            z = res.x

            # --- DEBUG ASSERTIONS ---
            #assert z is not None, f"OSQP returned None solution at iter {i} (Status: {res.info.status})"
            #assert not np.isnan(z).any() or not np.isinf(z).any(), f"OSQP returned NaN or Inf values in solution z at iter {i}"
            # ------------------------

            xs_flat = z[:self.num_x_vars]
            us_flat = z[self.num_x_vars:]
            
            xs_new = np.vstack([x0, xs_flat.reshape(self.N, self.n)])
            us_new = us_flat.reshape(self.N, self.m)

            if np.linalg.norm(xs_new - xs) < self.tol and i >= 2:
                xs, us = xs_new, us_new
                break
            
            xs, us = xs_new, us_new
        
        # Wrap output
        for ai in self.angle_idxs:
            xs[:, ai] = (xs[:, ai] + np.pi) % (2*np.pi) - np.pi

        return xs, us

class DDPSolver:
    """
    Iterative Linear Quadratic Regulator (iLQR) with Quadratic Penalty for constraints.
    Interface matches SQPSolver.
    """
    def __init__(self, 
                 field: PotentialField|QPotentailField, 
                 dynamics: Dynamics,
                 dt: float = 0.1,
                 horizon: int = 40,
                 x_bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                 u_bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 Qf: Optional[np.ndarray] = None,
                 minimum_dist: float = 0.1,
                 statefield_idxs: List[int] = [],
                 linearize_every: int = 1, # Unused in DDP (linearizes every iter)
                 field_every: int = 1,     # Re-check obstacles every iter
                 tol: float = 1e-4,
                 verbose: bool = False):
        
        self.dynamics = dynamics
        self.field = field
        self.is_qfield = isinstance(field, QPotentailField)
        self.dt = dt
        self.N = int(horizon) if isinstance(horizon, int) else int(np.ceil(horizon/dt))
        self.minimum_dist = minimum_dist
        self.tol = tol
        self.verbose = verbose
        self.angle_idxs = dynamics.angle_indices

        self.n = dynamics.first_order_state_n
        self.m = dynamics.first_order_control_n
        
        # Weights
        self.Q = Q if Q is not None else 1e-3 * np.eye(self.n)
        self.R = R if R is not None else 1e-3 * np.eye(self.m)
        self.Qf = Qf if Qf is not None else self.Q
        
        # Penalty Weights (Hyperparameters for DDP)
        self.w_obs = 1000.0  # Weight for obstacle penetration
        self.w_bounds = 100.0 # Weight for state bound violation

        # Parse Bounds
        self.x_min, self.x_max = self._parse_bounds(x_bounds, self.n)
        self.u_min, self.u_max = self._parse_bounds(u_bounds, self.m)
        self.statefield_idxs = np.array(statefield_idxs, dtype=int) if statefield_idxs else np.array([0, 1], dtype=int)

        # DDP Specifics
        self.mu = 1e-6    # Regularization for matrix inversion
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.alphas = [1.0, 0.5, 0.25, 0.125, 0.01] # Line search rates

    def _parse_bounds(self, bounds, dim):
        lb = np.asarray(bounds[0])
        ub = np.asarray(bounds[1])
        if lb.ndim == 0: lb = np.full(dim, lb)
        if ub.ndim == 0: ub = np.full(dim, ub)
        return lb, ub

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        # Re-use your existing logic or the JAX version if preferred
        # Simple Python Euler/RK4 for now to ensure no dependency issues
        xs = np.zeros((self.N + 1, self.n))
        xs[0] = x0
        curr_x = x0.reshape(1, -1)
        dt = self.dt
        
        for k in range(self.N):
            u_k = us[k:k+1]
            k1 = self.dynamics.f(curr_x, u_k)[0]
            k2 = self.dynamics.f(curr_x + 0.5 * dt * k1, u_k)[0]
            k3 = self.dynamics.f(curr_x + 0.5 * dt * k2, u_k)[0]
            k4 = self.dynamics.f(curr_x + dt * k3, u_k)[0]
            curr_x = curr_x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            xs[k+1] = curr_x
        return xs

    def _get_obs_derivatives(self, x):
        """Calculates gradient/hessian for obstacle penalty."""
        if self.field is None: return 0, np.zeros(self.n), np.zeros((self.n, self.n))

        origin = x[self.statefield_idxs]
        
        # Similar logic to SQP _get_field_constraints
        if self.is_qfield:
            quad = self.field.quadtree.find_quad(origin, max_depth=20)[0]
            if not len(quad.rgj_idx): return 0, np.zeros(self.n), np.zeros((self.n, self.n))
            repulsion_vecs = -self.field.field.repulsion_vectors(origin, filted_idx=quad.rgj_idx, min_dist_select=True)
        else:
            repulsion_vecs = -self.field.repulsion_vectors(origin, min_dist_select=False)

        if len(repulsion_vecs) == 0: return 0, np.zeros(self.n), np.zeros((self.n, self.n))

        # Calculate Distance and Normal
        dists = np.linalg.norm(repulsion_vecs, axis=1)
        valid_mask = dists > 1e-8
        if not np.any(valid_mask): return 0, np.zeros(self.n), np.zeros((self.n, self.n))
        
        dists = dists[valid_mask]
        vecs = repulsion_vecs[valid_mask]
        normals = vecs / dists[:, None] # Direction away from obstacle

        # Constraint: dist_to_obs >= minimum_dist
        # Violation: val = minimum_dist - dist_to_obs > 0
        # Cost: 0.5 * w * val^2
        
        cost = 0
        lx = np.zeros(self.n)
        lxx = np.zeros((self.n, self.n))

        for i in range(len(dists)):
            val = self.minimum_dist - dists[i]
            if val > 0:
                # Quadratic Penalty
                cost += 0.5 * self.w_obs * val**2
                
                # Gradient: w * val * (-normal) -> because d(val)/dx = -d(dist)/dx = normal
                # Note: repulsion vectors point FROM obs TO bot. 
                # gradient of distance function is -repulsion_dir
                grad_d = -normals[i] 
                
                # Chain rule for J = 0.5 * w * (min - dist)^2
                # dJ/dx = w * (min - dist) * (-1) * d(dist)/dx = -w * val * grad_d
                
                # Mapping back to full state
                local_grad = -self.w_obs * val * grad_d 
                lx[self.statefield_idxs] += local_grad
                
                # Hessian (Gauss-Newton approx): w * grad_d * grad_d.T
                local_hess = self.w_obs * np.outer(grad_d, grad_d)
                # Add to full hessian block
                for r_i, r in enumerate(self.statefield_idxs):
                    for c_i, c in enumerate(self.statefield_idxs):
                        lxx[r, c] += local_hess[r_i, c_i]
        
        return cost, lx, lxx

    def _compute_cost(self, x, u, x_ref, final=False):
        # 1. Tracking Cost
        Q = self.Q if not final else self.Qf
        
        diff = x - x_ref
        # Handle angles
        for ai in self.angle_idxs:
            diff[ai] = angle_diff(x[ai], x_ref[ai])

        l = 0.5 * diff.T @ Q @ diff
        lx = Q @ diff
        lxx = Q
        
        lu = np.zeros(self.m)
        luu = np.zeros((self.m, self.m))

        if not final:
            l += 0.5 * u.T @ self.R @ u
            lu = self.R @ u
            luu = self.R

        # 2. Obstacle Penalty
        c_obs, lx_obs, lxx_obs = self._get_obs_derivatives(x)
        l += c_obs
        lx += lx_obs
        lxx += lxx_obs

        # 3. State Bound Penalty (Soft constraints)
        # Lower
        vio_min = self.x_min - x
        mask_min = vio_min > 0
        if np.any(mask_min):
            l += 0.5 * self.w_bounds * np.sum(vio_min[mask_min]**2)
            lx[mask_min] -= self.w_bounds * vio_min[mask_min]
            # Diagonal Hessian approximation
            np.fill_diagonal(lxx, lxx.diagonal() + self.w_bounds * mask_min)
            
        # Upper
        vio_max = x - self.x_max
        mask_max = vio_max > 0
        if np.any(mask_max):
            l += 0.5 * self.w_bounds * np.sum(vio_max[mask_max]**2)
            lx[mask_max] += self.w_bounds * vio_max[mask_max]
            np.fill_diagonal(lxx, lxx.diagonal() + self.w_bounds * mask_max)

        return l, lx, lu, lxx, luu

    def solve(self, x0: np.ndarray, ref: np.ndarray,
              us_init: Optional[np.ndarray] = None,
              xmin: Optional[np.ndarray] = None, xmax: Optional[np.ndarray] = None,
              umin: Optional[np.ndarray] = None, umax: Optional[np.ndarray] = None,
              max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        
        # 1. Setup
        x0 = np.asarray(x0).reshape(-1)
        # Update bounds if provided
        if xmin is not None: self.x_min = self._parse_bounds((xmin, None), self.n)[0]
        if xmax is not None: self.x_max = self._parse_bounds((None, xmax), self.n)[1]
        u_lb = umin if umin is not None else self.u_min
        u_ub = umax if umax is not None else self.u_max
        u_lb, u_ub = self._parse_bounds((u_lb, u_ub), self.m)

        if us_init is None:
            us = np.zeros((self.N, self.m))
        else:
            us = np.array(us_init)
            if us.shape[0] != self.N: us = np.resize(us, (self.N, self.m))

        xs = self.rollout(x0, us)
        
        # Initial Cost
        J = 0
        for k in range(self.N):
            c, _, _, _, _ = self._compute_cost(xs[k], us[k], ref[min(k, len(ref)-1)])
            J += c
        c, _, _, _, _ = self._compute_cost(xs[-1], None, ref[min(self.N, len(ref)-1)], final=True)
        J += c

        # 2. Main Loop
        mu = self.mu # Damping
        
        for i in range(max_iters):
            # --- Backward Pass ---
            k_ff = np.zeros((self.N, self.m))
            K_fb = np.zeros((self.N, self.m, self.n))
            
            # Linearize Dynamics (All at once or per step)
            # Utilizing your dynamics.discretize which likely processes batches or loops
            # Note: discretize returns Ad, Bd, Gd
            Ad, Bd, _ = self.dynamics.discretize(xs[:-1], us, dt=self.dt)

            _, Vx, _, Vxx, _ = self._compute_cost(xs[-1], None, ref[min(self.N, len(ref)-1)], final=True)
            
            back_pass_error = False
            
            for k in range(self.N - 1, -1, -1):
                # Linearization at k
                A_k, B_k = Ad[k], Bd[k]
                
                # Cost derivatives
                _, lx, lu, lxx, luu = self._compute_cost(xs[k], us[k], ref[min(k, len(ref)-1)])
                
                # Q-function terms (Gauss-Newton approximation: no tensor contraction)
                Qx = lx + A_k.T @ Vx
                Qu = lu + B_k.T @ Vx
                Qxx = lxx + A_k.T @ Vxx @ A_k
                Quu = luu + B_k.T @ Vxx @ B_k
                Qux = B_k.T @ Vxx @ A_k
                
                # Regularization / Inversion
                Quu_reg = Quu + mu * np.eye(self.m)
                
                try:
                    # Cholesky is preferred for checking positive-definiteness
                    L = np.linalg.cholesky(Quu_reg)
                    Quu_inv = np.linalg.inv(Quu_reg) 
                except np.linalg.LinAlgError:
                    mu = max(1e-6, mu * 10)
                    back_pass_error = True
                    if self.verbose: print(f"  Non-PD Matrix at step {k}, increasing mu to {mu}")
                    break
                
                k_gain = -Quu_inv @ Qu
                K_gain = -Quu_inv @ Qux
                
                k_ff[k] = k_gain
                K_fb[k] = K_gain
                
                # Update Value Function
                Vx = Qx + K_gain.T @ Quu @ k_gain + K_gain.T @ Qu + Qux.T @ k_gain
                Vxx = Qxx + K_gain.T @ Quu @ K_gain + K_gain.T @ Qux + Qux.T @ K_gain
                Vxx = 0.5 * (Vxx + Vxx.T) # Symmetrize
                
            if back_pass_error:
                if mu > self.mu_max: 
                    if self.verbose: print("  Max regularization reached.")
                    break
                continue

            # --- Forward Pass ---
            step_accepted = False
            
            for alpha in self.alphas:
                xs_new = np.zeros_like(xs)
                us_new = np.zeros_like(us)
                xs_new[0] = x0
                
                J_new = 0
                curr_x = x0.reshape(1, -1)
                
                for k in range(self.N):
                    # Feedforward + Feedback
                    dx = xs_new[k] - xs[k]
                    # Handle angle wrapping in delta x
                    for ai in self.angle_idxs:
                        dx[ai] = angle_diff(xs_new[k, ai], xs[k, ai])

                    u_unclamped = us[k] + alpha * k_ff[k] + K_fb[k] @ dx
                    
                    # Box Constraints (Clamping)
                    us_new[k] = np.clip(u_unclamped, u_lb, u_ub)
                    
                    # Integration (inline simplified or call rollout)
                    # We do step-by-step to sum cost immediately
                    u_in = us_new[k:k+1]
                    # Python RK4 step (copy of your logic)
                    k1 = self.dynamics.f(curr_x, u_in)[0]
                    k2 = self.dynamics.f(curr_x + 0.5 * self.dt * k1, u_in)[0]
                    k3 = self.dynamics.f(curr_x + 0.5 * self.dt * k2, u_in)[0]
                    k4 = self.dynamics.f(curr_x + self.dt * k3, u_in)[0]
                    curr_x = curr_x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    
                    # Angle wrapping for next state
                    for ai in self.angle_idxs:
                        curr_x[0, ai] = (curr_x[0, ai] + np.pi) % (2*np.pi) - np.pi
                    
                    xs_new[k+1] = curr_x
                    
                    c, _, _, _, _ = self._compute_cost(xs_new[k], us_new[k], ref[min(k, len(ref)-1)])
                    J_new += c
                
                c, _, _, _, _ = self._compute_cost(xs_new[-1], None, ref[min(self.N, len(ref)-1)], final=True)
                J_new += c
                
                if J_new < J:
                    J = J_new
                    xs = xs_new
                    us = us_new
                    mu = max(self.mu_min, mu / 10.0)
                    step_accepted = True
                    break
            
            if not step_accepted:
                mu *= 10.0
                if self.verbose: print(f"Iter {i}: Line search failed. mu -> {mu}")
                if mu > self.mu_max: break
            elif abs(J - (J_new if 'J_new' in locals() else J)) < self.tol and i > 2:
                if self.verbose: print(f"Converged at iter {i}")
                break

        return xs, us
    
class ALDDPSolver(DDPSolver):
    """
    Augmented Lagrangian DDP.
    Inherits from DDPSolver to share rollout/setup, but overrides cost and solve.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # AL Hyperparameters
        self.al_mu = 1.0         # Penalty parameter
        self.al_mu_factor = 10.0 # Factor to increase penalty
        self.al_max_mu = 1e6
        self.al_tol = 1e-3       # Constraint tolerance
        self.max_outer_iters = 5
        self.max_inner_iters = 15 # Iterations of DDP per AL step
        
        # We handle state bounds in AL, but Control bounds via Clamping in DDP
        # Dual variables (Lagrange multipliers)
        # Dimensions: [N+1, n_constraints]
        # Constraints: 2*n for box bounds (min/max) + 1 for obstacle (dynamic size handling is tricky, strictly we simplify)
        self.lambdas_x_min = None
        self.lambdas_x_max = None
        self.lambdas_obs = None # List of arrays? Or simplified fixed size? 
        
        # NOTE: Handling dynamic number of obstacles in AL is complex. 
        # We will implement AL for State Bounds, but keep Obstacles as Quadratic Penalty 
        # for stability and speed, unless we fix the number of obstacle constraints.
        # Below implements AL for State Bounds.

    def _compute_al_cost(self, x, u, x_ref, k, final=False):
        # 1. Base Costs (Tracking + Control)
        l, lx, lu, lxx, luu = super()._compute_cost(x, u, x_ref, final)
        
        # Remove the soft penalty from super() for state bounds, 
        # because we are adding AL terms here.
        # (Assuming super() adds soft bounds, we'd need to subtract them or 
        #  clean implementation. For simplicity, we assume super() logic is acceptable 
        #  base, and we ADD AL terms for strictness).
        
        # 2. AL Terms for State Bounds
        # Constraint: x >= x_min  =>  c = x_min - x <= 0
        c_min = self.x_min - x
        lam_min = self.lambdas_x_min[k]
        
        # Constraint: x <= x_max => c = x - x_max <= 0
        c_max = x - self.x_max
        lam_max = self.lambdas_x_max[k]
        
        # Apply AL for Min Bounds
        # I_mu(c, lam) = (1/2mu) * (|max(0, lam + mu*c)|^2 - lam^2)
        # Gradient: max(0, lam + mu*c) * grad(c)
        
        # Min
        force_min = lam_min + self.al_mu * c_min
        mask_min = force_min > 0
        if np.any(mask_min):
            # c = x_min - x -> grad = -1
            viol = c_min[mask_min]
            l_terms = lam_min[mask_min] * viol + 0.5 * self.al_mu * (viol**2)
            l += np.sum(l_terms)
            
            lx[mask_min] -= force_min[mask_min] # grad(c) is -1
            np.fill_diagonal(lxx, lxx.diagonal() + self.al_mu * mask_min)

        # Max
        force_max = lam_max + self.al_mu * c_max
        mask_max = force_max > 0
        if np.any(mask_max):
            # c = x - x_max -> grad = 1
            viol = c_max[mask_max]
            l_terms = lam_max[mask_max] * viol + 0.5 * self.al_mu * (viol**2)
            l += np.sum(l_terms)
            
            lx[mask_max] += force_max[mask_max] # grad(c) is 1
            np.fill_diagonal(lxx, lxx.diagonal() + self.al_mu * mask_max)
            
        return l, lx, lu, lxx, luu

    def solve(self, x0: np.ndarray, ref: np.ndarray,
              us_init: Optional[np.ndarray] = None,
              xmin: Optional[np.ndarray] = None, xmax: Optional[np.ndarray] = None,
              umin: Optional[np.ndarray] = None, umax: Optional[np.ndarray] = None,
              max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        
        # Setup similar to DDPSolver
        x0 = np.asarray(x0).reshape(-1)
        if xmin is not None: self.x_min = self._parse_bounds((xmin, None), self.n)[0]
        if xmax is not None: self.x_max = self._parse_bounds((None, xmax), self.n)[1]
        u_lb = umin if umin is not None else self.u_min
        u_ub = umax if umax is not None else self.u_max
        u_lb, u_ub = self._parse_bounds((u_lb, u_ub), self.m)

        if us_init is None:
            us = np.zeros((self.N, self.m))
        else:
            us = np.array(us_init)
            if us.shape[0] != self.N: us = np.resize(us, (self.N, self.m))

        xs = self.rollout(x0, us)

        # Initialize Dual Variables
        self.lambdas_x_min = np.zeros((self.N + 1, self.n))
        self.lambdas_x_max = np.zeros((self.N + 1, self.n))
        self.al_mu = 1.0

        # Override weights to disable soft penalty in base class (optional but recommended)
        # self.w_bounds = 0.0 

        total_steps = 0
        
        for outer_k in range(self.max_outer_iters):
            
            # --- Inner DDP Loop ---
            # Using the exact logic from DDPSolver.solve, but calling _compute_al_cost
            # We copy logic here to inject the custom cost function call
            
            mu = self.mu # Damping
            
            for i in range(self.max_inner_iters):
                # Backward Pass
                k_ff = np.zeros((self.N, self.m))
                K_fb = np.zeros((self.N, self.m, self.n))
                Ad, Bd, _ = self.dynamics.discretize(xs[:-1], us, dt=self.dt)
                
                _, Vx, _, Vxx, _ = self._compute_al_cost(xs[-1], None, ref[min(self.N, len(ref)-1)], self.N, final=True)
                
                back_pass_error = False
                for k in range(self.N - 1, -1, -1):
                    _, lx, lu, lxx, luu = self._compute_al_cost(xs[k], us[k], ref[min(k, len(ref)-1)], k)
                    
                    A_k, B_k = Ad[k], Bd[k]
                    Qx = lx + A_k.T @ Vx
                    Qu = lu + B_k.T @ Vx
                    Qxx = lxx + A_k.T @ Vxx @ A_k
                    Quu = luu + B_k.T @ Vxx @ B_k
                    Qux = B_k.T @ Vxx @ A_k
                    
                    Quu_reg = Quu + mu * np.eye(self.m)
                    try:
                        Quu_inv = np.linalg.inv(Quu_reg) 
                    except:
                        mu *= 10
                        back_pass_error = True
                        break
                    
                    k_gain = -Quu_inv @ Qu
                    K_gain = -Quu_inv @ Qux
                    k_ff[k] = k_gain
                    K_fb[k] = K_gain
                    
                    Vx = Qx + K_gain.T @ Quu @ k_gain + K_gain.T @ Qu + Qux.T @ k_gain
                    Vxx = Qxx + K_gain.T @ Quu @ K_gain + K_gain.T @ Qux + Qux.T @ K_gain
                    Vxx = 0.5 * (Vxx + Vxx.T)

                if back_pass_error: continue

                # Forward Pass (Line Search)
                J_curr = 0
                # Calculate current cost for comparison
                for k in range(self.N):
                     c,_,_,_,_ = self._compute_al_cost(xs[k], us[k], ref[min(k, len(ref)-1)], k)
                     J_curr += c
                c,_,_,_,_ = self._compute_al_cost(xs[-1], None, ref[min(self.N, len(ref)-1)], self.N, final=True)
                J_curr += c

                step_accepted = False
                for alpha in self.alphas:
                    xs_new = np.zeros_like(xs)
                    us_new = np.zeros_like(us)
                    xs_new[0] = x0
                    curr_x = x0.reshape(1, -1)
                    J_new = 0
                    
                    for k in range(self.N):
                        dx = xs_new[k] - xs[k]
                        for ai in self.angle_idxs:
                            dx[ai] = angle_diff(xs_new[k, ai], xs[k, ai])

                        u_unclamped = us[k] + alpha * k_ff[k] + K_fb[k] @ dx
                        us_new[k] = np.clip(u_unclamped, u_lb, u_ub)
                        
                        # Integration
                        u_in = us_new[k:k+1]
                        k1 = self.dynamics.f(curr_x, u_in)[0]
                        k2 = self.dynamics.f(curr_x + 0.5 * self.dt * k1, u_in)[0]
                        k3 = self.dynamics.f(curr_x + 0.5 * self.dt * k2, u_in)[0]
                        k4 = self.dynamics.f(curr_x + self.dt * k3, u_in)[0]
                        curr_x = curr_x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                        for ai in self.angle_idxs:
                            curr_x[0, ai] = (curr_x[0, ai] + np.pi) % (2*np.pi) - np.pi
                        xs_new[k+1] = curr_x

                        c,_,_,_,_ = self._compute_al_cost(xs_new[k], us_new[k], ref[min(k, len(ref)-1)], k)
                        J_new += c
                    
                    c,_,_,_,_ = self._compute_al_cost(xs_new[-1], None, ref[min(self.N, len(ref)-1)], self.N, final=True)
                    J_new += c

                    if J_new < J_curr:
                        xs = xs_new
                        us = us_new
                        step_accepted = True
                        mu = max(1e-6, mu / 5)
                        break
                
                if not step_accepted: mu *= 10

            # --- Outer Loop: Update Dual Variables ---
            max_vio = 0.0
            
            # Check Max Constraints
            c_max_all = xs - self.x_max
            # Check Min Constraints
            c_min_all = self.x_min - xs
            
            # Update
            self.lambdas_x_max = np.maximum(0, self.lambdas_x_max + self.al_mu * c_max_all)
            self.lambdas_x_min = np.maximum(0, self.lambdas_x_min + self.al_mu * c_min_all)
            
            vio_max = np.max(np.maximum(0, c_max_all))
            vio_min = np.max(np.maximum(0, c_min_all))
            max_vio = max(vio_max, vio_min)
            
            if self.verbose:
                print(f"AL Outer {outer_k}: Max Vio: {max_vio:.5f}, mu: {self.al_mu}")

            if max_vio < self.al_tol:
                break
            
            self.al_mu = min(self.al_max_mu, self.al_mu * self.al_mu_factor)

        return xs, us