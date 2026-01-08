from typing import List, Optional, Tuple
import warnings
import numpy as np
import osqp
from scipy import sparse

from larp import PotentialField
from larp.dynamics import Dynamics

def angle_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Difference x - y wrapped to [-π, π]."""
    d = x - y
    d = (d + np.pi) % (2*np.pi) - np.pi
    return d

class MPCSolver:
    def __init__(self, 
                 field: PotentialField, 
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
                 angle_idxs: List[int] = [],
                 relinearize_every: int = 1,
                 tol: float = 1e-4,
                 verbose: bool = False):
        
        self.dynamics = dynamics
        self.field = field
        self.dt = dt
        self.N = int(horizon) if isinstance(horizon, (int, float)) else int(np.ceil(horizon/dt))
        self.minimum_dist = minimum_dist
        self.tol = tol
        self.verbose = verbose
        self.angle_idxs = angle_idxs
        self.relinearize_every = relinearize_every

        self.n = dynamics.first_order_state_n
        self.m = dynamics.first_order_control_n
        
        self.num_x_vars = self.N * self.n
        self.num_u_vars = self.N * self.m
        self.var_count = self.num_x_vars + self.num_u_vars

        # Weights
        self.Q = Q if Q is not None else 1e-3 * np.eye(self.n)
        self.R = R if R is not None else 1e-3 * np.eye(self.m)
        self.Qf = Qf if Qf is not None else self.Q

        # Bounds
        self.x_min_init, self.x_max_init = self._parse_bounds(x_bounds, self.n)
        self.u_min_init, self.u_max_init = self._parse_bounds(u_bounds, self.m)

        self.statefield_idxs = np.array(statefield_idxs, dtype=int) if statefield_idxs else np.array([0, 1], dtype=int)

        # Precompute Constant Hessian P
        P_x_blocks = [sparse.csc_matrix(2 * self.Q) for _ in range(self.N - 1)]
        P_x_blocks.append(sparse.csc_matrix(2 * self.Qf))
        self.P = sparse.block_diag(P_x_blocks + [sparse.kron(sparse.eye(self.N), sparse.csc_matrix(2 * self.R))], format='csc')

        self.prob = osqp.OSQP()
    
    def _parse_bounds(self, bounds, dim):
        lb = np.asarray(bounds[0])
        ub = np.asarray(bounds[1])
        if lb.ndim == 0: lb = np.full(dim, lb)
        if ub.ndim == 0: ub = np.full(dim, ub)
        return lb, ub

    def rollout(self, x0: np.ndarray, us: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0).reshape(-1)
        xs = np.zeros((self.N + 1, self.n))
        xs[0] = x0
        try:
            for k in range(self.N):
                dx = self.dynamics.f(xs[k:k+1], us[k:k+1])[0]
                xs[k+1] = xs[k] + dx * self.dt
        except Exception as e:
            warnings.warn(f"Rollout failed: {e}")
        return xs

    def _get_field_constraints(self, x_curr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generates linear half-plane constraints: A_obs * x <= b_obs 
        """
        if self.field is None: return None, None
        
        origin = x_curr[self.statefield_idxs]
        # rep_vec points from Robot -> Obstacle (Displacement)
        repulsion_vecs = -self.field.repulsion_vectors(origin, min_dist_select=False)

        if len(repulsion_vecs) == 0: return None, None

        norms = np.linalg.norm(repulsion_vecs, axis=1, keepdims=True)
        valid_mask = norms.flatten() > 1e-6
        if not np.any(valid_mask): return None, None
            
        repulsion_vecs = repulsion_vecs[valid_mask]
        norms = norms[valid_mask]
        
        # n: Unit vector pointing Robot -> Obstacle
        n_vecs = repulsion_vecs / norms
        
        # p: Contact point on obstacle surface
        contacts = origin + repulsion_vecs
        
        # A_local * x <= b_local
        # A = n (Project x onto the normal direction)
        A_local = np.zeros((len(n_vecs), self.n))
        A_local[:, self.statefield_idxs] = n_vecs  # FIX: Removed negative sign
        
        # b = n . p - d (Boundary is at contact point minus buffer)
        # Note: We sum across axis 1 to get dot product of each row
        b_local = np.sum(n_vecs * contacts, axis=1) - self.minimum_dist # FIX: Changed + to -

        return A_local, b_local

    def _build_qp_matrices(self, x0: np.ndarray, 
                           xs_lin: np.ndarray, us_lin: np.ndarray,
                           xs_ref: np.ndarray,
                           xmin, xmax, umin, umax) -> Tuple[sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
        
        # 1. Linearize Dynamics
        Ad_list, Bd_list, Gd_list = self.dynamics.discretize(xs_lin[:-1], us_lin, dt=self.dt)
        
        # 2. Optimized Triplet Construction for A Matrix
        data_list, row_list, col_list = [], [], []
        l_bounds, u_bounds = [], []
        
        # Track the current row index as we add constraints blocks
        current_row_idx = 0

        # --- A. DYNAMICS ---
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
            
        current_row_idx += self.N * self.n

        # --- B. FIELD CONSTRAINTS ---
        
        for k in range(self.N):
            # Constraints are computed around the linearization point xs_lin[k+1]
            # which corresponds to the variable x_{k+1} (index k in 0-based list)
            A_obs, b_obs = self._get_field_constraints(xs_lin[k+1])
            
            if A_obs is not None:
                A_obs_coo = sparse.coo_matrix(A_obs)
                
                # Column indices for x_{k+1}
                col_start_x = k * self.n 
                
                data_list.append(A_obs_coo.data)
                row_list.append(A_obs_coo.row + current_row_idx)
                col_list.append(A_obs_coo.col + col_start_x)
                
                # OSQP Inequality: l <= Ax <= u
                # Half-plane: -inf <= n.x <= n.p - d
                l_bounds.append(np.full(A_obs.shape[0], -np.inf))
                u_bounds.append(b_obs)
                
                current_row_idx += A_obs.shape[0]

        # --- C. VARIABLE BOUNDS ---
        # Identity matrix for all variables
        # A_box = I
        x_lo, x_up = np.tile(xmin, self.N), np.tile(xmax, self.N)
        u_lo, u_up = np.tile(umin, self.N), np.tile(umax, self.N)
        
        # Combine all bounds
        l = np.hstack([np.hstack(l_bounds), x_lo, u_lo])
        u = np.hstack([np.hstack(u_bounds), x_up, u_up])
        
        # Construct Final A Matrix
        # 1. Build the Dynamics/Field matrix from the accumulated lists
        if len(data_list) > 0:
            all_data = np.concatenate(data_list)
            all_rows = np.concatenate(row_list)
            all_cols = np.concatenate(col_list)
            A_general = sparse.coo_matrix((all_data, (all_rows, all_cols)), 
                                        shape=(current_row_idx, self.var_count))
        else:
            A_general = sparse.coo_matrix((0, self.var_count))

        # 2. Stack with Identity for Box Constraints
        A = sparse.vstack([A_general, sparse.eye(self.var_count)], format='csc')

        # --- COST VECTOR q ---
        q = np.zeros(self.var_count)
        max_ref_idx = len(xs_ref) - 1
        
        for k in range(self.N):
            idx = k * self.n
            Qk = self.Q if (k < self.N - 1) else self.Qf
            ref_k_idx = min(k + 1, max_ref_idx)
            xref = xs_ref[ref_k_idx].copy()

            if self.angle_idxs:
                for ai in self.angle_idxs:
                    delta = angle_diff(xs_lin[k+1, ai], xref[ai])
                    xref[ai] = xs_lin[k+1, ai] - delta
            
            q[idx : idx + self.n] = -2.0 * (Qk @ xref)

        return A, l, u, q

    def solve(self, x0: np.ndarray, ref: np.ndarray,
              us_init: Optional[np.ndarray] = None,
              xmin: Optional[np.ndarray] = None, 
              xmax: Optional[np.ndarray] = None,
              umin: Optional[np.ndarray] = None, 
              umax: Optional[np.ndarray] = None,
              max_iters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        
        x0 = np.asarray(x0).reshape(-1)
        xn, xm = self._parse_bounds((xmin, xmax), self.n) if xmin is not None else (self.x_min_init, self.x_max_init)
        un, um = self._parse_bounds((umin, umax), self.m) if umin is not None else (self.u_min_init, self.u_max_init)
        
        if us_init is None:
            us = np.zeros((self.N, self.m))
        else:
            us = np.asarray(us_init)
            if us.shape[0] != self.N: us = np.resize(us, (self.N, self.m))

        xs = self.rollout(x0, us)

        for i in range(max_iters):
            if i == 0 or (i % self.relinearize_every == 0):
                A, l, u, q = self._build_qp_matrices(x0, xs, us, ref, xn, xm, un, um)
                self.prob.setup(P=self.P, q=q, A=A, l=l, u=u, verbose=self.verbose, polish=False)

            res = self.prob.solve()
            
            if res.info.status not in ['solved', 'solved_inaccurate']:
                if self.verbose: print(f"OSQP failed: {res.info.status}")
                return xs, us

            z = res.x
            xs_flat = z[:self.num_x_vars]
            us_flat = z[self.num_x_vars:]
            
            xs_new = np.vstack([x0, xs_flat.reshape(self.N, self.n)])
            us_new = us_flat.reshape(self.N, self.m)

            if np.linalg.norm(xs_new - xs) < self.tol:
                xs, us = xs_new, us_new
                break
            
            xs, us = xs_new, us_new
        
        for ai in self.angle_idxs:
            xs[:, ai] = (xs[:, ai] + np.pi) % (2*np.pi) - np.pi

        return xs, us