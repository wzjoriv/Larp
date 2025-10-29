from typing import List, Optional, Tuple, Union
import warnings
import numpy as np
import osqp
from scipy import sparse

from larp import PotentialField
from larp.dynamics import Dynamics
from larp.types import FieldConstraints, TrajHorizon, Trajectory

"""
Author: Josue N Rivera
"""

class Solver:

    def __init__(self, field:PotentialField,
                 dynamics:Dynamics,
                 dt: float = 0.01,
                 horizon: TrajHorizon = 50,
                 control_horizon:Optional[int] = None,
                 Q=None, R=None,
                 minimim_dist:float = 1,
                 statefield_mask:List[bool] = [],
                 tol:float = 1e-4):

        self.dynamics = dynamics
        self.field = field
        self.dt = dt
        self.N = np.array(horizon, dtype=int) if isinstance(horizon, int) else np.ceil(horizon/dt, dtype=int)
        self.control_horizon = self.N if control_horizon is None else control_horizon
        self.minimim_dist = minimim_dist
        self.tol = float(tol)

        self.n = dynamics.first_order_state_n
        self.m = dynamics.first_order_control_n
        self.var_count = self.N*self.n + self.N*self.m # Total number of variables for solver: dx_k (k=1..N-1) and du_k (k=0..N-1)
        self.eq_count = self.N * self.n # 

        # Cost matrices (default to small identity if not provided)
        self.Q  = Q  if Q  is not None else 1e-3*np.eye(self.n)
        self.R  = R  if R  is not None else 1e-3*np.eye(self.m)

        if len(statefield_mask) != self.n or len(statefield_mask) != 2:
            self.statefield_mask = np.array([True]*2+[False]*(self.n-2), dtype=bool)
        else:
            self.statefield_mask = np.array(self.statefield_mask, dtype=bool)

        H = np.zeros((self.var_count, self.var_count))

        # Stage cost: dx_k' Q dx_k (k=1..N) and du_k' R du_k (k=0..N-1)
        for k in range(self.N):
            ix = k*self.n
            H[ix:ix+self.n, ix:ix+self.n] += self.Q

        for k in range(self.control_horizon):
            iu = self.N*self.n + k*self.m
            H[iu:iu+self.m, iu:iu+self.m] += self.R

        self.P = sparse.csc_matrix(H)
        self.f = np.zeros(self.var_count)

        self.prob = osqp.OSQP()

    def rollout_trajectory(self, x0:np.ndarray, us:Optional[np.ndarray] = None) -> Trajectory:

        us = np.zeros((self.N, self.m)) if us is None else us
        xs = np.zeros((self.N+1, self.n))
        xs[0] = x0

        for k in range(self.N):
            # Euler integration of dynamics for initial guess
            dx = self.dynamics.f(xs[k:k+1,:], us[k:k+1,:])[0]
            xs[k+1] = xs[k] + dx * self.dt

        return xs, us

    def field_constraints(self, x:np.ndarray) -> FieldConstraints:
        origin = x[self.statefield_mask]
        repulsion_vecs = -self.field.repulsion_vectors(origin)

        # Return none if no obstacles nearby found
        if len(repulsion_vecs) < 1:
            return None, None

        contact = origin + repulsion_vecs
        repulsion_vecs = repulsion_vecs / np.linalg.norm(repulsion_vecs, axis=1, keepdims=True)

        # Extend to state dim
        A_x = np.zeros((len(repulsion_vecs), self.n))
        A_x[:, self.statefield_mask] = repulsion_vecs
        b_x = np.sum(repulsion_vecs * contact, axis=1) - self.minimim_dist

        return A_x, b_x
    
    def solve(self, x0:np.ndarray, x_goal:np.ndarray,
             us_init:Optional[np.ndarray] = None,
             max_iters: int = 10, **kwargs) -> Trajectory:
        """
        Solve for a state-control trajectory from x0 to x_goal.

        Returns xs (shape [N+1,n]) and us (shape [N,m]).
        """

        raise NotImplementedError
    
class MPCTrackSolver(Solver):
    def __init__(self, field:PotentialField, dynamics:Dynamics,
                dt: float = 0.01,
                horizon: int = 50,
                control_horizon:int = 10,
                u_bounds: Tuple[np.ndarray,np.ndarray] = (-np.inf, np.inf),
                Q:Optional[np.ndarray]=None,
                R:Optional[np.ndarray]=None,
                minimim_dist:float = 0.1,
                statefield_mask:List[bool] = [],
                tol:float = 1e-4):
        """
        MPC tracker trajectory solver with hard state constraints and input bounds.

        dynamics: system model
        field: potential field
        dt: timestep, horizon: number of steps
        u_bounds: (u_min, u_max) arrays for input limits  
        Q, R: stage cost weights for ereference error and control effort  
        Qf: final-state cost weight (for goal tracking)  
        """
        Solver.__init__(self, field=field,
                        dynamics=dynamics,
                        dt=dt,
                        horizon=horizon,
                        Q=Q,
                        R=R,
                        minimim_dist=minimim_dist,
                        statefield_mask=statefield_mask,
                        tol=tol)

        self.u_min, self.u_max = u_bounds

    def build_eq_constraints(self, xs: np.ndarray, us: np.ndarray, **kwargs):
        """Linear dynamics constraints: dx_{k+1} = A_k dx_k + B_k du_k"""
        
        # Equality constraints (linearized dynamics): dx_{k+1} = A_k dx_k + B_k du_k
        A, B = self.dynamics.discretize(xs[:-1], us, self.dt, estimate=True)

        # dx_{k+1} = A_k dx_k + B_k du_k =====> I dx_{k+1} - A_k dx_k - B_k du_k = 0
        A_eq = np.zeros((self.eq_count, self.var_count))
        b_eq = np.zeros(self.eq_count)

        for k in range(self.N):
            row = k*self.n
            # dx_{k+1} term
            col_x_next = k*self.n
            A_eq[row:row+self.n, col_x_next:col_x_next+self.n] = np.eye(self.n)

            # dx_k term
            if k > 0:
                col_x = (k-1)*self.n
                A_eq[row:row+self.n, col_x:col_x+self.n] = -A[k]

            # du_k term
            col_u = self.N*self.n + k*self.m
            A_eq[row:row+self.n, col_u:col_u+self.m] = -B[k]

        return A_eq, b_eq

    def build_ineq_constraints(self, xs: np.ndarray, us: np.ndarray, **kwargs):
        """
        Construct inequality constraints:
        - State constraints derived from potential field

        Returns:
            A_ineq: (num_ineq, var_count)
            l_ineq: (num_ineq,)
            u_ineq: (num_ineq,)
        """

        A_ineq_rows = []
        l_ineq = []
        u_ineq = []

        # ---- State constraints: A_x (x_k + dx_k) <= b_x  => A_x dx_k <= b_x - A_x x_k ----
        for k in range(1, self.N + 1):

            A_x, b_x = self.field_constraints(xs[k])

            if A_x is None or b_x is None:
                continue

            rhs = b_x - A_x.dot(xs[k])
            row = np.zeros((A_x.shape[0], self.var_count))

            col_x = (k - 1) * self.n
            row[:, col_x:col_x + self.n] = A_x

            A_ineq_rows.append(row)
            l_ineq.append(-np.inf * np.ones(A_x.shape[0]))
            u_ineq.append(rhs)

        # ---- Control bounds: (u_min - u_k) <= du_k <= (u_max - u_k) ----
        for k in range(self.N):
            row = np.zeros((self.m, self.var_count))
            col_u = self.N * self.n + k * self.m
            row[:, col_u:col_u + self.m] = np.eye(self.m)

            A_ineq_rows.append(row)
            l_ineq.append(self.u_min - us[k])
            u_ineq.append(self.u_max - us[k])

        # Combine all rows and bounds
        if len(A_ineq):
            A_ineq = np.vstack(A_ineq_rows)
            l_ineq = np.hstack(l_ineq)
            u_ineq = np.hstack(u_ineq)

        return A_ineq, l_ineq, u_ineq

    def solve(self, x0:np.ndarray, x_goal:np.ndarray,
             us_init:Optional[np.ndarray] = None,
             max_iters:int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for a state-control trajectory from x0 to x_goal.

        Returns xs (shape [N+1,n]) and us (shape [N,m]).
        """
        x0 = x0.reshape(-1)
        x_goal = x_goal.reshape( -1)

        # Initialize (zero) controls and simulate forward if no initial guess
        xs, us = self.rollout_trajectory(x0, us=us_init)

        # Iterative SQP loop (solve QP at each iteration)
        for _ in range(max_iters):
            # QP Variables z = [dx_1,...,dx_N, du_0,...,du_{N-1}]

            # Build the QP cost matrices [Some precomputed in init]
            for k in range(self.N):
                ix = k*self.n
                self.f[ix:ix+self.n] = 2 * self.Q.dot(xs[k+1] - x_goal)

            # Equality constraints (linearized dynamics): dx_{k+1} = A_k dx_k + B_k du_k
            A_eq, b_eq = self.build_eq_constraints(xs, us, x_goal)

            # Inequality constraints:
            # State: A_x (x_bar_k + dx_k) <= b_x  =>  A_x dx_k <= b_x - A_x x_bar_k
            # Input bounds: u_min <= u_bar + du_k <= u_max
            A_ineq, l_ineq, u_ineq = self.build_ineq_constraints(xs, us, x_goal)

            # Combine equality and inequality into OSQP format
            A_total = np.vstack([A_eq, A_ineq])
            l_total = np.hstack([b_eq, l_ineq])
            u_total = np.hstack([b_eq, u_ineq])  # equality has l=u=b_eq

            # Solve the QP with OSQP
            A_sp = sparse.csc_matrix(A_total)

            self.prob.setup(P=self.P, f=self.f, A=A_sp, l=l_total, u=u_total, verbose=False, warm_start=True)
            res = self.prob.solve()

            if res.info.status != "solved":
                warnings.warn("Solution not found. Previous known solution returned.")
                return xs, us

            # Extract (dx, du) from solution
            dx = res.x[:self.N*self.n].reshape(self.N, self.n)
            du = res.x[self.N*self.n:].reshape(self.N, self.m)

            # Update trajectory: x_new = x_bar + dx
            xs[1:] = xs[1:] + dx
            us = us + du

            # Check convergence (e.g. small update)
            if np.linalg.norm(du) < self.tol:
                break

        return xs, us