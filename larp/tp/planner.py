
from larp.dynamics import Dynamics
import numpy as np
from typing import Optional, Tuple, List, Union

from larp.field import PotentialField
from larp.tp.solver import Solver, MPCTrackSolver
from larp.types import Point, Trajectory

"""
Author: Josue N Rivera

Module providing trajectory planning algorithms. They start from a provided reference parg.
"""

class Planner():

    """
    Course path without dynamics included
    """

    def __init__(self, solver:Solver,
                 path:Union[List[Point], np.ndarray],
                 state_field_mask:Optional[List[bool]] = None,
                 init_iter_mult:int = 5, waypoint_tol:float = 1.0):
        
        self.solver = solver
        self.field = solver.field
        self.state_field_mask = state_field_mask if state_field_mask is not None else [True]*2 + [False]*(self.dynamics.first_order_state_n-1)

        self.memory = {}
        self.path = path
        self.way_idx = 1
        self.waypoint_tol = waypoint_tol

        self.init_iter_mult = init_iter_mult

    def refresh(self):
        """
        Resets planner.
        """
        self.reset_path()

    def reset_waypoint(self):
        self.way_idx = 1

    def reset_path(self, path:Union[List[Point], np.ndarray]):
        self.path = path
        self.reset_waypoint()

    def skip_waypoint(self, step=1):
        self.way_idx = min(self.way_idx+step, len(self.path)-1)

    def waypoint(self, x0:Optional[np.ndarray] = None):

        if x0 is not None:
            while(np.linalg.norm(x0 - self.path[self.way_idx]) < self.waypoint_tol):
                self.skip_waypoint()

        return self.path[self.way_idx]

    def find_trajectory(self, x0: np.ndarray, path:List[Point], **kargs) -> np.ndarray:

        raise NotImplementedError
    
class GoalsPlanner(Planner):
    def __init__(self, field:PotentialField, dynamics:Dynamics, dt:float = 0.01, horizon: Union[int, float] = 50, 
                 u_bounds: Tuple[np.ndarray,np.ndarray] = (-np.inf, np.inf), Q=None, R=None):
        

        solver = MPCTrackSolver(field=field, dynamics=dynamics, dt=dt, horizon=horizon, u_bounds=u_bounds, Q=Q, R=Q)
        Planner.__init__(self, solver=solver)

        self.reset_memory()
        self.prev_us = None

    def reset_path(self, path):
        super().reset_path(path)
        self.prev_us = None

    def find_trajectory(self, x0: np.ndarray, max_iters: int = 10, reset_memory = False) -> Trajectory:

        """
        Solve for a state-control trajectory from x0 to x_goal.

        Returns xs (shape [N+1,n]) and us (shape [N,m]).
        """

        if reset_memory:
            self.prev_us = None
            max_iters = max_iters*self.init_iter_mult

        xs, us = self.solver.solve(x0, self.waypoint(x0), us_init = self.prev_us, max_iters=max_iters)

        self.memory['_prev_us_'] = us

        return xs, us