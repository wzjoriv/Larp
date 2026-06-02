
"""
larp.tp — Trajectory Planning
==============================
Solvers and reference planners for online trajectory optimisation.

Public API
----------
Solver          : abstract base for all solvers
SQPSolver       : Sequential QP via OSQP (warm-start cache)
ALILQRSolver    : Augmented-Lagrangian iLQR
ALDDPSolver     : Augmented-Lagrangian DDP (full 2nd-order dynamics)

Planner         : abstract base for all reference planners
WaypointPlanner : arc-length-projection waypoint follower (recommended default)
SplinePlanner   : B-spline path following with curvature-based velocity profiling;
                  degree selectable (default 3 cubic, 5 for quintic smoothness)

All planners share the same public interface:

    get_ref(x0, nominal_speed)          -> (N, n)    one-horizon reference
    get_full_ref(nominal_speed)         -> (T, n)    full-path reference
    find_trajectory(x0, ...)            -> (xs, us)  solve for one horizon
    get_full_trajectory(x0, ...)        -> (xs, us)  pre-planned full trajectory
"""

from larp.ltp.solver.solver import Solver, SQPSolver, ALILQRSolver, ALDDPSolver
from larp.ltp.solver.planner import (
    Planner,
    WaypointPlanner,
    SplinePlanner,
)

__all__ = [
    # Solvers
    "Solver",
    "SQPSolver",
    "ALILQRSolver",
    "ALDDPSolver",
    # Planners
    "Planner",
    "WaypointPlanner",
    "SplinePlanner",
]
