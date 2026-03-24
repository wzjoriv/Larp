
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
SplinePlanner   : cubic-spline (C2) path following with velocity profiling
QuinticPlanner  : quintic B-spline (C4) path following — smoothest acceleration
LinearPlanner   : alias for WaypointPlanner

All planners share the same public interface:

    get_ref(x0, nominal_speed)          -> (N, n)    one-horizon reference
    get_full_ref(nominal_speed)         -> (T, n)    full-path reference
    find_trajectory(x0, ...)            -> (xs, us)  solve for one horizon
    get_full_trajectory(x0, ...)        -> (xs, us)  pre-planned full trajectory
"""

from larp.tp.solver import Solver, SQPSolver, ALILQRSolver, ALDDPSolver
from larp.tp.planner import (
    Planner,
    WaypointPlanner,
    SplinePlanner,
    QuinticPlanner,
    LinearPlanner,
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
    "QuinticPlanner",
    "LinearPlanner",
]
