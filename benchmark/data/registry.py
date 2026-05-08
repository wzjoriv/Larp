"""Shared registries for the benchmark system.

Cities and vehicle physics live in cities.toml / vehicles.toml.
This module only keeps what must remain in Python: solver class lookups
and algorithm → (solver_class, field_key, flexible_bounds) mappings.
"""

import larp.dynamics as _dyn_module

from larp.tp.solver import SQPSolver, ALILQRSolver, ALDDPSolver


def get_dynamics_cls(name: str) -> type:
    """Return a dynamics class by name from larp.dynamics.

    Parameters
    ----------
    name : str
        Class name, e.g. ``"QuadcopterDynamics"``.

    Returns
    -------
    type
        The dynamics class.

    Raises
    ------
    ValueError
        If no class with that name exists in larp.dynamics.
    """
    cls = getattr(_dyn_module, name, None)
    if cls is None:
        raise ValueError(f"Dynamics class '{name}' not found in larp.dynamics.")
    return cls


ALGO_REGISTRY: dict[str, tuple] = {
    "SQP (flex bounds, no field)":   (SQPSolver,    None,     True),
    "SQP (flex bounds, QRiskField)": (SQPSolver,    "qfield", True),
    "SQP (no field)":                (SQPSolver,    None,     False),
    "SQP (QRiskField)":              (SQPSolver,    "qfield", False),
    "SQP (field)":                   (SQPSolver,    "field",  False),
    "iLQR (no field)":               (ALILQRSolver, None,     False),
    "iLQR (field)":                  (ALILQRSolver, "field",  False),
    "DDP (no field)":                (ALDDPSolver,  None,     False),
    "DDP (field)":                   (ALDDPSolver,  "field",  False),
}