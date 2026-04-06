import importlib.util

OSQP_INSTALLED = importlib.util.find_spec("osqp") is not None
OSM_INSTALLED = importlib.util.find_spec("osmnx") is not None
JAX_INSTALLED = importlib.util.find_spec("jax") is not None
MUJOCO_INSTALLED = importlib.util.find_spec("mujoco") is not None
MJX_INSTALLED = MUJOCO_INSTALLED and importlib.util.find_spec("mujoco.mjx") is not None
MATPLOTLIB_INSTALLED = importlib.util.find_spec("matplotlib") is not None

ON_EDGE_EPS = 1e-9

__all__ = [
    "OSQP_INSTALLED",
    "OSM_INSTALLED",
    "JAX_INSTALLED",
    "MUJOCO_INSTALLED",
    "MJX_INSTALLED",
    "MATPLOTLIB_INSTALLED",
    "ON_EDGE_EPS"
]