"""Configuration loading, defaults, and validation for the benchmark system."""

from __future__ import annotations
import copy
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError("Install 'tomli' (pip install tomli) on Python < 3.11")


DEFAULTS: dict = {
    "output": {
        "dir":   "results",
        "store": "results/benchmark.db",
    },
    "figure": {
        "save":       True,
        "format":     "pdf",
        "dpi":        150,
        "output_dir": "results/figures",
        "style":      "paper",
        "width":      7.0,
        "height":     4.5,
    },
    "analyze": {
        "per_city":            False,
        "show":                False,
        "clearance_central":   "median",
        "clearance_log_scale": False,
        "rate_log_scale":      False,
        "rate_ymin_zero":      False,
        "aliases":             {},
    },
    "run": {
        "max_workers":         6,
        "save_replay":         True,
        "clearance_threshold": 0.1,
        "u_bound_factor":      1.5,
        "print_metrics":       ["T", "L", "RefCL", "CL", "TT"],
        "al_solver": {
            "rho_init":   10.0,
            "rho_max":    1.0e6,
            "rho_scale":  5.0,
            "al_iters":   20,
            "ilqr_iters": 100,
            "reg":        1.0e-6,
        },
        "solver": {
            "dt":              0.1,
            "horizon_sec":     2.5,
            "stride":          3,
            "goal_tol":        5.0,
            "safety_dist":     10.0,
            "goal_blend":      12.0,
            "linearize_every": 1,
            "field_every":     1,
            "t_sim_min":       120.0,
            "t_sim_max":       600.0,
            "t_sim_factor":    3.5,
        },
    },
    "_benchmark": {
        "enabled": True,
        "quick":   False,
    },
}

_REQUIRED_BENCHMARK = {"name", "vehicle", "algorithms"}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _load_cities_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return {c["name"]: c for c in raw.get("city", [])}


def _load_vehicles_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return {v["name"]: v for v in raw.get("vehicle", [])}


def load_config(path: Path) -> dict:
    """Load benchmark.toml plus companion cities.toml and vehicles.toml."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg: dict = {}
    for key, default_val in DEFAULTS.items():
        if key.startswith("_"):
            continue
        if isinstance(default_val, dict):
            cfg[key] = _deep_merge(default_val, raw.get(key, {}))
        else:
            cfg[key] = raw.get(key, default_val)

    for key in raw:
        if key not in cfg and key != "benchmark":
            cfg[key] = raw[key]

    base_benchmark = DEFAULTS["_benchmark"]
    global_solver  = cfg["run"].get("solver", {})
    benchmarks: list[dict] = []
    for b in raw.get("benchmark", []):
        merged = _deep_merge(base_benchmark, b)
        merged["solver"] = _deep_merge(global_solver, merged.get("solver", {}))
        benchmarks.append(merged)
    cfg["benchmark"] = benchmarks

    parent = path.parent
    cfg["city_registry"]    = _load_cities_toml(parent / "cities.toml")
    cfg["vehicle_registry"] = _load_vehicles_toml(parent / "vehicles.toml")

    return cfg


def validate_config(cfg: dict) -> list[str]:
    """Return human-readable error strings. Empty list means valid."""
    errors: list[str] = []
    vehicle_registry = cfg.get("vehicle_registry", {})

    for i, bench in enumerate(cfg.get("benchmark", [])):
        tag = f"[[benchmark]] #{i} ({bench.get('name', '?')})"

        for field in _REQUIRED_BENCHMARK:
            if not bench.get(field):
                errors.append(f"{tag}: missing required field '{field}'")

        vehicle_name = bench.get("vehicle", "")
        if vehicle_name and vehicle_registry and vehicle_name not in vehicle_registry:
            errors.append(f"{tag}: vehicle '{vehicle_name}' not found in vehicles.toml")

        sc = bench.get("solver", {})
        if sc.get("dt", 0.1) <= 0:
            errors.append(f"{tag}: solver.dt must be positive")
        if sc.get("horizon_sec", 2.5) <= 0:
            errors.append(f"{tag}: solver.horizon_sec must be positive")

    return errors