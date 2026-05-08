"""
Benchmark scenarios and runner.

VehicleScenario handles aerial vehicle benchmarks for any dynamics model
defined in vehicles.toml. Vehicle physics, state bounds, and cost matrices
are read from the vehicle config; city environments come from cities.toml.

WMRScenario is an internal sanity check invoked only when the CLI receives
the --sanity flag. It is not user-configurable via TOML.
"""

from __future__ import annotations

import concurrent.futures
import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from data.registry import ALGO_REGISTRY, get_dynamics_cls

_BENCHMARK_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT  = _BENCHMARK_DIR.parent


def _resolve_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), _PROJECT_ROOT):
        candidate = base / p
        if candidate.exists():
            return candidate
    return p


from core.result import SimulationResult
from core.runner import (
    SimulationTask,
    effective_safety_dist,
    make_cost_matrices,
    make_x0,
)


class BenchmarkScenario(ABC):
    def __init__(self, bench_cfg: dict, run_cfg: dict) -> None:
        self.bench_cfg = bench_cfg
        self.run_cfg   = run_cfg
        self.name      = bench_cfg.get("name", "unnamed")

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def build_tasks(self) -> list[SimulationTask]: ...

    @abstractmethod
    def count_runs(self) -> int: ...


def _resolve_x_bounds(
    vehicle_cfg: dict,
    altitude: float,
    flexible: bool,
    z_lo_margin: float = 0.0,
    z_hi_margin: float = 20.0,
) -> tuple:
    """Build state bound arrays with altitude-relative z entries filled in."""
    bounds = vehicle_cfg.get("bounds", {})
    z_idx  = bounds.get("z_state_idx", 4)

    if flexible:
        lo = list(bounds.get("x_lo_flex", bounds.get("x_lo", [-1e9] * 12)))
        hi = list(bounds.get("x_hi_flex", bounds.get("x_hi", [ 1e9] * 12)))
    else:
        lo = list(bounds.get("x_lo", [-1e9] * 12))
        hi = list(bounds.get("x_hi", [ 1e9] * 12))

    if z_idx < len(lo):
        lo[z_idx] = altitude - z_lo_margin
        hi[z_idx] = altitude + z_hi_margin

    return lo, hi


def _resolve_u_bounds(vehicle_cfg: dict, dynamics, u_bound_factor: float) -> tuple[list, list, float]:
    """Return (u_lo, u_hi, w_hover). u_hi is physics-derived unless explicitly set."""
    bounds = vehicle_cfg.get("bounds", {})
    u_lo   = bounds.get("u_lo", [0.0] * dynamics.first_order_control_n)

    w_hover  = float(np.sqrt((dynamics.m * dynamics.g) / (4.0 * dynamics.kf)))
    u_hi     = [w_hover * u_bound_factor] * dynamics.first_order_control_n

    return u_lo, u_hi, w_hover


class VehicleScenario(BenchmarkScenario):
    """
    Aerial vehicle benchmark.

    Works with any dynamics class resolvable via larp.dynamics. Vehicle
    physics, state/control bounds, and cost matrices are read from vehicle_cfg
    (loaded from vehicles.toml). Cities come from city_registry (cities.toml).
    Altitude-relative z bounds and query margins are read from each city entry.
    """

    def __init__(
        self,
        bench_cfg: dict,
        run_cfg: dict,
        vehicle_cfg: dict,
        city_registry: dict,
    ) -> None:
        super().__init__(bench_cfg, run_cfg)
        self.vehicle_cfg   = vehicle_cfg
        self.city_registry = city_registry

    def setup(self) -> None:
        import larp as lp
        import larp.pp as pp
        from larp.env.environments import CityEnvironment
        import time

        sc       = self.bench_cfg.get("solver", {})
        phys     = self.vehicle_cfg.get("physics", {})
        dyn_name = self.vehicle_cfg.get("dynamics", "QuadcopterDynamics")
        dyn_cls  = get_dynamics_cls(dyn_name)
        self.dynamics = dyn_cls(**phys)

        layout = self.vehicle_cfg.get("state_layout", {})
        self.ref_indices    = layout.get("ref_indices",    [0, 2, 8])
        self.height_idx     = layout.get("z_idx",          4)
        self.heading_idx    = layout.get("heading_idx",    8)
        self.field_idx      = layout.get("field_idx",     [0, 2])
        self.ref_state_base = layout.get("ref_state_base", None)

        self.cost_cfg   = self.vehicle_cfg.get("cost", self.bench_cfg.get("cost", {}))
        self.solver_cfg = sc
        self.speeds     = self.bench_cfg.get("nominal_speeds", [5.0])
        self.algo_names = self.bench_cfg.get("algorithms", list(ALGO_REGISTRY.keys()))
        self.quick      = self.bench_cfg.get("quick", False)

        u_factor = self.run_cfg.get("u_bound_factor", 1.5)
        self.u_lo, self.u_hi, self.w_hover = _resolve_u_bounds(
            self.vehicle_cfg, self.dynamics, u_factor
        )

        replay_dir = self.bench_cfg.get(
            "replay_dir",
            str(Path(self.run_cfg.get("output_dir", "results")) / "replays"),
        )
        self.replay_dir = Path(replay_dir)
        self.replay_dir.mkdir(parents=True, exist_ok=True)

        print("\n[JAX warm-up] Triggering discretize JIT...")
        t0 = time.time()
        self.dynamics.discretize(
            np.zeros((1, self.dynamics.first_order_state_n)),
            np.full((1, self.dynamics.first_order_control_n), self.w_hover),
            dt=sc["dt"],
        )
        print(f"  Done in {time.time() - t0:.2f}s")

        self._city_data = {}
        self._load_cities(lp, pp, CityEnvironment)

    def _city_names(self) -> list[str]:
        requested = self.bench_cfg.get("cities", list(self.city_registry.keys()))
        names     = [c for c in requested if c in self.city_registry]
        return names[:2] if self.quick else names

    def _load_cities(self, lp, pp, CityEnvironment) -> None:
        sc = self.solver_cfg

        for city_name in self._city_names():
            cfg            = self.city_registry[city_name]
            altitude       = float(cfg["altitude"])
            z_lo_m         = float(cfg.get("z_lo_margin", 0.0))
            z_hi_m         = float(cfg.get("z_hi_margin", 20.0))
            query_altitude = altitude - z_lo_m

            print(f"\n{'='*60}\n  Loading: {city_name}\n{'='*60}")

            try:
                env = CityEnvironment(
                    location     = cfg["location"],
                    altitude     = query_altitude,
                    dist         = cfg["dist"],
                    safety_margin= sc["safety_dist"],
                )
                rgjs = env.get_obstacle_rgjs()
            except Exception as exc:
                print(f"  [ERROR] CityEnvironment: {exc}")
                traceback.print_exc()
                continue

            print(f"  Obstacles: {len(rgjs)}")
            field    = lp.RiskField(rgjs=rgjs, center_point=(0, 0), size=cfg["dist"] * 2)
            field.set_all_repulsion([[100.0, 0], [0, 100.0]])
            grid_sz  = cfg["dist"] / 256
            quadtree = lp.quad.QuadTree(
                field, minimum_length_limit=grid_sz,
                edge_bounds=[0.2, 0.4, 0.6, 0.8], build_tree=True,
            )
            qfield    = lp.QRiskField(quadtree)
            field_map = {"field": field, "qfield": qfield}

            waypoints = np.array(cfg["waypoints"], dtype=float)
            ordered   = np.append(waypoints, waypoints[:1], axis=0)
            segments  = list(zip(ordered[:-1], ordered[1:]))
            if self.quick:
                segments = segments[:3]

            path_planner = pp.QuadPlanner(
                lp.quad.QuadTree(
                    field, minimum_length_limit=grid_sz,
                    edge_bounds=[0.2, 0.4, 0.6, 0.8], build_tree=True,
                )
            )
            path_planner.select_alg("a*")

            layout = self.vehicle_cfg.get("state_layout", {})
            print(f"  Pre-planning {len(segments)} paths...")
            seg_data: dict = {}
            for seg_i, (start_pt, end_pt) in enumerate(segments):
                start_xy = np.array(start_pt, dtype=float)
                goal_xy  = np.array(end_pt,   dtype=float)
                try:
                    raw_path = path_planner.find_path(start_xy, goal_xy)
                except Exception as exc:
                    print(f"    Seg {seg_i}: PathPlan ERROR: {exc}")
                    raw_path = None
                if raw_path is None or len(raw_path) < 2:
                    print(f"    Seg {seg_i}: No path found")
                    seg_data[seg_i] = None
                else:
                    eff_sd = effective_safety_dist(raw_path[:, :2], field, sc["safety_dist"])
                    x0     = make_x0(start_xy, raw_path[1], altitude, self.dynamics, layout)
                    seg_data[seg_i] = (raw_path, x0, goal_xy, eff_sd)
                    print(f"    Seg {seg_i}: ok ({len(raw_path)} wpts, eff_sd={eff_sd:.1f}m)")

            self._city_data[city_name] = {
                "field_map":   field_map,
                "seg_data":    seg_data,
                "segments":    segments,
                "altitude":    altitude,
                "z_lo_margin": z_lo_m,
                "z_hi_margin": z_hi_m,
                "city_cfg":    cfg,
            }

    def build_tasks(self) -> list[SimulationTask]:
        tasks = []
        sc    = self.solver_cfg

        for city_name, city_info in self._city_data.items():
            field_map  = city_info["field_map"]
            seg_data   = city_info["seg_data"]
            segments   = city_info["segments"]
            altitude   = city_info["altitude"]
            z_lo_m     = city_info["z_lo_margin"]
            z_hi_m     = city_info["z_hi_margin"]
            city_cfg   = city_info["city_cfg"]

            city_meta = {
                "location": list(city_cfg["location"]),
                "altitude": float(city_cfg["altitude"]),
                "dist":     float(city_cfg["dist"]),
            }

            for algo_name in self.algo_names:
                if algo_name not in ALGO_REGISTRY:
                    print(f"  [WARN] Unknown algorithm '{algo_name}', skipping.")
                    continue

                _, _, flexible = ALGO_REGISTRY[algo_name]
                x_bounds = _resolve_x_bounds(
                    self.vehicle_cfg, altitude, flexible, z_lo_m, z_hi_m
                )

                for seg_i in range(len(segments)):
                    for speed in self.speeds:
                        tasks.append(SimulationTask(
                            city_name      = city_name,
                            algo_name      = algo_name,
                            seg_i          = seg_i,
                            speed          = speed,
                            route_data     = seg_data.get(seg_i),
                            altitude       = altitude,
                            dynamics       = self.dynamics,
                            field_map      = field_map,
                            solver_cfg     = sc,
                            cost_cfg       = self.cost_cfg,
                            replay_dir     = self.replay_dir,
                            ref_indices    = self.ref_indices,
                            height_idx     = self.height_idx,
                            heading_idx    = self.heading_idx,
                            u_hover        = np.full(self.dynamics.first_order_control_n, self.w_hover),
                            u_bounds       = (self.u_lo, self.u_hi),
                            x_bounds       = x_bounds,
                            field_idx      = self.field_idx,
                            run_cfg        = self.run_cfg,
                            city_meta      = city_meta,
                            ref_state_base = self.ref_state_base,
                        ))
        return tasks

    def count_runs(self) -> int:
        quick      = self.bench_cfg.get("quick", False)
        algo_names = self.bench_cfg.get("algorithms", list(ALGO_REGISTRY.keys()))
        speeds     = self.bench_cfg.get("nominal_speeds", [5.0])
        requested  = self.bench_cfg.get("cities", list(self.city_registry.keys()))
        cities     = [c for c in requested if c in self.city_registry]
        if quick:
            cities = cities[:2]
        total = 0
        for city_name in cities:
            n_wpts = len(self.city_registry[city_name].get("waypoints", []))
            n_segs = min(3, n_wpts) if quick else n_wpts
            total += n_segs * len(algo_names) * len(speeds)
        return total


class WMRScenario(BenchmarkScenario):
    """
    WMR ground sanity check. Not user-configurable; invoked via --sanity flag.

    Validates that all solvers produce correct trajectories on a simple 2-D
    navigation problem before running expensive aerial benchmarks.
    """

    _SOLVER_CFG = {
        "dt": 0.1, "horizon_sec": 2.5, "stride": 3,
        "goal_tol": 15.0, "safety_dist": 5.0, "goal_blend": 15.0,
        "linearize_every": 1, "field_every": 1,
        "t_sim_min": 60.0, "t_sim_max": 900.0, "t_sim_factor": 3.0,
    }
    _COST_CFG = {"Q_diag": [500.0, 500.0, 50.0], "Qf_scale": 5.0, "R_scale": 0.5}
    _ALGOS    = [
        "SQP (no field)", "SQP (field)",
        "iLQR (no field)", "iLQR (field)",
        "DDP (no field)", "DDP (field)",
    ]
    _ROUTES   = [
        (np.array([100.0, 350.0]), np.array([640.0, 350.0])),
        (np.array([372.0,  50.0]), np.array([372.0, 700.0])),
        (np.array([ 50.0,  50.0]), np.array([690.0, 700.0])),
    ]

    def __init__(self, run_cfg: dict, quick: bool = False) -> None:
        super().__init__({"name": "WMR Sanity Check"}, run_cfg)
        self.quick       = quick
        self.solver_cfg  = self._SOLVER_CFG
        self.algo_names  = self._ALGOS

    def setup(self) -> None:
        import larp as lp
        import larp.pp as pp
        import larp.io as larp_io
        from larp.dynamics import WMRDynamics

        self.dynamics = WMRDynamics()

        field_path = _resolve_path("test/.rug_city_hall.rgj")
        self.field = larp_io.loadRGeoJSONFile(field_path)
        self.field.set_all_repulsion([[100.0, 0], [0, 100.0]])

        grid_sz  = min(self.field.size) / 120.0
        quadtree = lp.quad.QuadTree(
            self.field, minimum_length_limit=grid_sz,
            edge_bounds=[0.2, 0.4, 0.6, 0.8], build_tree=True,
        )
        self.field_map = {"field": self.field, "qfield": lp.QRiskField(quadtree)}

        u_v_max     = 5.0
        u_omega_max = 2.0
        self.u_bounds = ([0.0, -u_omega_max], [u_v_max, u_omega_max])
        n = self.dynamics.first_order_state_n
        self.x_bounds = ([-np.inf] * n, [np.inf] * n)

        self.path_planner = pp.QuadPlanner(
            lp.quad.QuadTree(
                self.field, minimum_length_limit=10.0,
                edge_bounds=[0.2, 0.5, 0.8], build_tree=True,
            )
        )
        self.path_planner.select_alg("a*")
        routes = self._ROUTES[:2] if self.quick else self._ROUTES

        print(f"\n{'='*60}\n  WMR SANITY CHECK\n{'='*60}")

        self.routes = []
        for start_xy, goal_xy in routes:
            raw = self.path_planner.find_path(start_xy, goal_xy)
            if raw is None:
                self.routes.append((start_xy, goal_xy, None))
            else:
                eff_sd = effective_safety_dist(
                    raw[:, :2], self.field, self.solver_cfg["safety_dist"]
                )
                self.routes.append((start_xy, goal_xy, (raw, goal_xy, eff_sd)))

    def build_tasks(self) -> list[SimulationTask]:
        tasks = []
        m     = self.dynamics.first_order_control_n

        for algo_name in self.algo_names:
            if algo_name not in ALGO_REGISTRY:
                continue
            for seg_i, (start_xy, goal_xy, route_info) in enumerate(self.routes):
                if route_info is None:
                    route_data = None
                else:
                    raw_path, goal, eff_sd = route_info
                    x0 = np.array([start_xy[0], start_xy[1], 0.0])
                    route_data = (raw_path, x0, goal, eff_sd)

                tasks.append(SimulationTask(
                    city_name   = "cityhall",
                    algo_name   = algo_name,
                    seg_i       = seg_i,
                    speed       = self.solver_cfg.get("nominal_speed", 4.0),
                    route_data  = route_data,
                    altitude    = 0.0,
                    dynamics    = self.dynamics,
                    field_map   = self.field_map,
                    solver_cfg  = self.solver_cfg,
                    cost_cfg    = self._COST_CFG,
                    replay_dir  = None,
                    ref_indices = [0, 1, 2],
                    height_idx  = -1,
                    heading_idx = -1,
                    u_hover     = np.zeros(m),
                    u_bounds    = self.u_bounds,
                    x_bounds    = self.x_bounds,
                    field_idx   = [0, 1],
                    run_cfg     = self.run_cfg,
                ))
        return tasks

    def count_runs(self) -> int:
        n_routes = 2 if self.quick else len(self._ROUTES)
        return n_routes * len(self.algo_names)


_METRIC_SPEC: dict[str, tuple[str, str]] = {
    "T":     ("avg_solve_time",    "{:.4f}s"),
    "L":     ("path_length",       "{:.1f}m"),
    "RefCL": ("ref_min_clearance", "{:.2f}m"),
    "CL":    ("min_clearance",     "{:.2f}m"),
    "TT":    ("travel_time",       "{:.1f}s"),
    "CR":    ("converge_rate",     "{:.1%}"),
    "CE":    ("control_effort",    "{:.3f}"),
}


def _fmt_result_line(res: SimulationResult, metric_keys: list[str]) -> str:
    status = "OK" if res.success else f"FAIL({res.crash_reason[:20]})"
    line   = (
        f"    [{res.algorithm}] {res.city} Seg {res.segment}"
        f" v={res.nominal_speed:.0f}m/s -> {status}"
    )
    for key in metric_keys:
        spec = _METRIC_SPEC.get(key)
        if spec is None:
            continue
        field, fmt = spec
        val = getattr(res, field, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            line += f" | {key}:n/a"
        else:
            line += f" | {key}:" + fmt.format(val)
    return line


class BenchmarkRunner:
    """Owns the thread pool and result collection for a benchmark session."""

    def __init__(self, run_cfg: dict) -> None:
        self.run_cfg      = run_cfg
        self.max_workers  = min(run_cfg.get("max_workers", 6), os.cpu_count() or 1)
        self.print_metrics = run_cfg.get("print_metrics", ["T", "L", "RefCL", "CL", "TT"])

    def run_scenario(
        self,
        scenario: BenchmarkScenario,
        progress,
        task,
        on_result=None,
    ) -> list[SimulationResult]:
        """
        Execute all tasks for a scenario and return the typed results.

        on_result: optional callable(SimulationResult) invoked immediately after
        each task completes — use it to save results incrementally.
        """
        tasks = scenario.build_tasks()
        print(f"\n  [Parallel] Executing {len(tasks)} route-runs (workers={self.max_workers})...")

        results: list[SimulationResult] = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        futures  = {executor.submit(t.run): t for t in tasks}
        try:
            for future in concurrent.futures.as_completed(futures):
                res: SimulationResult = future.result()
                print(_fmt_result_line(res, self.print_metrics))
                results.append(res)
                if on_result is not None:
                    try:
                        on_result(res)
                    except Exception as exc:
                        print(f"    [WARN] Failed to persist result: {exc}")
                progress.advance(task)
        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] Cancelling pending tasks (running tasks will finish)...")
            for f in futures:
                f.cancel()
            executor.shutdown(wait=False)
            raise
        else:
            executor.shutdown(wait=False)

        return results