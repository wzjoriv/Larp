"""
Low-level simulation primitives used by all benchmark scenarios.

High-level orchestration (scenario setup, parallel execution) lives in
core/scenario.py. Invoke via the unified CLI:
  python cli.py run
  python cli.py run --only Quadcopter
  python cli.py run --config my.toml
"""

from __future__ import annotations
import os
import time
import traceback
from pathlib import Path

os.environ["JAX_PLATFORMS"]        = "cpu"
os.environ["OSQP_ALGEBRA_BACKEND"] = "builtin"

import numpy as np
import pandas as pd

from data.config import load_config, validate_config
from data.registry import ALGO_REGISTRY
from core.result import SimulationResult

from larp.tp.solver import SQPSolver, ALILQRSolver, ALDDPSolver
from larp.tp.planner import WaypointPlanner

try:
    from rich.progress import (
        Progress, BarColumn, TaskProgressColumn,
        TimeElapsedColumn, TimeRemainingColumn, TextColumn, SpinnerColumn,
    )
    from rich.console import Console
    _RICH = True
except ImportError:
    _RICH = False


def make_cost_matrices(cost_cfg: dict, n: int, m: int) -> tuple:
    Q_diag = cost_cfg.get("Q_diag", [1.0] * n)
    Q  = np.diag(Q_diag)
    Qf = Q * cost_cfg.get("Qf_scale", 2.0)
    R  = np.eye(m) * cost_cfg.get("R_scale", 0.0)
    return Q, Qf, R


def build_solver(
    algo_name: str,
    dynamics,
    u_bounds: tuple,
    x_bounds: tuple,
    field_map: dict,
    solver_cfg: dict,
    cost_cfg: dict,
    field_idx: list,
    al_cfg: dict | None = None,
) -> object:
    cls, field_key, _ = ALGO_REGISTRY[algo_name]
    field = field_map.get(field_key)

    n, m     = dynamics.first_order_state_n, dynamics.first_order_control_n
    Q, Qf, R = make_cost_matrices(cost_cfg, n, m)

    kwargs = dict(
        field           = field,
        dynamics        = dynamics,
        dt              = solver_cfg["dt"],
        horizon         = solver_cfg["horizon_sec"],
        Q=Q, R=R, Qf=Qf,
        u_bounds        = u_bounds,
        x_bounds        = x_bounds,
        minimum_dist    = solver_cfg["safety_dist"],
        linearize_every = solver_cfg["linearize_every"],
        field_every     = solver_cfg["field_every"],
        statefield_idxs = field_idx,
        verbose         = False,
    )
    if cls in (ALILQRSolver, ALDDPSolver):
        al = al_cfg or {}
        kwargs.update(
            rho_init   = al.get("rho_init",   10.0),
            rho_max    = al.get("rho_max",     1e6),
            rho_scale  = al.get("rho_scale",   5.0),
            al_iters   = al.get("al_iters",    20),
            ilqr_iters = al.get("ilqr_iters",  100),
            reg        = al.get("reg",         1e-6),
        )
    return cls(**kwargs)


def step_rk4(dynamics, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    xi, ui = x[np.newaxis, :], u[np.newaxis, :]
    k1 = dynamics.f(xi, ui)[0]
    k2 = dynamics.f(xi + 0.5 * dt * k1, ui)[0]
    k3 = dynamics.f(xi + 0.5 * dt * k2, ui)[0]
    k4 = dynamics.f(xi + dt * k3,        ui)[0]
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def min_obstacle_clearance(pos_2d: np.ndarray, field, safety_dist: float) -> float:
    if field is None or len(field) == 0:
        return np.inf
    try:
        vecs  = field.repulsion_vectors(np.atleast_2d(pos_2d), min_dist_select=True)
        norms = np.linalg.norm(vecs, axis=1)
        valid = norms > 1e-8
        return float(np.min(norms[valid])) - safety_dist if np.any(valid) else np.inf
    except Exception:
        return np.inf


def path_min_clearance(path_xy: np.ndarray, field, safety_dist: float) -> float:
    return min(min_obstacle_clearance(pt, field, safety_dist) for pt in path_xy)


def effective_safety_dist(path_xy: np.ndarray, field, desired: float) -> float:
    raw = path_min_clearance(path_xy, field, 0.0)
    if raw <= 0:
        return 0.0
    return min(desired, raw)


def route_timeout(path: np.ndarray, speed: float,
                  factor: float, t_min: float, t_max: float) -> float:
    if path is None or len(path) < 2:
        return t_min
    length = float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
    return float(np.clip(length / max(speed, 0.1) * factor, t_min, t_max))


def make_x0(
    start_xy: np.ndarray,
    next_xy: np.ndarray,
    altitude: float,
    dynamics,
    layout: dict,
) -> np.ndarray:
    x_idx = layout.get("x_idx",       0)
    y_idx = layout.get("y_idx",       2)
    z_idx = layout.get("z_idx",       4)
    h_idx = layout.get("heading_idx", 8)

    n    = dynamics.first_order_state_n
    base = layout.get("x0_base")
    if base is not None and len(base) == n:
        x0 = np.array(base, dtype=float)
    else:
        x0 = np.zeros(n)

    x0[x_idx] = start_xy[0]
    x0[y_idx] = start_xy[1]
    x0[z_idx] = altitude
    dx, dy    = next_xy[0] - start_xy[0], next_xy[1] - start_xy[1]
    raw_yaw   = np.arctan2(dy, dx) + dynamics.heading_convention_offset
    x0[h_idx] = (raw_yaw + np.pi) % (2 * np.pi) - np.pi
    return x0


def run_route(
    solver,
    dynamics,
    path: np.ndarray,
    stable_state: np.ndarray,
    field,
    x0: np.ndarray,
    goal_xy: np.ndarray,
    altitude: float,
    nominal_speed: float,
    solver_cfg: dict,
    ref_indices: list,
    height_state_idx: int = 4,
    hover_control: np.ndarray = None,
    save_replay: bool = True,
    clearance_threshold: float = 0.1,
    z_lo: float = None,
    z_hi: float = None,
) -> dict:
    dt         = solver_cfg["dt"]
    t_sim      = route_timeout(
        path, nominal_speed,
        solver_cfg.get("t_sim_factor", 3.5),
        solver_cfg.get("t_sim_min", 120.0),
        solver_cfg.get("t_sim_max", 600.0),
    )
    stride      = solver_cfg["stride"]
    goal_tol    = solver_cfg["goal_tol"]
    goal_blend  = solver_cfg["goal_blend"]
    safety_dist = solver.minimum_dist

    if z_lo is None:
        z_lo = -np.inf
    if z_hi is None:
        z_hi = np.inf

    ix, iy     = ref_indices[0], ref_indices[1]
    ref_min_cl = path_min_clearance(path[:, :2], field, safety_dist)

    planner = WaypointPlanner(
        solver=solver, path=path, stable_state=stable_state,
        ref_state_indices=ref_indices, goal_blend_dist=goal_blend,
    )

    m      = dynamics.first_order_control_n
    _hover = np.zeros(m) if hover_control is None else np.asarray(hover_control, dtype=float)
    prev_us = np.tile(_hover, (solver.N, 1))
    x_cur   = x0.copy()

    solve_times, clearances, control_efforts = [], [], []
    all_xs, all_us = [x_cur.copy()], []
    crash_reason   = ""
    goal_reached   = False
    n_fallback     = 0

    for step in range(int(t_sim / dt)):
        if step % stride == 0:
            ref_traj = planner.get_ref(x_cur, nominal_speed=nominal_speed)
            t0 = time.time()
            try:
                xs_pred, us = solver.solve(x_cur, ref_traj, us_init=prev_us)
            except Exception as exc:
                xs_pred = np.tile(x_cur, (solver.N + 1, 1))
                us      = np.tile(_hover, (solver.N, 1))
                n_fallback += 1
                if not crash_reason:
                    crash_reason = f"SolverError@step{step}: {type(exc).__name__}"
            solve_times.append(time.time() - t0)
            prev_us = np.vstack([us[1:], us[-1:]])

        u_cur  = us[step % stride]
        x_next = step_rk4(dynamics, x_cur, u_cur, dt)

        if np.any(np.isnan(x_next)) or np.any(np.isinf(x_next)):
            crash_reason = crash_reason or f"NaN/Inf@step{step}"
            break
        if height_state_idx >= 0 and not (z_lo <= x_next[height_state_idx] <= z_hi):
            crash_reason = crash_reason or (
                f"HeightBounds@step{step}(z={x_next[height_state_idx]:.1f})"
            )
            break

        x_cur = x_next
        all_xs.append(x_cur.copy())
        all_us.append(u_cur.copy())
        pos_xy = np.array([x_cur[ix], x_cur[iy]])
        clearances.append(min_obstacle_clearance(pos_xy, field, safety_dist))
        control_efforts.append(float(np.linalg.norm(u_cur)))

        if np.linalg.norm(pos_xy - goal_xy) < goal_tol:
            goal_reached = True
            break

        t_elapsed = (step + 1) * dt
        if t_elapsed >= 3.0 and not crash_reason:
            dist_start = np.linalg.norm(pos_xy - np.array([x0[ix], x0[iy]]))
            if dist_start < 0.5:
                crash_reason = f"StuckAtStart({dist_start:.2f}m in {t_elapsed:.1f}s)"
                break

    xs_full  = np.array(all_xs)
    us_full  = np.array(all_us) if all_us else np.zeros((0, m))
    xy_traj  = xs_full[:, [ix, iy]]
    path_len = float(np.sum(np.linalg.norm(np.diff(xy_traj, axis=0), axis=1)))
    min_cl   = float(np.min(clearances)) if clearances else np.nan
    n_solves = len(solve_times)

    metrics = {
        "Success":           goal_reached,
        "Is Clear":          bool(min_cl >= -clearance_threshold) if not np.isnan(min_cl) else False,
        "Crash Reason":      crash_reason,
        "Avg Solve Time":    float(np.mean(solve_times))  if solve_times else np.nan,
        "Std Solve Time":    float(np.std(solve_times))   if solve_times else np.nan,
        "Min Clearance":     float(np.min(clearances))    if clearances  else np.nan,
        "Ref Min Clearance": float(ref_min_cl),
        "Travel Time":       len(all_xs) * dt,
        "Path Length":       path_len,
        "Converge Rate":     1.0 - n_fallback / n_solves  if n_solves > 0 else np.nan,
        "Control Effort":    float(np.mean(control_efforts)) if control_efforts else np.nan,
        "Steps":             len(all_xs),
    }
    if save_replay:
        metrics["_replay"] = {
            "xs": xs_full, "us": us_full, "path": path, "x0": x0, "goal_xy": goal_xy,
        }
    return metrics


class SimulationTask:
    """One route-run, ready to be submitted to a thread pool."""

    def __init__(
        self,
        city_name: str,
        algo_name: str,
        seg_i: int,
        speed: float,
        route_data,
        altitude: float,
        dynamics,
        field_map: dict,
        solver_cfg: dict,
        cost_cfg: dict,
        replay_dir,
        ref_indices: list,
        height_idx: int,
        heading_idx: int,
        u_hover,
        u_bounds: tuple,
        x_bounds: tuple,
        field_idx: list,
        run_cfg: dict | None = None,
        city_meta: dict | None = None,
        ref_state_base: list | None = None,
    ):
        self.city_name      = city_name
        self.algo_name      = algo_name
        self.seg_i          = seg_i
        self.speed          = speed
        self.route_data     = route_data
        self.altitude       = altitude
        self.dynamics       = dynamics
        self.field_map      = field_map
        self.solver_cfg     = solver_cfg
        self.cost_cfg       = cost_cfg
        self.replay_dir     = replay_dir
        self.ref_indices    = ref_indices
        self.height_idx     = height_idx
        self.heading_idx    = heading_idx
        self.u_hover        = u_hover
        self.u_bounds       = u_bounds
        self.x_bounds       = x_bounds
        self.field_idx      = field_idx
        self.run_cfg        = run_cfg or {}
        self.city_meta      = city_meta or {}
        self.ref_state_base = ref_state_base

    def run(self) -> SimulationResult:
        if self.route_data is None:
            return SimulationResult.failure(
                self.city_name, self.city_name, self.algo_name,
                self.seg_i, self.speed, "NoPath",
            )

        raw_path, x0, goal_xy, eff_sd = self.route_data
        al_cfg       = self.run_cfg.get("al_solver", {})
        cl_threshold = self.run_cfg.get("clearance_threshold", 0.1)
        save_replay  = self.run_cfg.get("save_replay", True) and self.replay_dir is not None

        try:
            solver = build_solver(
                self.algo_name, self.dynamics, self.u_bounds, self.x_bounds,
                self.field_map, self.solver_cfg, self.cost_cfg, self.field_idx,
                al_cfg=al_cfg,
            )
            solver.minimum_dist = eff_sd
        except Exception as e:
            return SimulationResult.failure(
                self.city_name, self.city_name, self.algo_name,
                self.seg_i, self.speed, f"BuildErr:{e}",
            )

        n    = self.dynamics.first_order_state_n
        base = self.ref_state_base
        if base is not None and len(base) == n:
            stable_state = np.array(base, dtype=float)
        else:
            stable_state = np.zeros(n)
        if self.height_idx >= 0:
            stable_state[self.height_idx] = self.altitude
        if self.heading_idx >= 0 and self.heading_idx < len(stable_state):
            stable_state[self.heading_idx] = x0[self.heading_idx]

        # Extract z crash bounds from x_bounds (already altitude-adjusted).
        z_lo = z_hi = None
        z_idx = self.height_idx
        if z_idx >= 0 and len(self.x_bounds[0]) > z_idx:
            z_lo = self.x_bounds[0][z_idx]
            z_hi = self.x_bounds[1][z_idx]

        try:
            metrics = run_route(
                solver, self.dynamics, raw_path, stable_state,
                self.field_map.get("field"), x0, goal_xy, self.altitude,
                self.speed, self.solver_cfg, self.ref_indices,
                self.height_idx, self.u_hover,
                save_replay=save_replay,
                clearance_threshold=cl_threshold,
                z_lo=z_lo, z_hi=z_hi,
            )
        except Exception as e:
            return SimulationResult.failure(
                self.city_name, self.city_name, self.algo_name,
                self.seg_i, self.speed, f"SimErr:{e}",
            )

        replay_npz = ""
        if "_replay" in metrics and self.replay_dir is not None:
            rep  = metrics.pop("_replay")
            slug = (
                f"{self.city_name.replace(' ', '_')}"
                f"_{self.algo_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                f"_{self.seg_i}_v{self.speed:.0f}"
            )
            npz_path = self.replay_dir / f"{slug}.npz"
            np.savez(
                str(npz_path),
                xs=rep["xs"], us=rep["us"], path=rep["path"],
                x0=rep["x0"], goal_xy=rep["goal_xy"],
                nominal_speed=self.speed, dt=self.solver_cfg["dt"],
            )
            replay_npz = str(npz_path)

        return SimulationResult(
            city              = self.city_name,
            scenario          = self.city_name,
            algorithm         = self.algo_name,
            segment           = self.seg_i,
            nominal_speed     = self.speed,
            success           = metrics["Success"],
            is_clear          = metrics["Is Clear"],
            crash_reason      = metrics["Crash Reason"],
            avg_solve_time    = metrics["Avg Solve Time"],
            std_solve_time    = metrics["Std Solve Time"],
            min_clearance     = metrics["Min Clearance"],
            ref_min_clearance = metrics["Ref Min Clearance"],
            travel_time       = metrics["Travel Time"],
            path_length       = metrics["Path Length"],
            converge_rate     = metrics["Converge Rate"],
            control_effort    = metrics["Control Effort"],
            steps             = metrics["Steps"],
            replay_npz        = replay_npz,
            city_meta         = self.city_meta,
        )


def _fmt(vals, p_small=4, p_med=2, p_large=1) -> str:
    if len(vals) == 0:
        return "--"
    mu, sd = vals.mean(), vals.std()
    p = p_small if abs(mu) < 1 else (p_med if abs(mu) < 100 else p_large)
    return f"{mu:.{p}f}±{sd:.3f}"


def print_console_table(df: pd.DataFrame, title: str = "") -> None:
    if df is None or df.empty or "Success" not in df.columns:
        print(f"\n  [{title}] No results.")
        return
    df_ok = df[df["Success"] == True]
    algos = df["Algorithm"].unique()
    cols  = ["Avg Solve Time", "Min Clearance", "Ref Min Clearance", "Travel Time", "Path Length"]
    cw    = 18

    print()
    sep = "-" * (42 + (cw + 3) * len(cols))
    print(sep)
    if title:
        print(f"  {title}")
    print(f"{'Algorithm':<28} {'SR%':>5} | {'CR%':>5} | " + " | ".join(f"{c[:cw]:>{cw}}" for c in cols))
    print(sep)
    for algo in algos:
        sub_ok = df_ok[df_ok["Algorithm"] == algo]
        sr = df[df["Algorithm"] == algo]["Success"].mean() * 100
        cr = df[df["Algorithm"] == algo].get("Is Clear", pd.Series([float("nan")])).mean() * 100
        cells = [_fmt(sub_ok[c].dropna()) if c in sub_ok.columns else "--" for c in cols]
        print(f"{algo:<28} {sr:5.1f} | {cr:5.1f} | " + " | ".join(f"{c:>{cw}}" for c in cells))
    print(sep)


def print_latex_table(df: pd.DataFrame, title: str = "") -> None:
    if df is None or df.empty or "Success" not in df.columns:
        print(f"% [{title}] No results.")
        return
    df_ok = df[df["Success"] == True]
    algos = df["Algorithm"].unique()
    cols  = ["Avg Solve Time", "Min Clearance", "Ref Min Clearance",
             "Travel Time", "Path Length", "Control Effort"]

    print()
    if title:
        print(f"% {title}")
    print(r"\begin{tabular}{l c c c c c c c c}")
    print(r"\hline")
    print(r"Algorithm & SR\% & CR\% & Solve[s] & MinCl[m] & RefCl[m] & TTime[s] & PLen[m] & CtrlEff \\")
    print(r"\hline")
    for algo in algos:
        sub_ok = df_ok[df_ok["Algorithm"] == algo]
        sr = df[df["Algorithm"] == algo]["Success"].mean() * 100
        cr = df[df["Algorithm"] == algo].get("Is Clear", pd.Series([float("nan")])).mean() * 100
        cells = []
        for c in cols:
            vals = sub_ok[c].dropna() if c in sub_ok.columns else pd.Series([], dtype=float)
            if len(vals) == 0:
                cells.append("--")
            else:
                mu, sd = vals.mean(), vals.std()
                p = 4 if abs(mu) < 1 else (2 if abs(mu) < 100 else 1)
                cells.append(f"{mu:.{p}f} $\\pm$ {sd:.3f}" if pd.notna(sd) else f"{mu:.{p}f}")
        print(f"{algo} & {sr:.1f}\\% & {cr:.1f}\\% & " + " & ".join(cells) + r" \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print()


def print_safety_violations(df: pd.DataFrame, title: str = "",
                             clearance_threshold: float = 0.1) -> None:
    if df is None or df.empty or "Min Clearance" not in df.columns:
        return
    id_cols = [c for c in ["City", "Scenario", "Algorithm", "Segment", "Nominal Speed"]
               if c in df.columns]
    viol = df[df["Min Clearance"] < -clearance_threshold][
        id_cols + ["Min Clearance", "Ref Min Clearance", "Success"]
    ]
    print()
    if viol.empty:
        print(f"  [{title}] No safety violations.")
    else:
        print(f"  [{title}] SAFETY VIOLATIONS ({len(viol)} routes):")
        print(viol.to_string(index=False))


def _make_progress(total: int):
    if not _RICH:
        class _Stub:
            def start(self): pass
            def stop(self): pass
            def add_task(self, *a, **kw): return 0
            def advance(self, *a, **kw): pass
            def update(self, *a, **kw): pass
        return _Stub()
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        refresh_per_second=4,
    )


def run_cli(
    config_path: str,
    only: str | None,
    quick: bool,
    no_store: bool = False,
    sanity: bool = False,
) -> None:
    from core.scenario import BenchmarkRunner, VehicleScenario, WMRScenario
    from data.store import open_store

    cfg          = load_config(Path(config_path))
    run_cfg      = cfg.get("run", {})
    output_cfg   = cfg.get("output", {})

    errors = validate_config(cfg)
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return

    vehicle_registry = cfg.get("vehicle_registry", {})
    city_registry    = cfg.get("city_registry", {})

    active = [b for b in cfg.get("benchmark", [])
              if b.get("enabled", True) and (not only or b.get("name") == only)]

    if not active and not sanity:
        print("No enabled benchmarks found.")
        return

    store_path = output_cfg.get("store", "results/benchmark.db")
    store      = None if no_store else open_store(store_path)

    runner    = BenchmarkRunner(run_cfg)
    scenarios: list[tuple[dict, object]] = []

    if sanity:
        wmr_bench = {"name": "WMR Sanity Check", "quick": quick}
        scenarios.append((wmr_bench, WMRScenario(run_cfg, quick=quick)))

    for bench in active:
        if quick:
            bench = {**bench, "quick": True}
        vehicle_name = bench.get("vehicle", "")
        vehicle_cfg  = vehicle_registry.get(vehicle_name, {})
        if not vehicle_cfg:
            print(f"[WARN] Vehicle '{vehicle_name}' not found in vehicles.toml, skipping.")
            continue
        scenarios.append((bench, VehicleScenario(bench, run_cfg, vehicle_cfg, city_registry)))

    total = sum(s.count_runs() for _, s in scenarios)

    store_label = Path(store_path).name if store else "disabled"
    print("=" * 60)
    print("  TRAJECTORY PLANNER BENCHMARK")
    print(f"  Benchmarks : {[s.name for _, s in scenarios]}")
    print(f"  Total runs : {total}")
    print(f"  Workers    : {runner.max_workers}")
    print(f"  Store      : {store_label}")
    print("=" * 60)

    progress = _make_progress(total)
    progress.start()
    overall  = progress.add_task("Benchmark", total=total)

    all_dfs: dict[str, pd.DataFrame] = {}

    for bench, scenario in scenarios:
        dyn_name = bench.get("vehicle", bench.get("dynamics", ""))
        run_id   = None

        try:
            scenario.setup()
        except Exception as exc:
            print(f"[ERROR] Setup for '{scenario.name}' failed: {exc}")
            traceback.print_exc()
            continue

        if store is not None:
            run_id = store.create_run(
                benchmark_name = scenario.name,
                dynamics       = dyn_name,
                quick          = quick or bench.get("quick", False),
                bench_cfg      = bench,
            )

        def on_result(res, _store=store, _run_id=run_id):
            if _store is not None and _run_id is not None:
                _store.add_result(_run_id, res)

        try:
            results = runner.run_scenario(scenario, progress, overall, on_result=on_result)
        except KeyboardInterrupt:
            progress.stop()
            print(f"\n[INTERRUPTED] Results saved so far are in the store (run_id: {run_id})")
            return
        except Exception as exc:
            print(f"[ERROR] Scenario '{scenario.name}' failed: {exc}")
            traceback.print_exc()
            continue

        df = pd.DataFrame([r.to_record() for r in results])
        all_dfs[scenario.name] = df

        if run_id:
            print(f"\nSaved to store: {store_path}  (run_id: {run_id})")

    progress.stop()

    cl_threshold = run_cfg.get("clearance_threshold", 0.1)

    print("\n\n" + "=" * 60 + "\n  RESULTS SUMMARY\n" + "=" * 60)
    for name, df in all_dfs.items():
        print_console_table(df, title=name)

    print("\n\n=== SAFETY VIOLATIONS ===")
    for name, df in all_dfs.items():
        print_safety_violations(df, title=name, clearance_threshold=cl_threshold)

    print("\n\n% ===== LaTeX Tables =====\n")
    for name, df in all_dfs.items():
        print_latex_table(df, title=name)


if __name__ == "__main__":
    print("Please use the unified CLI: python cli.py run")
