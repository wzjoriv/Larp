"""
Replay a saved benchmark flight from the database.

Usage:
  python cli.py replay --list
  python cli.py replay <result_id>
  python cli.py replay <result_id> --speed 2.0
  python cli.py replay <result_id> --save replay.mp4

Replay files (NPZ) are stored at the path recorded in the database.
City and vehicle metadata needed by the visualizer are also read from
the database, not from filenames.
"""

import os
from pathlib import Path

os.environ["JAX_PLATFORMS"]        = "cpu"
os.environ["OSQP_ALGEBRA_BACKEND"] = "builtin"

import json
import numpy as np

from data.config import load_config


N_PRED = 30


def _load_vehicle_physics(vehicle_name: str, cfg: dict) -> dict:
    """Return the physics dict for a named vehicle from the loaded config."""
    registry = cfg.get("vehicle_registry", {})
    vehicle  = registry.get(vehicle_name, {})
    return vehicle.get("physics", {})


def list_replay_results(store) -> None:
    df = store.list_replay_results()
    if df.empty:
        print("No replay files found in the database.")
        return
    print(f"\n{'ID':>6}  {'Run ID':<22}  {'City':<14}  {'Algorithm':<28}  {'Seg':>3}  {'Speed':>7}  {'OK'}")
    print("-" * 100)
    for _, row in df.iterrows():
        spd = f"{row['nominal_speed']:.0f} m/s" if row["nominal_speed"] else "  --"
        ok  = "yes" if row["success"] else "no"
        print(f"{int(row['id']):>6}  {row['run_id']:<22}  {str(row['city']):<14}  "
              f"{str(row['algorithm']):<28}  {int(row['segment']):>3}  {spd:>7}  {ok}")


def replay(result_id: int, config_path: str, speed_mult: float = 1.0, save_path: str = None):
    import matplotlib
    from data.store import open_store
    from data.config import load_config

    cfg        = load_config(Path(config_path))
    store_path = cfg.get("output", {}).get("store", "results/benchmark.db")
    store      = open_store(store_path)

    row = store.get_result(result_id)
    if row is None:
        print(f"Error: result id {result_id} not found in database.")
        return

    npz_path = row.get("replay_npz", "")
    if not npz_path or not Path(npz_path).exists():
        print(f"Error: replay file not found: '{npz_path}'")
        return

    city_meta_raw = row.get("city_meta", "")
    city_meta: dict = json.loads(city_meta_raw) if city_meta_raw else {}

    data = np.load(str(npz_path))
    xs   = data["xs"]
    us   = data["us"]
    path = data["path"]
    dt   = float(data["dt"])
    T    = len(xs) - 1

    algo    = row.get("algorithm", "")
    city    = row.get("city", "")
    seg     = row.get("segment", 0)
    spd_lbl = f"v={row['nominal_speed']:.0f}m/s" if row.get("nominal_speed") else ""
    print(f"\nReplaying result #{result_id}: {city} | {algo} | seg {seg} {spd_lbl}")
    print(f"  Steps: {T}  |  Duration: {T*dt:.1f}s  |  dt: {dt}s")

    if not city_meta:
        print("  [WARN] No city metadata in database for this result; visualizer may be incomplete.")

    location = city_meta.get("location", [0.0, 0.0])
    altitude = city_meta.get("altitude", 50.0)
    dist     = city_meta.get("dist",     500.0)

    # Find vehicle physics from vehicles.toml via the benchmark that produced this run
    dynamics_name = ""
    from data.store import BenchmarkStore
    if isinstance(store, BenchmarkStore):
        with store._conn() as conn:
            run_row = conn.execute(
                "SELECT dynamics FROM runs WHERE run_id = ?", (row["run_id"],)
            ).fetchone()
            if run_row:
                dynamics_name = run_row["dynamics"]
    else:
        # CSV store: dynamics stored per-row
        dynamics_name = row.get("dynamics", "")

    vehicle_physics: dict = {}
    vehicle_registry = cfg.get("vehicle_registry", {})
    for v in vehicle_registry.values():
        if v.get("dynamics") == dynamics_name or v.get("name") == dynamics_name:
            vehicle_physics = v.get("physics", {})
            break

    if save_path:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from larp.environment import QuadcopterPerspective, ZoomedCityVisualizer

    robot = QuadcopterPerspective(vehicle_physics, projection="perspective", color="black")
    viz   = ZoomedCityVisualizer(
        robot,
        location          = location,
        altitude          = altitude,
        dist              = dist,
        view_range        = 200.0,
        zoomed_view_range = 10.0,
        feasible_region   = False,
        safety_margin     = 10.0,
    )

    ax_main = viz.fig.axes[0] if viz.fig.axes else None
    if ax_main is not None and len(path) > 1:
        ax_main.plot(path[:, 0], path[:, 1], "--", color="cyan",
                     linewidth=1.2, alpha=0.7, label="ref path", zorder=3)

    def _frame(k: int):
        viz.update(
            k * dt,
            xs[k],
            us[min(k, T - 1)],
            xs[k: k + N_PRED + 1],
            xs[: k + 1],
            ref_traj=None,
            A_constraint=None,
            B_constraint=None,
        )

    if save_path:
        fps    = 1.0 / dt
        writer = FFMpegWriter(fps=fps, metadata={"title": f"Replay {city}"},
                              extra_args=["-vcodec", "libx264"])
        plt.ioff()
        with writer.saving(viz.fig, save_path, dpi=100):
            for k in range(T + 1):
                _frame(k)
                writer.grab_frame()
                if (k + 1) % 50 == 0:
                    print(f"  Encoding {(k+1)/(T+1)*100:.0f}%...", end="\r")
        print(f"\nSaved: {save_path}")
    else:
        pause = dt / max(speed_mult, 1e-3)
        plt.ion()
        for k in range(T + 1):
            _frame(k)
            plt.pause(max(pause, 0.001))
        plt.ioff()
        plt.show()


def replay_cli(
    result_id: int | None,
    config_path: str,
    speed: float | None,
    save: str | None,
    list_files: bool,
):
    from data.config import load_config
    from data.store import open_store

    cfg        = load_config(Path(config_path))
    store_path = cfg.get("output", {}).get("store", "results/benchmark.db")

    if not Path(store_path).exists():
        print(f"No store found at '{store_path}'. Run a benchmark first.")
        return

    store      = open_store(store_path)
    speed_mult = speed if speed is not None else 1.0

    if list_files:
        list_replay_results(store)
        return

    if result_id is None:
        print("Error: provide a result ID or use --list to see available replays.")
        return

    replay(result_id, config_path, speed_mult=speed_mult, save_path=save)


if __name__ == "__main__":
    print("Please use the unified CLI: python cli.py replay")
