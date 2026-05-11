"""UAV simulation using configuration from larp.toml."""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OSQP_ALGEBRA_BACKEND", "builtin")

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import larp
import larp.environment as _env_module
import larp.pp as pp
from larp.rug_utils.progress_bar import progress_bar
from larp.tp.planner import LinearPlanner
from larp.tp.solver import SQPSolver, ALILQRSolver, ALDDPSolver
from larp.environment import ZoomedCityVisualizer

_SOLVER_REGISTRY: dict[str, type] = {
    "SQP":  SQPSolver,
    "iLQR": ALILQRSolver,
    "DDP":  ALDDPSolver,
}


def _resolve_x_bounds(cfg: dict, altitude: float) -> tuple[list, list]:
    ob    = cfg["optimizer"]["x_bounds"]
    z_idx = ob.get("z_idx", 4)
    lo    = list(ob["lo"])
    hi    = list(ob["hi"])
    lo[z_idx] = altitude - cfg["city"]["z_lo_margin"]
    hi[z_idx] = altitude + cfg["city"]["z_hi_margin"]
    return lo, hi


def _resolve_u_bounds(cfg: dict, w_hover: float) -> tuple[list, list]:
    ub   = cfg["optimizer"]["u_bounds"]
    u_lo = list(ub["lo"])
    u_hi = [w_hover * ub["u_hi_hover_ratio"]] * len(u_lo)
    return u_lo, u_hi


def _build_solver(cfg: dict, dynamics, field, u_bounds: tuple, x_bounds: tuple):
    algo = cfg["optimizer"]["algorithm"]
    cls  = _SOLVER_REGISTRY[algo]

    n, m  = dynamics.first_order_state_n, dynamics.first_order_control_n
    Q_diag = cfg["optimizer"]["Q_diag"]
    Q  = np.diag(Q_diag)
    Qf = Q * cfg["optimizer"]["Qf_scale"]
    R  = np.eye(m) * cfg["optimizer"]["R_scale"]

    kwargs = dict(
        field           = field,
        dynamics        = dynamics,
        dt              = cfg["sim"]["dt"],
        horizon         = cfg["optimizer"]["horizon"],
        Q=Q, R=R, Qf=Qf,
        u_bounds        = u_bounds,
        x_bounds        = x_bounds,
        minimum_dist    = cfg["optimizer"]["safety_distance"],
        linearize_every = cfg["optimizer"]["linearize_every"],
        field_every     = cfg["optimizer"]["field_every"],
        statefield_idxs = cfg["planning"]["ref_indices"][:2],
        verbose         = False,
    )
    if cls in (ALILQRSolver, ALDDPSolver):
        al = cfg["optimizer"].get("al_params", {})
        kwargs.update(
            rho_init   = al.get("rho_init",   10.0),
            rho_max    = al.get("rho_max",     1e6),
            rho_scale  = al.get("rho_scale",   5.0),
            al_iters   = al.get("al_iters",    20),
            ilqr_iters = al.get("ilqr_iters",  100),
            reg        = al.get("reg",         1e-6),
        )
    return cls(**kwargs)


@contextmanager
def _video_writer(fig, filename: str, fps: float, dpi: int, save: bool):
    if save:
        print(f"VIDEO MODE → {filename}")
        writer = FFMpegWriter(
            fps=fps,
            metadata=dict(title="LARP Demo"),
            extra_args=["-vcodec", "libx264"],
        )
        with writer.saving(fig, filename, dpi=dpi):
            yield writer
    else:
        print("INTERACTIVE MODE — live plot")
        yield None


def simulate(cfg: dict) -> list[np.ndarray]:
    """Run the full UAV simulation loop.

    Parameters
    ----------
    cfg : dict
        Parsed larp.toml configuration.

    Returns
    -------
    list of np.ndarray
        State trajectory [x0, x1, ..., xT].
    """
    dt       = cfg["sim"]["dt"]
    city     = cfg["city"]
    altitude = city["altitude"]
    save_video = cfg["vis"]["save_video"]

    print(f"\n{'='*45}")
    print(f"  INITIALISING  (dt={dt} s  algo={cfg['optimizer']['algorithm']})")
    print(f"{'='*45}")

    plt.ion() if not save_video else plt.ioff()

    # [vehicle]
    vcfg        = cfg["vehicle"]
    perspective = getattr(_env_module, vcfg["perspective"])
    robot       = perspective(vcfg["physics"], projection=vcfg["projection"], color=vcfg["color"])
    dyn         = robot.dynamics

    w_hover   = float(np.sqrt(dyn.m * dyn.g / (4.0 * dyn.kf)))
    rpm_hover = w_hover * 60.0 / (2.0 * np.pi)
    print(f"\nVehicle ({vcfg['perspective']}) — mass={dyn.m} kg  "
          f"hover ω={w_hover:.1f} rad/s ({rpm_hover:.0f} RPM)")

    x_eq = np.zeros((1, dyn.first_order_state_n)); x_eq[0, 4] = 10.0
    u_eq = np.full((1, dyn.first_order_control_n), w_hover)
    print(f"  Hover accel_z = {dyn.f(x_eq, u_eq)[0, 5]:.6f} m/s²  (expect ≈ 0)")

    # [environment]
    print("\nBuilding environment …")
    viz = ZoomedCityVisualizer(
        robot,
        location          = tuple(city["location"]),
        altitude          = altitude - city["z_lo_margin"],
        dist              = city["dist"],
        default_repulsion = city.get("default_repulsion"),
        view_range        = cfg["vis"]["view_range"],
        zoomed_view_range = cfg["vis"]["zoomed_view_range"],
        feasible_region   = False,
    )
    field = larp.RiskField(
        rgjs=viz.get_obstacle_rgjs(),
        center_point=(0, 0),
        size=city["dist"] * 2,
    )
    print(f"Number of obstacles: {len(field)}")

    dist     = city["dist"]
    quadtree = larp.quad.QuadTree(
        field,
        minimum_length_limit=dist / cfg["planning"]["min_quad_size_divisor"],
    )
    if cfg["optimizer"]["use_risk_field"]:
        search = larp.quad.QuadTree(
            field,
            minimum_length_limit=field.size[0] / cfg["planning"]["search_quad_divisor"],
            edge_bounds=cfg["planning"]["search_edge_bounds"],
        )
        opt_field = larp.quad.QRiskField(search)
    else:
        opt_field = field

    # [jax warm-up]
    stable_state = np.array(cfg["sim"]["stable_state"])
    t0 = time.time()
    dyn.discretize(stable_state[None, :], np.zeros((1, dyn.first_order_control_n)), dt=dt)
    print(f"JAX JIT warm-up: {time.time() - t0:.3f} s")

    # [path planning]
    x0          = np.array(cfg["sim"]["x0"])
    xf          = np.array(cfg["sim"]["xf"])
    ref_indices = cfg["planning"]["ref_indices"]
    start_xy    = x0[ref_indices[:2]]
    goal_xy     = xf[ref_indices[:2]]
    print(f"\nPath planning: {start_xy} → {goal_xy}")

    path = pp.QuadPlanner(quadtree).find_path(start_xy, goal_xy)
    if path is None:
        raise RuntimeError("QuadPlanner found no path. Check start/goal or map.")

    # [optimizer]
    x_bounds = _resolve_x_bounds(cfg, altitude)
    u_bounds = _resolve_u_bounds(cfg, w_hover)
    solver   = _build_solver(cfg, dyn, opt_field, u_bounds, x_bounds)
    print(f"\nSolver  N={solver.N}  safety={cfg['optimizer']['safety_distance']} m")

    traj_planner = LinearPlanner(
        solver            = solver,
        path              = path,
        stable_state      = stable_state,
        ref_state_indices = ref_indices,
        goal_blend_dist   = cfg["planning"]["goal_blend_dist"],
    )

    # [simulation loop]
    save_video = cfg["vis"]["save_video"]
    plt.ion() if not save_video else plt.ioff()

    x_cur   = x0.copy()
    us_prev = np.full((solver.N, dyn.first_order_control_n), w_hover)
    traj    = [x_cur.copy()]
    T_steps = int(cfg["sim"]["T_sim"] / dt)
    fps     = 1.0 / (dt * cfg["vis"]["render_every"])

    nominal_speed    = cfg["planning"]["nominal_speed"]
    height_sim_bound = cfg["sim"]["height_sim_bound"]
    z_idx            = cfg["optimizer"]["x_bounds"].get("z_idx", 4)

    print(f"\nRunning {T_steps} steps …")
    progress_bar(0.0)

    with _video_writer(viz.fig, cfg["vis"]["output_video"], fps,
                       cfg["vis"]["dpi"], save_video) as writer:
        for k in range(T_steps):
            ref = traj_planner.get_ref(x_cur, nominal_speed=nominal_speed)
            xs_pred, us = solver.solve(x_cur, ref, us_prev)

            u_cur   = us[0]
            us_prev = np.vstack([us[1:], us[-1:]])

            Ad, Bd, gd = dyn.discretize(x_cur[None, :], u_cur[None, :], dt, estimate=False)
            x_next = Ad[0] @ x_cur + Bd[0] @ u_cur + gd[0]

            z = x_next[z_idx]
            if np.any(np.isnan(x_next)) or z < 0 or z > height_sim_bound:
                raise RuntimeError(f"Simulation unstable at step {k}: z={z:.2f} m")

            if k % cfg["vis"]["render_every"] == 0:
                A_c, B_c = solver.get_field_constraints(x_cur)
                viz.update(
                    (k + 1) * dt, x_cur, u_cur, xs_pred,
                    np.array(traj), ref_traj=ref,
                    A_constraint=A_c, B_constraint=B_c,
                )
                if writer is not None:
                    writer.grab_frame()

            x_cur = x_next
            traj.append(x_cur.copy())
            progress_bar((k + 1) / T_steps)

    #plt.savefig(cfg["vis"]["output_pdf"])
    if not save_video:
        plt.ioff()
        plt.show()

    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description="LARP UAV simulation")
    parser.add_argument("--config", default="larp.toml", help="Path to larp.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as fh:
        cfg = tomllib.load(fh)

    simulate(cfg)


if __name__ == "__main__":
    main()
