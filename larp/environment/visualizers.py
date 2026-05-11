"""
larp/env/visualizers.py
=======================
Visualiser classes with blit rendering and interactive camera controls.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.spatial import HalfspaceIntersection

from larp.environment.twin import Robot
from larp.environment.environments import CityEnvironment, Environment, FieldHeatmapEnvironment
VIZ_LINE_ZORDER = 10

class _FeasibilityOverlay:
    def __init__(self, ax, robot: Robot, view_range: float, update_hz: float = 5.0):
        self._ax = ax
        self._robot = robot
        self._vrange = view_range
        self._period = max(1, round(100 / update_hz))
        self._frame = 0
        self._poly = mpatches.Polygon(
            np.empty((0, 2)), closed=True,
            facecolor="#20B2AA", edgecolor="#008B8B",
            alpha=0.2, label="Feasible Region", zorder=50,
        )
        ax.add_patch(self._poly)

    @property
    def patch(self) -> mpatches.Polygon:
        return self._poly

    def update(self, A, b, pos: Tuple[float, float]) -> None:
        self._frame += 1
        if A is None or b is None:
            self._poly.set_visible(False)
            return
        if self._frame % self._period != 0:
            return
        ix, iy = self._robot.pos_indices
        A2 = A[:, [ix, iy]]
        px, py = pos
        R = self._vrange
        bb = np.array([[1, 0, px + R], [-1, 0, -(px - R)], [0, 1, py + R], [0, -1, -(py - R)]])
        hs = np.vstack([
            np.hstack([bb[:, :2], -bb[:, 2:3]]),
            np.hstack([A2, -b.reshape(-1, 1)]),
        ])
        
        try:
            hi = HalfspaceIntersection(hs, np.array([px, py]))
            v = hi.intersections
            ctr = np.mean(v, axis=0)
            ang = np.arctan2(v[:, 1] - ctr[1], v[:, 0] - ctr[0])
            self._poly.set_xy(v[np.argsort(ang)])
            self._poly.set_visible(True)
        except Exception:
            self._poly.set_visible(False)


class Visualizer:
    """
    Base visualiser: figure + axis + robot painter + blit engine.

    Parameters
    ----------
    robot       : Robot
    view_range  : float   Camera half-width [m].
    use_blit    : bool    Enable blit rendering (default True).
    figsize     : tuple   Matplotlib figure size.
    ax          : plt.Axes, optional
        If given, render into this existing axis instead of creating a new
        figure.  The caller is responsible for the figure layout.
    """

    def __init__(self, robot: Robot, view_range: float = 15.0,
                 use_blit: bool = True, figsize: Tuple = (10, 10),
                 ax: Optional[plt.Axes] = None):
        self.robot = robot
        self.view_range = view_range
        self.use_blit = use_blit
        self._follow = True

        if ax is not None:
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize)

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")

        self.robot_painter = self.robot.create_painter(self.ax)
        self._blit_bg = None
        self._cam_pos = (None, None)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key == "t":
            self._follow = not self._follow
            print(f"[Visualizer] Camera follow: {'ON' if self._follow else 'OFF'}")
            self.invalidate_background()

    def _capture_bg(self):
        """Rasterise current figure state into the background cache."""
        self.fig.canvas.draw()
        self._blit_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _render_frame(self):
        """Restore cached background, draw robot on top, flush to screen."""
        if self.use_blit and self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            for a in self.robot.robot_artists(self.robot_painter):
                self.ax.draw_artist(a)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def invalidate_background(self):
        """Force a full background redraw on the next frame."""
        self._blit_bg = None

    def _move_camera(self, px: float, py: float, ax: Optional[plt.Axes] = None,
                     vrange: Optional[float] = None) -> bool:
        """
        Reposition the camera when the robot travels beyond the threshold.
        Returns True when the camera actually moved (background must be re-cached).
        Does nothing when camera follow is disabled.
        """
        if not self._follow or self._cam_pos == (px, py):
            return False
        ax = ax or self.ax
        vrange = vrange or self.view_range
        ax.set_xlim(px - vrange, px + vrange)
        ax.set_ylim(py - vrange, py + vrange)
        self._cam_pos = (px, py)
        return True

    def _setup_trajectory_lines(self, ax, hide_legend=False):
        lbl = "_" if hide_legend else ""
        traj, = ax.plot([], [], "-.", color="#3F8CD8", alpha=0.8, markersize=2, label=f"{lbl}Trajectory", zorder=VIZ_LINE_ZORDER)
        ref, = ax.plot([], [], "--", color="#3FB33F", lw=1.0, alpha=0.7, label=f"{lbl}Reference", zorder=VIZ_LINE_ZORDER)
        pred, = ax.plot([], [], "--", color="#1a237e", lw=1.5, alpha=0.95, label=f"{lbl}Predicted", zorder=VIZ_LINE_ZORDER)
        return traj, ref, pred

    def _update_trajectory_lines(self, lines, x_cur, traj_history, xs_pred, ref_traj):
        traj, ref, pred = lines
        ix, iy = self.robot.pos_indices
        if traj_history is not None and len(traj_history):
            traj.set_data(traj_history[:, ix], traj_history[:, iy])
        if xs_pred is not None:
            pred.set_data(xs_pred[:, ix], xs_pred[:, iy])
        if ref_traj is not None:
            ref.set_data(ref_traj[:, ix], ref_traj[:, iy])

    def update(self, t: float, x_cur: np.ndarray, u_cur: np.ndarray, **kwargs):
        self.robot.update_painter(self.robot_painter, t, x_cur, u_cur, **kwargs)
        ix, iy = self.robot.pos_indices
        moved = self._move_camera(x_cur[ix], x_cur[iy])
        if moved or (self.use_blit and self._blit_bg is None):
            self._capture_bg()
        self._render_frame()


class FieldTrajectoryVisualizer(Visualizer):
    """Risk-field heatmap with trajectory / prediction lines and HUD."""

    def __init__(self, field, robot: Robot, view_range: float = 15.0,
                 use_blit: bool = True, ax: Optional[plt.Axes] = None):
        super().__init__(robot, view_range, use_blit, ax=ax)
        self._env = FieldHeatmapEnvironment(field)
        self._env.draw(self.ax)
        self.traj_line, = self.ax.plot([], [], "ow",  markersize=2,  label="Trajectory")
        self.pred_line, = self.ax.plot([], [], "--w",  alpha=0.5,     label="Predicted")
        self.ref_line,  = self.ax.plot([], [], "--y",  lw=1.5,        label="Reference")
        self.hud = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )
        self.ax.legend(loc="upper right")

    def update(self, t, x_cur, u_cur, xs_pred=None, traj_history=None, ref_traj=None, **kwargs):
        ref = ref_traj[0] if ref_traj is not None and len(ref_traj) else None
        self.robot.update_painter(self.robot_painter, t, x_cur, u_cur, ref=ref)
        
        ix, iy = self.robot.pos_indices
        if traj_history is not None and len(traj_history):
            self.traj_line.set_data(traj_history[:, ix], traj_history[:, iy])
        if xs_pred is not None:
            self.pred_line.set_data(xs_pred[:, ix], xs_pred[:, iy])
        if ref_traj is not None:
            self.ref_line.set_data(ref_traj[:, ix], ref_traj[:, iy])
            
        self.hud.set_text(f"t={t:.2f}s\n{self.robot.get_telemetry(x_cur, u_cur)}")
        moved = self._move_camera(x_cur[ix], x_cur[iy])
        if moved or self._blit_bg is None:
            self._capture_bg()
        self._render_frame()


class CityVisualizer(Visualizer):
    """
    Single-axis OSM city visualiser.

    Parameters
    ----------
    robot           : Robot
    env             : CityEnvironment  Pre-built environment (recommended).
    view_range      : float
    use_blit        : bool     Default True (~228x faster than canvas.draw).
    feasible_region : bool     Draw MPC feasible-region polygon.
    feasibility_hz  : float    Recompute polygon frequency (default 5 Hz).
    ax              : plt.Axes, optional  Inject an existing axis.
    **env_kwargs    : Passed to CityEnvironment when *env* is None (compat shim).
    """

    def __init__(self, robot: Robot, env: Optional[CityEnvironment] = None,
                 view_range: float = 20.0, use_blit: bool = True,
                 feasible_region: bool = False, feasibility_hz: float = 5.0,
                 ax: Optional[plt.Axes] = None, **env_kwargs):
        super().__init__(robot, view_range, use_blit, ax=ax)
        self.env = env if env is not None else CityEnvironment(**env_kwargs)

        self.ax.set_xlabel("West-East [m]")
        self.ax.set_ylabel("North-South [m]")
        self.ax.grid(True, linestyle="--", alpha=0.1)
        self.ax.set_title(f"Location: {self.env.location_query}")
        self._layers = self.env.draw(self.ax)

        self._feasibility = _FeasibilityOverlay(self.ax, robot, view_range, feasibility_hz) if feasible_region else None
        self.lines = self._setup_trajectory_lines(self.ax)
        
        self.hud = self.ax.text(
            0.02, 0.98, "Initialising...", transform=self.ax.transAxes,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=1.0), zorder=90000,
        )
        self.ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white").set_zorder(89999)
        self._capture_bg()

    def get_obstacle_rgjs(self) -> list:
        return self.env.get_obstacle_rgjs()

    def update(self, t, x_cur, u_cur, xs_pred=None, traj_history=None,
               ref_traj=None, A_constraint=None, B_constraint=None, **kwargs):
        ix, iy = self.robot.pos_indices
        px, py = x_cur[ix], x_cur[iy]
        ref = ref_traj[0] if ref_traj is not None and len(ref_traj) else None
        
        self.robot.update_painter(self.robot_painter, t, x_cur, u_cur, ref=ref)

        if self._feasibility:
            self._feasibility.update(A_constraint, B_constraint, (px, py))

        self._update_trajectory_lines(self.lines, x_cur, traj_history, xs_pred, ref_traj)
        
        self.hud.set_text(
            f"t={t:.2f}s \n"
            f"----------\n"
            f"{self.robot.get_telemetry(x_cur, u_cur)}"
        )

        moved = self._move_camera(px, py)
        if moved or self._blit_bg is None:
            self._capture_bg()
        self._render_frame()


class ZoomedCityVisualizer(Visualizer):
    """
    OSM city visualiser with picture-in-picture zoom inset.

    Both main and zoom cameras update together whenever the robot travels
    beyond their respective thresholds.  The blit cache covers the full figure
    (both axes), so robot artists on both axes are redrawn each frame.

    Parameters
    ----------
    robot             : Robot
    env               : CityEnvironment  (or pass **env_kwargs)
    view_range        : float  Main camera half-width [m].
    zoomed_view_range : float  Inset camera half-width [m].
    use_blit          : bool   Default True.
    feasible_region   : bool
    feasibility_hz    : float  Default 5 Hz.
    ax                : plt.Axes, optional  Inject an existing axis for the main panel.
    **env_kwargs      : Forwarded to CityEnvironment if *env* is None.
    """

    def __init__(self, robot: Robot, env: Optional[CityEnvironment] = None,
                 view_range: float = 50.0, zoomed_view_range: float = 10.0,
                 use_blit: bool = True, feasible_region: bool = False,
                 feasibility_hz: float = 5.0, ax: Optional[plt.Axes] = None, **env_kwargs):
        super().__init__(robot, view_range, use_blit, ax=ax)
        self.env = env if env is not None else CityEnvironment(**env_kwargs)
        self.zoom_range = zoomed_view_range
        self._zoom_cam = (None, None)

        self.ax_zoom = inset_axes(self.ax, width="30%", height="30%", loc="lower right")
        self.ax_zoom.patch.set_alpha(0.90)
        self.ax_zoom.set_aspect("equal")
        self.ax_zoom.grid(True, linestyle="--", alpha=0.3)
        self.ax_zoom.set_xticks([]); self.ax_zoom.set_yticks([])
        
        self.ax.set_xlabel("West-East [m]")
        self.ax.set_ylabel("North-South [m]")
        self.ax.set_title(f"Location: {self.env.location_query}")
        self._zoom_connector = mark_inset(
            self.ax, self.ax_zoom, loc1=1, loc2=3, fc="none", ec="0.5", alpha=0.5, zorder=300,
        )

        self.robot_painter_main = self.robot_painter
        self.robot_painter_zoom = self.robot.create_painter(self.ax_zoom)
        self._feasibility = _FeasibilityOverlay(self.ax, robot, view_range, feasibility_hz) if feasible_region else None

        self.ax.set_facecolor("#f0ede8")
        self.ax_zoom.set_facecolor("#f0ede8")
        self.env.draw(self.ax)
        self.env.draw(self.ax_zoom)

        self.lines_main = self._setup_trajectory_lines(self.ax)
        self.lines_zoom = self._setup_trajectory_lines(self.ax_zoom, hide_legend=True)

        self.hud = self.ax.text(
            0.02, 0.98, "Initialising...", transform=self.ax.transAxes,
            va="top", family="monospace", bbox=dict(boxstyle="round", fc="white", alpha=1.0), zorder=90000,
        )
        self.ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white").set_zorder(89999)
        self._capture_bg()

    def _capture_bg(self):
        """Redraw full figure (both axes) and cache."""
        self.fig.canvas.draw()
        self._blit_bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _render_frame(self):
        if self.use_blit and self._blit_bg is not None:
            self.fig.canvas.restore_region(self._blit_bg)
            for a in self.robot.robot_artists(self.robot_painter_main):
                self.ax.draw_artist(a)
            for a in self.robot.robot_artists(self.robot_painter_zoom):
                self.ax_zoom.draw_artist(a)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def get_obstacle_rgjs(self) -> list:
        return self.env.get_obstacle_rgjs()

    def update(self, t, x_cur, u_cur, xs_pred=None, traj_history=None,
               ref_traj=None, A_constraint=None, B_constraint=None, **kwargs):
        
        ix, iy = self.robot.pos_indices
        px, py = x_cur[ix], x_cur[iy]
        ref = ref_traj[0] if ref_traj is not None and len(ref_traj) else None
        
        self.robot.update_painter(self.robot_painter_main, t, x_cur, u_cur, ref=ref)
        self.robot.update_painter(self.robot_painter_zoom, t, x_cur, u_cur, ref=ref)

        if self._feasibility:
            self._feasibility.update(A_constraint, B_constraint, (px, py))

        self._update_trajectory_lines(self.lines_main, x_cur, traj_history, xs_pred, ref_traj)
        self._update_trajectory_lines(self.lines_zoom, x_cur, traj_history, xs_pred, ref_traj)

        self.hud.set_text(
            f"t={t:.2f}s\n"
            f"----------\n"
            f"{self.robot.get_telemetry(x_cur, u_cur)}"
        )

        main_moved = self._move_camera(px, py)

        # Zoom camera ALWAYS follows, perfectly centered
        zoom_moved = False
        if self._zoom_cam != (px, py):
            self.ax_zoom.set_xlim(px - self.zoom_range, px + self.zoom_range)
            self.ax_zoom.set_ylim(py - self.zoom_range, py + self.zoom_range)
            self._zoom_cam = (px, py)
            zoom_moved = True

        if main_moved or zoom_moved or self._blit_bg is None:
            self._capture_bg()
        self._render_frame()