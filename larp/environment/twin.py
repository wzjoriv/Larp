"""
larp/env/robots.py
==================
Robot ABC + WMR, Car, Quadcopter, QuadcopterPerspective, FixedWing.

Rendering notes
---------------
* base_circle cached in __init__ — no linspace/cos/sin per frame.
* FancyArrow.set_data() instead of remove+recreate each frame (20x faster).
* All heading/ref/velocity arrows are drawn *outside* the body so they never
  overlap the drone silhouette.  They pivot around the body centre but start
  at ``pivot_r`` from it (set per robot to ~propeller radius).
* Velocity arrow length scales with magnitude (not constant), so zero speed
  hides it cleanly.
* robot_artists() exposes exactly the dynamic artists for blit support.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from larp.dynamics import CarDynamics, Dynamics, QuadcopterDynamics, WMRDynamics, SmallFixedWingDynamics

def hex_to_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def interpolate_colors(c1, c2, t):
    return (1.0 - t) * c1 + t * c2


class Robot(ABC):
    """
    Abstract base for all robots.

    Attributes
    ----------
    dynamics    : Dynamics
    pos_indices : [x_idx, y_idx] in the state vector for 2-D plotting
    heading_offset : float  Visual-only offset for the reference-heading arrow.
    """

    def __init__(self, dynamics: Dynamics) -> None:
        self.dynamics = dynamics

    @property
    @abstractmethod
    def pos_indices(self) -> List[int]: ...

    @property
    def heading_offset(self) -> float:
        return 0.0

    @abstractmethod
    def get_telemetry(self, state: np.ndarray, control: np.ndarray) -> str: ...
    @abstractmethod
    def create_painter(self, ax: plt.Axes) -> Dict: ...
    @abstractmethod
    def update_painter(self, painter: Dict, time: float,
                       x_state: np.ndarray, u_control: np.ndarray,
                       **kwargs) -> None: ...

    def robot_artists(self, painter: Dict) -> List:
        """Return flat list of artists that change every frame (for blit support)."""
        return list(painter.values())


def _set_arrow_outside(arrow: patches.FancyArrow,
                       px: float, py: float,
                       dx: float, dy: float,
                       pivot_r: float,
                       scale: float = 1.0) -> None:
    """
    Reposition arrow to start at pivot_r from (px, py) in the
    direction (dx, dy) and extend scale units further.
    """
    d = np.hypot(dx, dy)
    if d < 1e-6:
        arrow.set_visible(False)
        return
    arrow.set_visible(True)
    nx, ny = dx / d, dy / d
    # start point: on body perimeter
    sx, sy = px + nx * pivot_r, py + ny * pivot_r
    arrow.set_data(x=sx, y=sy, dx=nx * scale, dy=ny * scale)


class WMR(Robot):
    """Differential-drive robot. State: [x, y, theta]."""
    def __init__(self, config: dict, color="blue", body_radius=0.3):
        super().__init__(WMRDynamics(**config))
        self.color = color; self.r = body_radius; self.wd = self.dynamics.wd

    @property
    def pos_indices(self): return [0, 1]

    def get_telemetry(self, x, u):
        ws = self.dynamics.extract_wheel_speed(u.reshape(1, -1)).flatten()
        return (f"Pos: ({x[0]:.2f},{x[1]:.2f})\nHead: {np.degrees(x[2])%360:.1f}°\n"
                f"─────────────\nLin: {u[0]:.2f} m/s  Ang: {u[1]:.2f} rad/s\n"
                f"L-Wheel: {ws[0]:.2f}  R-Wheel: {ws[1]:.2f} m/s")

    def create_painter(self, ax):
        p = {}
        p["body"] = patches.Circle((0, 0), self.r, fc=self.color, alpha=0.6, zorder=2)
        ax.add_patch(p["body"])
        for k in ("l_wheel", "r_wheel"):
            w = patches.Rectangle((0, 0), self.r*0.6, self.r*0.2, fc="k", zorder=3)
            ax.add_patch(w); p[k] = w
        p["heading"] = patches.FancyArrow(0, 0, 1, 0, width=0.05, color="red", zorder=4)
        ax.add_patch(p["heading"])
        return p

    def robot_artists(self, painter):
        return [painter["body"], painter["l_wheel"],
                painter["r_wheel"], painter["heading"]]

    def update_painter(self, painter, time, x_state, u_control, **kwargs):
        px, py, th = x_state[:3]
        c, s = np.cos(th), np.sin(th); R = np.array([[c, -s], [s, c]])
        painter["body"].center = (px, py)
        al = self.r * 1.2
        painter["heading"].set_data(x=px, y=py, dx=al*c, dy=al*s)
        wl, ww = self.r*0.6, self.r*0.2
        for name, ly in [("l_wheel", self.wd/2 - ww/2),
                          ("r_wheel", -self.wd/2 - ww/2)]:
            painter[name].set_xy((R @ np.array([-wl/2, ly])) + [px, py])
            painter[name].angle = np.degrees(th)


class Car(Robot):
    """Ackermann-steering car.  State: [x, y, v, theta]."""

    def __init__(self, config: dict, color="orange", width=1.0):
        super().__init__(CarDynamics(**config))
        self.color = color; self.L = self.dynamics.frd; self.W = width
        self.wheel_len = self.L*0.3; self.wheel_width = self.W*0.2

    @property
    def pos_indices(self): return [0, 1]

    def get_telemetry(self, x, u):
        return (f"Pos: ({x[0]:.2f},{x[1]:.2f})\nHead: {np.degrees(x[3])%360:.1f}°\n"
                f"Speed: {x[2]:.2f} m/s\n─────────────\n"
                f"Accel: {u[0]:.2f} m/s²  Steer: {np.degrees(u[1]):.1f}°")

    def create_painter(self, ax):
        p = {}
        bl = self.L*1.4
        p["chassis"] = patches.Rectangle((0, 0), bl, self.W, fc=self.color, alpha=0.7, zorder=2002)
        ax.add_patch(p["chassis"])
        for k in ("RL", "RR", "FL", "FR"):
            w = patches.Rectangle((0, 0), self.wheel_len, self.wheel_width, fc="black", zorder=2003)
            ax.add_patch(w); p[k] = w
        p["heading"] = patches.FancyArrow(0, 0, 1, 0, width=self.W*0.1, color="cyan", zorder=2004, alpha=0.8)
        ax.add_patch(p["heading"])
        return p

    def robot_artists(self, painter):
        return [painter[k] for k in ("chassis", "RL", "RR", "FL", "FR", "heading")]

    def update_painter(self, painter, time, x_state, u_control, **kwargs):
        px, py, v, th = x_state[:4]; delta = u_control[1]
        c, s = np.cos(th), np.sin(th); Rb = np.array([[c, -s], [s, c]])
        bl = self.L*1.4
        painter["chassis"].set_xy((Rb @ np.array([-bl*0.2, -self.W/2])) + [px, py])
        painter["chassis"].angle = np.degrees(th)
        al = max(abs(v), 0.5)
        painter["heading"].set_data(x=px, y=py, dx=al*c, dy=al*s)
        woff = np.array([-self.wheel_len/2, -self.wheel_width/2])
        for name, lp in [("RL", [0, self.W/2]), ("RR", [0, -self.W/2])]:
            painter[name].set_xy((Rb @ (np.array(lp) + woff)) + [px, py])
            painter[name].angle = np.degrees(th)
        tw = th + delta; cw, sw = np.cos(tw), np.sin(tw)
        Rw = np.array([[cw, -sw], [sw, cw]])
        for name, ap in [("FL", [self.L, self.W/2]), ("FR", [self.L, -self.W/2])]:
            axle = (Rb @ np.array(ap)) + [px, py]
            painter[name].set_xy(axle + (Rw @ woff))
            painter[name].angle = np.degrees(tw)


class Quadcopter(Robot):
    """Top-down quadcopter silhouette.  heading_offset=-π/2 (visual only)."""

    def __init__(self, config: dict, color="k"):
        super().__init__(QuadcopterDynamics(**config))
        self.color = color
        self.body_pts_3d = np.vstack([self.dynamics.motor_pos.T, np.zeros((1, 4))])
        self.prop_radius = self.dynamics.l * 0.35

    @property
    def pos_indices(self): return [0, 2]

    @property
    def heading_offset(self): return -np.pi / 2

    def get_telemetry(self, x, u):
        rpm = u * 9.54929658551
        return (f"Alt: {x[4]:.2f} m  Speed: {np.sqrt(x[1]**2+x[3]**2+x[5]**2):.2f} m/s\n"
                f"Gnd: {np.hypot(x[1],x[3]):.2f} m/s  Climb: {x[5]:+.2f} m/s\n─────────────\n"
                f"φ:{np.degrees(x[6]):5.1f}° θ:{np.degrees(x[7]):5.1f}° "
                f"ψ:{(np.degrees(x[8])+180)%360-180:5.1f}°\n"
                f"p:{np.degrees(x[9]):5.1f} q:{np.degrees(x[10]):5.1f} "
                f"r:{np.degrees(x[11]):5.1f} °/s\n"
                f"─────────────\nRPM: {rpm[0]:.0f} {rpm[1]:.0f} {rpm[2]:.0f} {rpm[3]:.0f}")

    def _R(self, phi, theta, psi):
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi), np.sin(psi)
        return (np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]])
                @ np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
                @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]))

    def create_painter(self, ax):
        p = {}
        p["arm1"], = ax.plot([], [], "-", color=self.color, lw=2)
        p["arm2"], = ax.plot([], [], "-", color=self.color, lw=2)
        p["props"] = [ax.add_patch(patches.Circle((0, 0), self.prop_radius, fc=c, alpha=0.8, zorder=2005))
                      for c in ["blue", "blue", "red", "red"]]
        pivot = self.prop_radius * 1.2
        p["heading"] = patches.FancyArrow(
            pivot, 0, pivot, 0, width=self.dynamics.l * 0.3, color="magenta", zorder=2010, length_includes_head=True)
        ax.add_patch(p["heading"])
        return p

    def robot_artists(self, painter):
        return ([painter["arm1"], painter["arm2"], painter["heading"]]
                + painter["props"])

    def update_painter(self, painter, time, x_state, u_control, **kwargs):
        px, py = x_state[0], x_state[2]
        R = self._R(x_state[6], x_state[7], x_state[8])
        pr = R @ self.body_pts_3d
        wx, wy = pr[0] + px, pr[1] + py
        painter["arm1"].set_data(wx[[0,2]], wy[[0,2]])
        painter["arm2"].set_data(wx[[1,3]], wy[[1,3]])
        for i, c in enumerate(painter["props"]):
            c.center = (wx[i], wy[i])
        nose = R @ np.array([[0], [1], [0]])
        pivot = self.prop_radius * 1.2
        al = self.dynamics.l * 1.5
        _set_arrow_outside(painter["heading"], px, py, nose[0, 0], nose[1, 0], pivot, scale=al)


class QuadcopterPerspective(Robot):
    """
    Quadcopter with perspective-projected props, drop-shadow, and arrows
    that start outside the body perimeter.

    Arrow conventions
    -----------------
    heading     (magenta) — nose direction from current attitude.
    heading_ref (green)   — reference yaw from the planner.
    vel_head    (skyblue) — horizontal velocity direction; length = speed.
    All three pivot at the body centre but originate at ``pivot_r`` from it
    so they never overlap the drone silhouette.
    """

    def __init__(self, config: dict, color="k", projection="orthographic",
                 camera_dist=100.0, shadow_z=20.0):
        super().__init__(QuadcopterDynamics(**config))
        self.color = color
        self.projection_mode = projection
        self.camera_dist = camera_dist
        self.shadow_z = shadow_z
        self.body_pts_3d = np.vstack([self.dynamics.motor_pos.T, np.zeros((1, 4))])
        self.prop_radius = self.dynamics.l * 0.4

        t = np.linspace(0, 2*np.pi, 40)
        self._base_circle = np.array([self.prop_radius * np.cos(t), self.prop_radius * np.sin(t), np.zeros(40)])

    @property
    def pos_indices(self): return [0, 2]

    @property
    def heading_offset(self): return -np.pi / 2

    def get_telemetry(self, x, u):
        rpm = u * 9.54929658551
        return (
            f"Alt (Z):    {x[4]:.2f} m\n"
            f"Speed 3D:   {np.sqrt(x[1]**2+x[3]**2+x[5]**2):.2f} m/s\n"
            f"Gnd Speed:  {np.hypot(x[1],x[3]):.2f} m/s\n"
            f"Climb:      {x[5]:+.2f} m/s\n"
            f"────────────────\n"
            f"Pitch(φ): {np.degrees(x[6]):5.1f}° (p: {np.degrees(x[9]):5.1f}°/s)\n"
            f"Roll(θ):  {np.degrees(x[7]):5.1f}° (q: {np.degrees(x[10]):5.1f}°/s)\n"
            f"Yaw(ψ):   {(np.degrees(x[8])+180)%360-180:5.1f}° (r: {np.degrees(x[11]):5.1f}°/s)\n"
            f"────────────────\n"
            f"Motors (RPM):\n"
            f"1:{rpm[0]:5.0f}  2:{rpm[1]:5.0f}\n"
            f"3:{rpm[2]:5.0f}  4:{rpm[3]:5.0f}"
        )

    def _R(self, phi, theta, psi):
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi), np.sin(psi)
        return (np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]])
                @ np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
                @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]))

    def _scale(self, z_world):
        if self.projection_mode == "orthographic":
            return np.ones_like(np.asarray(z_world, dtype=float))
        return self.camera_dist / (self.camera_dist - np.clip(z_world, -100, self.camera_dist * 0.9))

    def create_painter(self, ax):
        p = {}
        pivot = self.prop_radius * 1.5
        al = self.dynamics.l * 3.0

        # Shadow
        sh = dict(color="grey", alpha=0.4, zorder=1901, lw=2)
        p["shadow_arm1"], = ax.plot([], [], "-", **sh)
        p["shadow_arm2"], = ax.plot([], [], "-", **sh)
        p["shadow_props"] = [ax.add_patch(patches.Polygon(np.zeros((40, 2)), fc="black", alpha=0.3, zorder=1900)) for _ in range(4)]

        # Arms
        p["arm1"], = ax.plot([], [], "-", color=self.color, lw=1.5, zorder=2002)
        p["arm2"], = ax.plot([], [], "-", color=self.color, lw=1.5, zorder=2002)

        # Props
        p["prop_fills"] = []
        p["prop_lines"] = []
        for c in ["blue", "blue", "red", "red"]:
            poly = patches.Polygon(np.zeros((40, 2)), fc=c, alpha=0.8, zorder=2003)
            line, = ax.plot([], [], color=c, lw=0.5, zorder=2004)
            ax.add_patch(poly)
            p["prop_fills"].append(poly)
            p["prop_lines"].append(line)

        arrow_w = self.dynamics.l * 0.9
        p["heading"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="magenta", alpha=0.95, zorder=2101, length_includes_head=True)
        p["heading_ref"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="#2ecc71", alpha=0.95, zorder=2100, length_includes_head=True)
        p["vel_head"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="skyblue", alpha=0.95, zorder=2099, length_includes_head=True)

        for key in ("heading", "heading_ref", "vel_head"):
            ax.add_patch(p[key])
        p["_pivot_r"] = pivot
        return p

    def robot_artists(self, painter):
        return (
            [painter["arm1"], painter["arm2"],
             painter["shadow_arm1"], painter["shadow_arm2"],
             painter["heading"], painter["heading_ref"], painter["vel_head"]]
            + painter["prop_fills"] + painter["prop_lines"]
            + painter["shadow_props"]
        )

    def update_painter(self, painter, time, x_state, u_control, ref: Optional[np.ndarray] = None, **kwargs):
        px, py, pz = x_state[0], x_state[2], x_state[4]
        vx, vy     = x_state[1], x_state[3]
        R = self._R(x_state[6], x_state[7], x_state[8])

        p_local  = R @ self.body_pts_3d
        factors  = self._scale(p_local[2] + pz)
        shad_f   = float(self._scale(np.array([self.shadow_z]))[0])
        pivot_r  = painter.get("_pivot_r", self.prop_radius * 1.5)
        off = 0.2

        # Shadow arms
        so = p_local[0:2] * shad_f
        painter["shadow_arm1"].set_data(so[0,[0,2]] + px + off, so[1,[0,2]] + py + off)
        painter["shadow_arm2"].set_data(so[0,[1,3]] + px + off, so[1,[1,3]] + py + off)

        # Arms
        bo = p_local[0:2] * factors
        wx, wy = bo[0] + px, bo[1] + py
        painter["arm1"].set_data(wx[[0,2]], wy[[0,2]])
        painter["arm2"].set_data(wx[[1,3]], wy[[1,3]])

        # Props (batched)
        expanded = self._base_circle[:, :, None] + self.body_pts_3d[:, None, :]  # (3,40,4)
        all_pts  = (R @ expanded.reshape(3, -1)).reshape(3, 40, 4)
        fp_all   = self._scale(all_pts[2] + pz)

        for i in range(4):
            rx = all_pts[0, :, i] * fp_all[:, i] + px
            ry = all_pts[1, :, i] * fp_all[:, i] + py
            painter["prop_fills"][i].set_xy(np.column_stack((rx, ry)))
            painter["prop_lines"][i].set_data(rx, ry)
            sp = all_pts[0:2, :, i] * shad_f
            painter["shadow_props"][i].set_xy(np.column_stack((sp[0] + px + off, sp[1] + py + off)))

        avg_f = float(np.mean(factors))
        al = self.dynamics.l * 3.0 * avg_f
        nose = R @ np.array([[0], [1], [0]])
        _set_arrow_outside(painter["heading"], px, py, nose[0, 0], nose[1, 0], pivot_r, scale=al)

        # Reference-heading arrow
        if ref is None:
            painter["heading_ref"].set_visible(False)
        else:
            psi_ref = ref[8]
            _set_arrow_outside(painter["heading_ref"], px, py, -np.sin(psi_ref) * al, np.cos(psi_ref) * al, pivot_r, scale=al)

        # Velocity arrow — length = speed, direction = velocity
        v_mag = np.hypot(vx, vy)
        if v_mag > 0.2:
            # Scale: 1 m/s → 1 unit, capped at 3× arm length
            speed_scale = np.clip(v_mag, 0.5, self.dynamics.l * 9)
            _set_arrow_outside(painter["vel_head"], px, py, vx, vy, pivot_r, scale=speed_scale)
        else:
            painter["vel_head"].set_visible(False)


class SmallFixedWing(Robot):
    """
    Top-down fixed-wing UAV silhouette.

    Draws a delta-wing body with elevator and rudder surfaces.
    State layout matches SmallFixedWingDynamics: [x, vx, y, vy, z, vz, φ, θ, ψ, p, q, r].
    """

    def __init__(self, config: dict, color="#e67e22", wingspan: Optional[float] = None):
        super().__init__(SmallFixedWingDynamics(**config))
        self.color = color
        self.b = wingspan or self.dynamics.b   # wingspan for drawing scale

    @property
    def pos_indices(self): return [0, 2]

    @property
    def heading_offset(self): return -np.pi / 2

    def get_telemetry(self, x, u):
        vx, vy, vz = x[1], x[3], x[5]
        Va = np.sqrt(vx**2 + vy**2 + vz**2)
        return (
            f"Alt:   {x[4]:.2f} m\n"
            f"Va:    {Va:.2f} m/s   Climb: {vz:+.2f} m/s\n"
            f"────────────────\n"
            f"Roll(φ): {np.degrees(x[6]):5.1f}°\n"
            f"Pitch(θ):{np.degrees(x[7]):5.1f}°\n"
            f"Yaw(ψ):  {(np.degrees(x[8])+180)%360-180:5.1f}°\n"
            f"────────────────\n"
            f"δt:{u[0]:.2f}  δa:{np.degrees(u[1]):4.1f}°  "
            f"δe:{np.degrees(u[2]):4.1f}°  δr:{np.degrees(u[3]):4.1f}°"
        )

    @staticmethod
    def _wing_polygon(b: float):
        """Local-frame wing outline (tapered delta wing)."""
        hs = b / 2.0   # half-span
        fc = b * 0.35  # fuselage chord
        # Points: nose → right-tip → right-te → tail → left-te → left-tip → back
        pts = np.array([
            [0,      fc],         # nose (y forward)
            [ hs,    -fc*0.1],    # right wingtip
            [ hs*0.55, -fc*0.4],  # right trailing edge
            [0,     -fc*0.6],     # tail centreline
            [-hs*0.55, -fc*0.4],  # left trailing edge
            [-hs,    -fc*0.1],    # left wingtip
        ])
        return pts  # (6, 2) in body X-Y (right, fwd)

    @staticmethod
    def _fuselage_polygon(b: float):
        """Narrow fuselage rectangle."""
        fc = b * 0.35
        hw = b * 0.05
        return np.array([
            [-hw,  fc],
            [ hw,  fc],
            [ hw, -fc*0.7],
            [-hw, -fc*0.7],
        ])

    def create_painter(self, ax):
        p = {}
        b = self.b
        pivot = b * 0.6

        p["wing"] = patches.Polygon(self._wing_polygon(b), closed=True, fc=self.color, alpha=0.75, zorder=2002, ec="k", lw=0.5)
        p["fuse"] = patches.Polygon(self._fuselage_polygon(b), closed=True, fc=self.color, alpha=0.90, zorder=2003, ec="k", lw=0.7)
        ax.add_patch(p["wing"]); ax.add_patch(p["fuse"])

        arrow_w = b * 0.08
        p["heading"] = patches.FancyArrow(pivot, 0, pivot, 0, width=arrow_w, fc="magenta", alpha=0.95, zorder=2101, length_includes_head=True)
        p["heading_ref"] = patches.FancyArrow(pivot, 0, pivot, 0, width=arrow_w, fc="#2ecc71", alpha=0.95, zorder=2100, length_includes_head=True)
        p["vel_head"] = patches.FancyArrow(pivot, 0, pivot, 0, width=arrow_w, fc="skyblue", alpha=0.95, zorder=2102, length_includes_head=True)
        
        for key in ("heading", "heading_ref", "vel_head"): ax.add_patch(p[key])
        p["_pivot_r"] = pivot
        return p

    def robot_artists(self, painter):
        return [painter[k] for k in
                ("wing", "fuse", "heading", "heading_ref", "vel_head")]

    def update_painter(self, painter, time, x_state, u_control, ref: Optional[np.ndarray] = None, **kwargs):
        px, py = x_state[0], x_state[2]
        vx, vy = x_state[1], x_state[3]
        psi    = x_state[8]
        pivot  = painter.get("_pivot_r", self.b * 0.6)

        # Rotation matrix (body → world, 2-D top-down: X right, Y forward)
        # psi=0 → nose +Y, so rotate basis vectors
        cs, ss = np.cos(psi), np.sin(psi)
        # Body X→World: (cos psi, sin psi);  Body Y→World: (-sin psi, cos psi)
        Rxy = np.array([[cs, -ss], [ss,  cs]])

        def _transform(local_pts): return (Rxy @ local_pts.T).T + [px, py]

        painter["wing"].set_xy(_transform(self._wing_polygon(self.b)))
        painter["fuse"].set_xy(_transform(self._fuselage_polygon(self.b)))

        al = self.b * 1.2
        nose_dx, nose_dy = -ss * al, cs * al   # psi=0 → +Y

        _set_arrow_outside(painter["heading"], px, py,
                           nose_dx, nose_dy, pivot, scale=al)

        if ref is None:
            painter["heading_ref"].set_visible(False)
        else:
            psi_r = ref[8]
            _set_arrow_outside(painter["heading_ref"], px, py, -np.sin(psi_r)*al, np.cos(psi_r)*al, pivot, scale=al)

        v_mag = np.hypot(vx, vy)
        if v_mag > 0.5:
            _set_arrow_outside(painter["vel_head"], px, py, vx, vy, pivot, scale=np.clip(v_mag, 1.0, self.b * 3))
        else:
            painter["vel_head"].set_visible(False)


class SmallFixedWingPerspective(Robot):
    """
    Small fixed-wing UAV with perspective projection, pitch/roll tilt, drop-shadow, 
    and arrows that start outside the body perimeter.
    """

    def __init__(self, config: dict, color="#e67e22", wingspan: Optional[float] = None, projection="orthographic", camera_dist=100.0, shadow_z=20.0):
        super().__init__(SmallFixedWingDynamics(**config))
        self.color = color
        self.projection_mode = projection
        self.camera_dist = camera_dist
        self.shadow_z = shadow_z
        self.b = wingspan or self.dynamics.b
        
        # Get 2D base points and append Z=0 to make them 3D
        wp_2d = SmallFixedWing._wing_polygon(self.b)
        fp_2d = SmallFixedWing._fuselage_polygon(self.b)
        self.wing_pts_3d = np.column_stack((wp_2d, np.zeros(len(wp_2d))))
        self.fuse_pts_3d = np.column_stack((fp_2d, np.zeros(len(fp_2d))))

    @property
    def pos_indices(self): return [0, 2]

    @property
    def heading_offset(self): return -np.pi / 2

    def get_telemetry(self, x, u):
        vx, vy, vz = x[1], x[3], x[5]
        Va = np.sqrt(vx**2 + vy**2 + vz**2)
        return (
            f"Alt (Z): {x[4]:.2f} m\n"
            f"Va:      {Va:.2f} m/s   Climb: {vz:+.2f} m/s\n"
            f"────────────────\n"
            f"Roll(φ): {np.degrees(x[6]):5.1f}° (p: {np.degrees(x[9]):5.1f}°/s)\n"
            f"Pitch(θ):{np.degrees(x[7]):5.1f}° (q: {np.degrees(x[10]):5.1f}°/s)\n"
            f"Yaw(ψ):  {(np.degrees(x[8])+180)%360-180:5.1f}° (r: {np.degrees(x[11]):5.1f}°/s)\n"
            f"────────────────\n"
            f"δt:{u[0]:.2f}  δa:{np.degrees(u[1]):4.1f}°  "
            f"δe:{np.degrees(u[2]):4.1f}°  δr:{np.degrees(u[3]):4.1f}°"
        )

    def _R(self, phi, theta, psi):
        """ZYX Euler Rotation Matrix"""
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi), np.sin(psi)
        return (np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]])
                @ np.array([[ct,0,st],[0,1,0],[-st,0,ct]])
                @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]))

    def _scale(self, z_world):
        if self.projection_mode == "orthographic":
            return np.ones_like(np.asarray(z_world, dtype=float))
        return self.camera_dist / (self.camera_dist - np.clip(z_world, -100, self.camera_dist * 0.9))

    def create_painter(self, ax):
        p = {}
        b = self.b
        pivot = b * 0.6
        
        # Shadows
        sh_kw = dict(fc="black", alpha=0.3, zorder=1900, ec="none")
        p["shadow_wing"] = patches.Polygon(np.zeros((6, 2)), **sh_kw)
        p["shadow_fuse"] = patches.Polygon(np.zeros((4, 2)), **sh_kw)
        ax.add_patch(p["shadow_wing"])
        ax.add_patch(p["shadow_fuse"])

        # Body
        p["wing"] = patches.Polygon(np.zeros((6, 2)), closed=True, fc=self.color, alpha=0.85, zorder=2002, ec="k", lw=0.5)
        p["fuse"] = patches.Polygon(np.zeros((4, 2)), closed=True, fc=self.color, alpha=0.95, zorder=2003, ec="k", lw=0.7)
        ax.add_patch(p["wing"])
        ax.add_patch(p["fuse"])

        # Arrows
        arrow_w = b * 0.08
        al = b * 1.2
        p["heading"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="magenta", alpha=0.95, zorder=2101, length_includes_head=True)
        p["heading_ref"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="#2ecc71", alpha=0.95, zorder=2100, length_includes_head=True)
        p["vel_head"] = patches.FancyArrow(pivot, 0, al, 0, width=arrow_w, fc="skyblue", alpha=0.95, zorder=2102, length_includes_head=True)
        
        for key in ("heading", "heading_ref", "vel_head"): ax.add_patch(p[key])
        p["_pivot_r"] = pivot
        return p

    def robot_artists(self, painter):
        return [painter[k] for k in (
            "shadow_wing", "shadow_fuse", "wing", "fuse",
            "heading", "heading_ref", "vel_head")]

    def update_painter(self, painter, time, x_state, u_control, ref: Optional[np.ndarray] = None, **kwargs):
        px, py, pz = x_state[0], x_state[2], x_state[4]
        vx, vy = x_state[1], x_state[3]
        R = self._R(x_state[6], x_state[7], x_state[8])
        
        pivot_r = painter.get("_pivot_r", self.b * 0.6)
        shad_f  = float(self._scale(np.array([self.shadow_z]))[0])
        off = 0.2 # Shadow offset

        # Function to process 3D points -> 2D projected points
        def _project(pts_3d):
            local_3d = R @ pts_3d.T
            factors = self._scale(local_3d[2] + pz)
            # Projected body
            body_pts = np.column_stack((local_3d[0] * factors + px, local_3d[1] * factors + py))
            # Shadow body
            shad_pts = np.column_stack((local_3d[0] * shad_f + px + off, local_3d[1] * shad_f + py + off))
            return body_pts, shad_pts, np.mean(factors)

        # Apply projections
        wing_body, wing_shad, f_avg_wing = _project(self.wing_pts_3d)
        fuse_body, fuse_shad, f_avg_fuse = _project(self.fuse_pts_3d)

        # Update artists
        painter["wing"].set_xy(wing_body)
        painter["shadow_wing"].set_xy(wing_shad)
        painter["fuse"].set_xy(fuse_body)
        painter["shadow_fuse"].set_xy(fuse_shad)

        # Arrows
        avg_f = float((f_avg_wing + f_avg_fuse) / 2.0)
        al = self.b * 1.2 * avg_f

        # Heading arrow (Nose direction projected)
        nose = R @ np.array([[0], [1], [0]])
        _set_arrow_outside(painter["heading"], px, py, nose[0, 0], nose[1, 0], pivot_r, scale=al)

        # Reference-heading arrow
        if ref is None:
            painter["heading_ref"].set_visible(False)
        else:
            psi_ref = ref[8]
            _set_arrow_outside(painter["heading_ref"], px, py, -np.sin(psi_ref) * al, np.cos(psi_ref) * al, pivot_r, scale=al)

        # Velocity arrow
        v_mag = np.hypot(vx, vy)
        if v_mag > 0.5:
            _set_arrow_outside(painter["vel_head"], px, py, vx, vy, pivot_r, scale=np.clip(v_mag, 1.0, self.b * 3) * avg_f)
        else:
            painter["vel_head"].set_visible(False)