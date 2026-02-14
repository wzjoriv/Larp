from abc import ABC, abstractmethod
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from larp.dynamics import CarDynamics, Dynamics, QuadcopterV1Dynamics, QuadcopterV2Dynamics, WMRDynamics
from scipy.spatial import HalfspaceIntersection

import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd
import re

# ==========================================
# Helper Functions (Vectorized)
# ==========================================

def hex_to_rgb(hex_color):
    """Converts hex string to (3,) numpy array (0-1)."""
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0

def interpolate_colors(c1, c2, t):
    """Vectorized linear interpolation between two RGB arrays."""
    # t shape: (N, 1), c1/c2 shape: (3,)
    # Result: (N, 3)
    return (1.0 - t) * c1 + t * c2

# ==========================================
# Robot Abstraction
# ==========================================

class Robot(ABC):
    """
    Abstract Base Class for Robots.
    Connects Dynamics to Visualization and defines State Mappings.
    """
    def __init__(self, dynamics: Dynamics):
        self.dynamics = dynamics

    @property
    @abstractmethod
    def pos_indices(self) -> List[int]:
        """ Returns the indices [x_idx, y_idx] in the state vector for 2D plotting. """
        pass

    @abstractmethod
    def get_telemetry(self, state: np.ndarray, control: np.ndarray) -> str:
        """ Returns a formatted string of robot-specific states and controls (Height, Velocity, etc.) """
        pass

    @abstractmethod
    def create_painter(self, ax: plt.Axes):
        """ Creates Matplotlib artists for the robot. """
        pass

    @abstractmethod
    def update_painter(self, painter, time:float, x_state: np.ndarray, u_control: np.ndarray):
        """ Updates the artists based on current state. """
        pass

class WMR(Robot):
    def __init__(self, config: dict, color='blue', body_radius=0.3):
        dynamics = WMRDynamics(**config)
        super().__init__(dynamics)
        self.color = color
        self.r = body_radius
        self.wd = self.dynamics.wd # Wheel distance

    @property
    def pos_indices(self) -> List[int]:
        return [0, 1] # x, y

    def get_telemetry(self, x: np.ndarray, u: np.ndarray) -> str:
        # State: [x, y, theta]
        # Control: [v, w]
        
        px, py, theta = x[0], x[1], x[2]
        v, w = u[0], u[1]
        
        # Calculate individual wheel speeds
        # v_l = v - (wd/2)*w, v_r = v + (wd/2)*w
        wheel_speeds = self.dynamics.extract_wheel_speed(u.reshape(1, -1)).flatten()
        vl, vr = wheel_speeds[0], wheel_speeds[1]

        theta_deg = np.degrees(theta) % 360
        
        return (f"Pos: ({px:.2f}, {py:.2f})\n"
                f"Head: {theta_deg:.1f}°\n"
                f"----------------\n"
                f"Lin Vel: {v:.2f} m/s\n"
                f"Ang Vel: {w:.2f} rad/s\n"
                f"----------------\n"
                f"L-Wheel: {vl:.2f} m/s\n"
                f"R-Wheel: {vr:.2f} m/s")

    def create_painter(self, ax: plt.Axes):
        painter = {}
        # Main Body
        painter['body'] = patches.Circle((0, 0), radius=self.r, fc=self.color, alpha=0.6, zorder=2)
        ax.add_patch(painter['body'])
        
        # Wheels (Rectangles)
        # Dimensions: length, width
        w_len, w_width = self.r * 0.6, self.r * 0.2
        painter['l_wheel'] = patches.Rectangle((0, 0), w_len, w_width, fc='k', zorder=3)
        painter['r_wheel'] = patches.Rectangle((0, 0), w_len, w_width, fc='k', zorder=3)
        ax.add_patch(painter['l_wheel'])
        ax.add_patch(painter['r_wheel'])
        
        # Heading Arrow
        painter['heading'] = patches.FancyArrow(0, 0, 1, 0, width=0.05, color='red', zorder=4)
        ax.add_patch(painter['heading'])
        
        return painter

    def update_painter(self, painter, time: float, x_state: np.ndarray, u_control: np.ndarray):
        px, py, theta = x_state[0], x_state[1], x_state[2]
        
        # Transform helpers
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        
        # Update Body Center
        painter['body'].center = (px, py)
        
        # Update Heading Arrow (Visual length = body radius)
        arrow_len = self.r * 1.2
        dx, dy = arrow_len * c, arrow_len * s
        painter['heading'].set_data(x=px, y=py, dx=dx, dy=dy)
        
        # Update Wheels
        # Local positions: centered at X=0, Y= +/- wd/2
        w_len, w_width = self.r * 0.6, self.r * 0.2
        
        # Left Wheel
        lw_local = np.array([-w_len/2, self.wd/2 - w_width/2])
        lw_global = (R @ lw_local) + np.array([px, py])
        painter['l_wheel'].set_xy(lw_global)
        painter['l_wheel'].angle = np.degrees(theta)
        
        # Right Wheel
        rw_local = np.array([-w_len/2, -self.wd/2 - w_width/2])
        rw_global = (R @ rw_local) + np.array([px, py])
        painter['r_wheel'].set_xy(rw_global)
        painter['r_wheel'].angle = np.degrees(theta)


class Car(Robot):
    def __init__(self, config: dict, color='orange', width=1.0):
        dynamics = CarDynamics(**config)
        super().__init__(dynamics)
        self.color = color
        
        self.L = self.dynamics.frd  # Wheelbase
        
        self.W = width   
        
        # Visualization Dimensions
        self.wheel_len = self.L * 0.3
        self.wheel_width = self.W * 0.2

    @property
    def pos_indices(self) -> List[int]:
        return [0, 1] # x, y

    def get_telemetry(self, x: np.ndarray, u: np.ndarray) -> str:
        # State: [x, y, v, theta]
        # Control: [a, steering_angle]
        
        px, py = x[0], x[1]
        v = x[2]
        theta_deg = np.degrees(x[3]) % 360
        
        accel = u[0]
        steering_rad = u[1]
        steering_deg = np.degrees(steering_rad)
        
        return (f"Pos: ({px:.2f}, {py:.2f})\n"
                f"Head: {theta_deg:.1f}°\n"
                f"Speed: {v:.2f} m/s\n"
                f"----------------\n"
                f"Accel: {accel:.2f} m/s²\n"
                f"Steer: {steering_deg:.1f}°")

    def create_painter(self, ax: plt.Axes):
        painter = {}
        
        # Chassis (Body)
        # Modeled as rectangle centered between axles
        body_len = self.L * 1.4
        painter['chassis'] = patches.Rectangle((0, 0), body_len, self.W, fc=self.color, alpha=0.7, zorder=2)
        ax.add_patch(painter['chassis'])
        
        # Wheels
        # RL, RR (Rear Left/Right) - Fixed Orientation relative to body
        # FL, FR (Front Left/Right) - Steerable
        for w_name in ['RL', 'RR', 'FL', 'FR']:
            w = patches.Rectangle((0, 0), self.wheel_len, self.wheel_width, fc='black', zorder=3)
            ax.add_patch(w)
            painter[w_name] = w
            
        # Velocity/Heading Arrow
        painter['heading'] = patches.FancyArrow(0, 0, 1, 0, width=self.W*0.1, color='cyan', zorder=4, alpha=0.8)
        ax.add_patch(painter['heading'])
        
        return painter

    def update_painter(self, painter, time: float, x_state: np.ndarray, u_control: np.ndarray):
        # State: [x, y, v, theta]
        px, py, v, theta = x_state
        steering_angle = u_control[1]
        
        # Rotation Matrix for Body (Global Heading)
        c, s = np.cos(theta), np.sin(theta)
        R_body = np.array([[c, -s], [s, c]])
        
        # --- Update Chassis ---
        # Center of rear axle is (px, py).
        # We want to draw chassis slightly offset so rear axle is near the back
        body_len = self.L * 1.4
        # Local bottom-left corner of chassis rectangle relative to Rear Axle Center
        chassis_local_offset = np.array([-body_len * 0.2, -self.W / 2])
        chassis_global = (R_body @ chassis_local_offset) + np.array([px, py])
        
        painter['chassis'].set_xy(chassis_global)
        painter['chassis'].angle = np.degrees(theta)
        
        # --- Update Heading Arrow ---
        arrow_len = max(abs(v), 0.5) * 1.0 # Scale by speed
        painter['heading'].set_data(x=px, y=py, dx=arrow_len*c, dy=arrow_len*s)
        
        # --- Update Wheels ---
        # Wheel offset to center the rectangle on its pivot point
        w_center_off = np.array([-self.wheel_len/2, -self.wheel_width/2])
        
        # 1. Rear Wheels (Fixed to Body)
        # Positions relative to Rear Axle Center (0,0)
        pos_RL = np.array([0, self.W/2])
        pos_RR = np.array([0, -self.W/2])
        
        for name, pos_local in [('RL', pos_RL), ('RR', pos_RR)]:
            # Apply offset to center wheel rect, then rotate by body, then translate
            corner_local = pos_local + w_center_off
            corner_global = (R_body @ corner_local) + np.array([px, py])
            
            painter[name].set_xy(corner_global)
            painter[name].angle = np.degrees(theta)
            
        # 2. Front Wheels (Steerable)
        # Positions relative to Rear Axle Center: (L, +/- W/2)
        pos_FL_axle = np.array([self.L, self.W/2])
        pos_FR_axle = np.array([self.L, -self.W/2])
        
        # Rotation for Steering (Local to Body)
        # Total angle = Theta (Body) + Delta (Steering)
        total_wheel_angle = theta + steering_angle
        
        # Matrix just for the steering deflection (for visualization offset calculation if strictly needed, 
        # but rectangle rotation handles visual angle)
        
        for name, pos_axle_local in [('FL', pos_FL_axle), ('FR', pos_FR_axle)]:
            # Where is the axle on the map?
            axle_global = (R_body @ pos_axle_local) + np.array([px, py])
            
            # The wheel rectangle needs to be rotated by (theta + steering)
            # And we need to find the bottom-left corner of the rotated rectangle 
            # relative to the axle center.
            
            # Global Rotation Matrix for the wheel
            cw, sw = np.cos(total_wheel_angle), np.sin(total_wheel_angle)
            R_wheel = np.array([[cw, -sw], [sw, cw]])
            
            # Rotate the center offset vector
            corner_offset_global = R_wheel @ w_center_off
            corner_global = axle_global + corner_offset_global
            
            painter[name].set_xy(corner_global)
            painter[name].angle = np.degrees(total_wheel_angle)

# ==========================================
# Concrete Robot: Quadcopter
# ==========================================

class Quadcopter(Robot):
    def __init__(self, config: dict, color='k'):
        # 1. Use V1 Dynamics (Custom Frame)
        dynamics = QuadcopterV1Dynamics(**config)
        super().__init__(dynamics)
        self.color = color
        
        # 3D Body Points
        # V1 motor_pos is (4, 2). Stack 0 for Z.
        zeros = np.zeros((1, 4))
        self.body_pts_3d = np.vstack([self.dynamics.motor_pos.T, zeros])
        
        # Prop radius proportional to arm length
        self.prop_radius = self.dynamics.l * 0.35

    @property
    def pos_indices(self) -> List[int]:
        return [0, 2] # X (Lat/Right), Y (Lon/Forward)

    def get_telemetry(self, x: np.ndarray, u: np.ndarray) -> str:
        # V1 State: [x, vx, y, vy, z, vz, phi, theta, psi, p, q, r]
        
        z = x[4]
        vx, vy, vz = x[1], x[3], x[5]
        
        speed_total = np.sqrt(vx**2 + vy**2 + vz**2)
        ground_speed = np.hypot(vx, vy)
        climb_rate = vz
        
        # Angles
        # Frame: X=Right, Y=Forward
        # Rot X (Phi) -> Pitch (Nose Up/Down)
        # Rot Y (Theta) -> Roll (Wing Up/Down)
        phi_deg   = np.degrees(x[6])
        theta_deg = np.degrees(x[7])
        psi_deg   = (np.degrees(x[8]) + 180) % 360 - 180 
        
        # Rates
        p_deg = np.degrees(x[9])
        q_deg = np.degrees(x[10])
        r_deg = np.degrees(x[11])

        # Motors
        u_rpm = u * 9.54929658551

        return (f"Alt (Z):    {z:.2f} m\n"
                f"Speed 3D:   {speed_total:.2f} m/s\n"
                f"Gnd Speed:  {ground_speed:.2f} m/s\n"
                f"Climb:      {climb_rate:+.2f} m/s\n"
                f"----------------\n"
                f"Pitch(φ): {phi_deg:5.1f}° (p: {p_deg:5.1f})\n"
                f"Roll(θ):  {theta_deg:5.1f}° (q: {q_deg:5.1f})\n"
                f"Yaw(ψ):   {psi_deg:5.1f}° (r: {r_deg:5.1f})"
                f"\n----------------\n"
                f"Motors (RPM):\n"
                f"1: {u_rpm[0]:5.0f}  2: {u_rpm[1]:5.0f}\n"
                f"3: {u_rpm[2]:5.0f}  4: {u_rpm[3]:5.0f}")

    def create_painter(self, ax: plt.Axes):
        painter = {}
        # Arms
        painter['arm1'], = ax.plot([], [], '-', color=self.color, lw=2)
        painter['arm2'], = ax.plot([], [], '-', color=self.color, lw=2)
        
        # Props
        painter['props'] = []
        # Colors based on X=Right, Y=Forward Frame:
        # 1(BR), 2(BL) -> Back -> Blue
        # 3(FL), 4(FR) -> Front -> Red
        colors = ['blue', 'blue', 'red', 'red']
        for i in range(4):
            c = patches.Circle((0, 0), radius=self.prop_radius, fc=colors[i], alpha=0.8, zorder=5)
            ax.add_patch(c)
            painter['props'].append(c)
        
        # Heading
        painter['heading'] = ax.arrow(0, 0, 1, 0, head_width=0.1, fc='magenta', ec='magenta', zorder=10)
        return painter

    def update_painter(self, painter, time:float, x_state: np.ndarray, u_control: np.ndarray):
        px, py = x_state[0], x_state[2]
        phi, theta, psi = x_state[6], x_state[7], x_state[8]
        
        # Rotation
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_s, s_s = np.cos(psi), np.sin(psi)
        
        Rx = np.array([[1, 0, 0], [0, c_p, -s_p], [0, s_p, c_p]])
        Ry = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
        Rz = np.array([[c_s, -s_s, 0], [s_s, c_s, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        # Transform Body Points
        p_rot = R @ self.body_pts_3d
        p_world_x = p_rot[0, :] + px
        p_world_y = p_rot[1, :] + py
        
        # Update Arms
        # 1(BR)-3(FL) and 2(BL)-4(FR)
        painter['arm1'].set_data(p_world_x[[0, 2]], p_world_y[[0, 2]])
        painter['arm2'].set_data(p_world_x[[1, 3]], p_world_y[[1, 3]])
        
        for i, prop_patch in enumerate(painter['props']):
            prop_patch.center = (p_world_x[i], p_world_y[i])
        
        # Update Heading Arrow (Points +Y in Body Frame)
        nose_dir = R @ np.array([[0], [1], [0]]) 
        painter['heading'].remove()

        arrow_len = self.dynamics.l * 1.5
        painter['heading'] = painter['arm1'].axes.arrow(
            px, py, 
            arrow_len * nose_dir[0,0], arrow_len * nose_dir[1,0],
            head_width=self.dynamics.l*0.4, fc='magenta', ec='magenta', zorder=10
        )

# ==========================================
# QuadcopterPerspective
# ==========================================

class QuadcopterV1Perspective(Robot):
    def __init__(self, config: dict, color='k', projection='orthographic', camera_dist=100, shadow_z=20.0):
        # 1. Use V1 Dynamics
        dynamics = QuadcopterV1Dynamics(**config)
        super().__init__(dynamics)
        
        self.color = color
        self.projection_mode = projection
        self.camera_dist = camera_dist
        self.shadow_z = shadow_z
        
        zeros = np.zeros((1, 4))
        self.body_pts_3d = np.vstack([self.dynamics.motor_pos.T, zeros])
        self.prop_radius = self.dynamics.l * 0.4

    @property
    def pos_indices(self) -> List[int]:
        return [0, 2] # X, Y

    def get_telemetry(self, x: np.ndarray, u: np.ndarray) -> str:
        z = x[4]
        vx, vy, vz = x[1], x[3], x[5]
        speed_total = np.sqrt(vx**2 + vy**2 + vz**2)
        ground_speed = np.hypot(vx, vy)
        climb_rate = vz
        
        phi_deg   = np.degrees(x[6])
        theta_deg = np.degrees(x[7])
        psi_deg   = (np.degrees(x[8]) + 180) % 360 - 180 
        
        p_deg = np.degrees(x[9])
        q_deg = np.degrees(x[10])
        r_deg = np.degrees(x[11])

        u_rpm = u * 9.54929658551

        return (f"Alt (Z):    {z:.2f} m\n"
                f"Speed 3D:   {speed_total:.2f} m/s\n"
                f"Gnd Speed:  {ground_speed:.2f} m/s\n"
                f"Climb:      {climb_rate:+.2f} m/s\n"
                f"----------------\n"
                f"Pitch(φ): {phi_deg:5.1f}° (p: {p_deg:5.1f}°/s)\n"
                f"Roll(θ):  {theta_deg:5.1f}° (q: {q_deg:5.1f}°/s)\n"
                f"Yaw(ψ):   {psi_deg:5.1f}° (r: {r_deg:5.1f}°/s)"
                f"\n----------------\n"
                f"Motors (RPM):\n"
                f"1: {u_rpm[0]:5.0f}  2: {u_rpm[1]:5.0f}\n"
                f"3: {u_rpm[2]:5.0f}  4: {u_rpm[3]:5.0f}")

    def _get_rotation_matrix(self, phi, theta, psi):
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_s, s_s = np.cos(psi), np.sin(psi)
        
        Rx = np.array([[1, 0, 0], [0, c_p, -s_p], [0, s_p, c_p]])
        Ry = np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])
        Rz = np.array([[c_s, -s_s, 0], [s_s, c_s, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _project(self, pts_3d):
        if self.projection_mode == 'orthographic':
            return pts_3d[0:2, :]
        z = pts_3d[2, :]
        factor = self.camera_dist / (self.camera_dist - np.clip(z, -100, self.camera_dist * 0.9))
        return pts_3d[0:2, :] * factor

    def create_painter(self, ax: plt.Axes):
        painter = {}
        shadow_style = {'color': 'grey', 'alpha': 0.4, 'zorder': 1.5, 'lw': 2}
        painter['shadow_arm1'], = ax.plot([], [], '-', **shadow_style)
        painter['shadow_arm2'], = ax.plot([], [], '-', **shadow_style)
        painter['shadow_props'] = [ax.add_patch(patches.Polygon(np.zeros((40, 2)), fc='black', alpha=0.3, zorder=1.5)) for _ in range(4)]

        painter['arm1'], = ax.plot([], [], '-', color=self.color, lw=1.5, zorder=2002)
        painter['arm2'], = ax.plot([], [], '-', color=self.color, lw=1.5, zorder=2002)
        
        painter['prop_fills'] = []
        painter['prop_lines'] = []
        
        # Colors: Back (Blue), Front (Red)
        colors = ['blue', 'blue', 'red', 'red'] 
        
        for i in range(4):
            poly = patches.Polygon(np.zeros((40, 2)), fc=colors[i], alpha=0.8, zorder=2003)
            line, = ax.plot([], [], color=colors[i], lw=0.5, zorder=2004)
            ax.add_patch(poly)
            painter['prop_fills'].append(poly)
            painter['prop_lines'].append(line)
        
        painter['heading'] = patches.Polygon(np.zeros((3, 2)), fc='magenta', alpha=0.99, zorder=2100, label="Heading")
        ax.add_patch(painter['heading'])
        painter['vel_head'] = patches.Polygon(np.zeros((3, 2)), fc='skyblue', alpha=0.99, zorder=2099, label="Velocity Vector")
        ax.add_patch(painter['vel_head'])
        return painter

    def update_painter(self, painter, time:float, x_state: np.ndarray, u_control: np.ndarray):
        px, py, pz = x_state[0], x_state[2], x_state[4]
        vx, vy = x_state[1], x_state[3]
        phi, theta, psi = x_state[6], x_state[7], x_state[8]
        
        R = self._get_rotation_matrix(phi, theta, psi)
        
        # --- 1. Calculate Local 3D Points (Rotated) ---
        # These are relative to the drone's center of mass (0,0,0)
        p_local = R @ self.body_pts_3d
        
        # --- 2. Calculate Perspective Factors ---
        # We scale the *offsets* based on how close they are to the camera
        z_points = p_local[2, :] + pz
        
        if self.projection_mode == 'orthographic':
            factors = 1.0
            shadow_factor = 1.0
        else:
            # Body Factor: d / (d - z)
            factors = self.camera_dist / (self.camera_dist - np.clip(z_points, -100, self.camera_dist * 0.9))
            # Shadow Factor: d / (d - z_ground)
            shadow_factor = self.camera_dist / (self.camera_dist - self.shadow_z)

        # --- 3. Update Shadows (Projected to Ground) ---
        offset = 0.2 if self.projection_mode == 'orthographic' else 0.2
        
        # Scale local shape by shadow factor (usually ~1.0) and add global pos
        s_offsets = p_local[0:2, :] * shadow_factor
        s_x = s_offsets[0, :] + px + offset
        s_y = s_offsets[1, :] + py + offset
        
        painter['shadow_arm1'].set_data(s_x[[0, 2]], s_y[[0, 2]])
        painter['shadow_arm2'].set_data(s_x[[1, 3]], s_y[[1, 3]])

        # --- 4. Update Body (Projected at Altitude) ---
        # Scale local shape by altitude factor (Get bigger as we go up)
        p_2d_offsets = p_local[0:2, :] * factors
        
        # Add Global Position *AFTER* scaling offsets
        p_world_x = p_2d_offsets[0, :] + px
        p_world_y = p_2d_offsets[1, :] + py
        
        painter['arm1'].set_data(p_world_x[[0, 2]], p_world_y[[0, 2]])
        painter['arm2'].set_data(p_world_x[[1, 3]], p_world_y[[1, 3]])
        
        # --- 5. Update Props ---
        t = np.linspace(0, 2*np.pi, 40)
        base_circle = np.array([self.prop_radius*np.cos(t), self.prop_radius*np.sin(t), np.zeros(40)])
        
        for i in range(4):
            # Calculate Prop Circle in Local Body Frame
            motor_center = self.body_pts_3d[:, i:i+1]
            prop_local_3d = R @ (base_circle + motor_center)
            
            # Perspective Scale for Props
            z_prop = prop_local_3d[2, :] + pz
            
            if self.projection_mode == 'orthographic':
                f_prop = 1.0
            else:
                f_prop = self.camera_dist / (self.camera_dist - np.clip(z_prop, -100, self.camera_dist * 0.9))
            
            # Apply Scale to Offsets -> Add Global Pos
            prop_offsets_2d = prop_local_3d[0:2, :] * f_prop
            r_x = prop_offsets_2d[0, :] + px
            r_y = prop_offsets_2d[1, :] + py
            
            painter['prop_fills'][i].set_xy(np.column_stack((r_x, r_y)))
            painter['prop_lines'][i].set_data(r_x, r_y)
            
            # Shadow Props
            s_prop_offsets = prop_local_3d[0:2, :] * shadow_factor
            painter['shadow_props'][i].set_xy(np.column_stack((
                s_prop_offsets[0, :] + px + offset,
                s_prop_offsets[1, :] + py + offset
            )))

        # --- 6. Heading Arrow ---
        nose_dir_body = np.array([[0], [1], [0]]) 
        nose_local = (R @ nose_dir_body) 
        
        # Use average body factor for arrow scale
        avg_factor = np.mean(factors) if not isinstance(factors, float) else factors
        arrow_len = self.dynamics.l * 3.0 * avg_factor
        
        tip_x = px + nose_local[0,0] * arrow_len
        tip_y = py + nose_local[1,0] * arrow_len
        
        heading_angle = np.arctan2(nose_local[1,0], nose_local[0,0])
        base_arrow = np.array([[0, 0], [-1.0, 1.0], [-1.0, -1.0]]) * self.dynamics.l * avg_factor
        c, s = np.cos(heading_angle), np.sin(heading_angle)
        rot_arrow = (np.array([[c, -s], [s, c]]) @ base_arrow.T).T
        painter['heading'].set_xy(rot_arrow + np.array([tip_x, tip_y]))

        # Velocity Arrow
        v_mag = np.hypot(vx, vy)
        if v_mag > 0.1:
            scale = np.clip(v_mag, 1.0, 4.0)
            # Velocity arrow doesn't need perspective scaling, it's abstract
            v_end_x = px + (vx / v_mag) * scale
            v_end_y = py + (vy / v_mag) * scale
            
            v_angle = np.arctan2(vy, vx)
            c, s = np.cos(v_angle), np.sin(v_angle)
            rot_arrow_v = (np.array([[c, -s], [s, c]]) @ base_arrow.T).T
            
            painter['vel_head'].set_xy(rot_arrow_v + np.array([v_end_x, v_end_y]))
            painter['vel_head'].set_visible(True)
        else:
            painter['vel_head'].set_visible(False)

class QuadcopterV2Perspective(QuadcopterV1Perspective):
    def __init__(self, config: dict, color='k', projection='orthographic', camera_dist=60.0, shadow_z=0.0):
        # 1. Use V2 Dynamics
        dynamics = QuadcopterV2Dynamics(**config)
        Robot.__init__(self, dynamics)
        
        self.color = color
        self.projection_mode = projection
        self.camera_dist = camera_dist
        self.shadow_z = shadow_z
        
        zeros = np.zeros((1, 4))
        self.body_pts_3d = np.vstack([self.dynamics.motor_pos.T, zeros])
        self.prop_radius = self.dynamics.l * 0.4

    def get_telemetry(self, x: np.ndarray, u: np.ndarray) -> str:
        z = x[4]
        vx, vy, vz = x[1], x[3], x[5]
        speed_total = np.sqrt(vx**2 + vy**2 + vz**2)
        ground_speed = np.hypot(vx, vy)
        climb_rate = vz
        
        phi_deg   = np.degrees(x[6])
        theta_deg = np.degrees(x[7])
        psi_deg   = (np.degrees(x[8]) + 180) % 360 - 180 
        
        # 3. Angular Rates (Degrees/sec)
        dphi_deg   = np.degrees(x[7])
        dtheta_deg = np.degrees(x[9])
        dpsi_deg   = np.degrees(x[11])

        # 4. Motors (rad/s -> RPM)
        # 1 rad/s = 60 / (2*pi) RPM ≈ 9.549 RPM
        u_rpm = u * 9.54929658551

        return (f"Alt (Z):    {z:.2f} m\n"
                f"Speed 3D:   {speed_total:.2f} m/s\n"
                f"Gnd Speed:  {ground_speed:.2f} m/s\n"
                f"Climb:      {climb_rate:+.2f} m/s\n"
                f"----------------\n"
                f"Roll:  {phi_deg:5.1f}° ({dphi_deg:5.1f}°/s)\n"
                f"Pitch: {theta_deg:5.1f}° ({dtheta_deg:5.1f}°/s)\n"
                f"Yaw:   {psi_deg:5.1f}° ({dpsi_deg:5.1f}°/s)"
                f"\n----------------\n"
                f"Motors (RPM):\n"
                f"FL: {u_rpm[2]:5.0f}  FR: {u_rpm[3]:5.0f}\n"
                f"BL: {u_rpm[1]:5.0f}  BR: {u_rpm[0]:5.0f}")

# ==========================================
# Visualizer
# ==========================================

class Visualizer:
    def __init__(self, robot: Robot, view_range=15.0):
        self.robot = robot
        self.view_range = view_range

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        #self.ax.grid(True, alpha=0.3)
        self.robot_painter = self.robot.create_painter(self.ax)

    def update(self, t, x_cur, u_cur):
        self.robot.update_painter(self.robot_painter, t, x_cur, u_cur)
        ix, iy = self.robot.pos_indices
        px, py = x_cur[ix], x_cur[iy]
        self.ax.set_xlim(px - self.view_range, px + self.view_range)
        self.ax.set_ylim(py - self.view_range, py + self.view_range)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==========================================
# 3D Trajectory Visualizer (with HUD)
# ==========================================

class FieldTrajectoryVisualizer(Visualizer):
    """
    Visualization with Trajectory History, Prediction, and Telemetry HUD.
    """
    def __init__(self, field, robot: Robot, view_range=15.0):
        super().__init__(robot, view_range)

        self.field = field
        
        # Plot Lines

        self.traj_line, = self.ax.plot([], [], "ow", markersize=2, label="Trajectory")
        self.pred_line, = self.ax.plot([], [], "--w", alpha=0.5, label="Predicted Horizon")
        self.ref_line, = self.ax.plot([], [], "--y", lw=1.5, alpha=0.7, label="Reference")
        
        # HUD Text (Top-Left, fixed to axes coords)
        self.hud_text = self.ax.text(
            0.02, 0.98, "", 
            transform=self.ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        self._draw_field() 
        
        self.ax.legend(loc="upper right")

    def _draw_field(self):
        display, extent = self.field.to_image(resolution=900, return_extent=True)
        img = self.ax.imshow(display, cmap='jet', extent=extent)
        img.set_clim(0.0, 1.0)

    def update(self, t: float, x_cur: np.ndarray, u_cur: np.ndarray, xs_pred=None, traj_history=None, ref_traj=None):
        # 1. Update Base
        super().update(t, x_cur, u_cur)
        
        # 2. Update Lines
        ix, iy = self.robot.pos_indices
        if traj_history is not None and len(traj_history) > 0:
            self.traj_line.set_data(traj_history[:, ix], traj_history[:, iy])
        if xs_pred is not None:
            self.pred_line.set_data(xs_pred[:, ix], xs_pred[:, iy])

        if ref_traj is not None:
            self.ref_line.set_data(ref_traj[:, ix], ref_traj[:, iy])
            
        # 3. Update HUD
        telemetry = self.robot.get_telemetry(x_cur, u_cur)
        hud_content = f"Time: {t:.2f} s\n{telemetry}"
        self.hud_text.set_text(hud_content)
        
        self.ax.set_title(f"Pos: [{x_cur[ix]:.1f}, {x_cur[iy]:.1f}]")

class CityVisualizer(Visualizer):
    """
    Urban city visualizer.
    - Loads OSM data
    - Draw buildings
    - Includes HUD and history/prediction lines
    """
    def __init__(self, robot: Robot,
                 location = "NTU, Singapore",
                 altitude = 50,
                 dist = 500,
                 view_range=20.0,
                 safety_margin = 0.0,
                 feasible_region = False):
        # Initialize base Visualizer (handles fig, ax, and robot_painter)
        super().__init__(robot, view_range)

        self.location_query = location
        self.dist = dist
        self.altitude = altitude
        self.safety_margin = safety_margin
        self.feasible_region = feasible_region

        # Map Data
        self.buildings = None
        self.water = None
        self.center_point = None
        self.map_center_x = 0
        self.map_center_y = 0
        self.map_radius = dist * 0.9

        # Track city artists explicitly
        self.city_patches = [] 

        # Graphics
        self.ax.set_facecolor("#f8f9fa")
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("East-West [m]")
        self.ax.set_ylabel("North-South [m]")
        self.ax.grid(True, linestyle="--", alpha=0.15)
        self.ax.set_title(f"Location: {self.location_query}")

        if self.feasible_region:
            self.constraint_poly = patches.Polygon(
                np.empty((0, 2)), 
                closed=True, 
                facecolor='#20B2AA', 
                edgecolor='#008B8B', 
                alpha=0.2, 
                label='Feasible Region',
                zorder=50 # Draw above map, below robot
            )
            self.ax.add_patch(self.constraint_poly)

        # Plot Lines for Trajectory and Prediction
        self.traj_line, = self.ax.plot([], [], "-.", color="#3F8CD8", alpha=0.8, markersize=2, label="Trajectory")
        self.ref_line, = self.ax.plot([], [], "--", color="#3FB33F", lw=1.0, alpha=0.6, label="Reference")
        self.pred_line, = self.ax.plot([], [], "--", color="#0616A5", lw=1.5, alpha=0.9, label="Predicted")

        # HUD Text (Top-Left, fixed to axes coordinates)
        self.hud_text = self.ax.text(
            0.02, 0.98, "Initializing...", 
            transform=self.ax.transAxes, 
            verticalalignment='top', 
            family='monospace',
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
            zorder=90000
        )

        # Compile building
        ox.settings.use_cache = True
        ox.settings.log_console = False

        # Load & Draw
        print(f"Loading Map Data for {location}...")
        self._load_data()
        self.update_altitude(altitude) # Pre-compute styles
        self._draw_city()
        
        # Adjust legend
        self.ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white")

    def _load_data(self):
        """Downloads and prepares geometric data."""
        if isinstance(self.location_query, str):
            self.center_point = ox.geocode(self.location_query)
        else:
            self.center_point = self.location_query

        # Fetch Data
        tags = {
            "building": True,
            "building:levels": True,
            "building:part": True, 
            "structure": True
        }
        gdf = ox.features_from_point(self.center_point, tags=tags, dist=self.dist)

        # Filter & Project
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        
        # Filter underground if available
        if "location" in gdf.columns:
            gdf = gdf[gdf["location"] != "underground"]
        
        utm_crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm_crs)

        # Calculate Height 
        gdf["calc_height"] = gdf.apply(self._extract_height, axis=1)

        # Center Geometry
        self.map_center_x = gdf.geometry.centroid.x.mean()
        self.map_center_y = gdf.geometry.centroid.y.mean()
        
        gdf["center_dist"] = np.sqrt(
            (gdf.geometry.centroid.x - self.map_center_x)**2 + 
            (gdf.geometry.centroid.y - self.map_center_y)**2
        )
        self.buildings = gdf

        tags_w = {
            "natural": "water", 
            "waterway": ["riverbank", "dock", "canal", "river"]
        }
        try:
            water_gdf = ox.features_from_point(self.center_point, tags=tags_w, dist=self.dist)
            if not water_gdf.empty:
                # Crucial: Project to the exact same CRS as the buildings
                water_gdf = water_gdf.to_crs(utm_crs)
                self.water = water_gdf
        except Exception as e:
            print(f"No water found: {e}")
            self.water = None

    def _extract_height(self, row):
        """Robust height extraction helper."""
        # 1. Try explicit height
        val = row.get("height")
        if pd.notna(val):
            # Extract first number found (e.g. "45m" -> 45.0)
            m = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            if m: return float(m[0])
            
        # 2. Try levels * 3.5m
        levels = row.get("building:levels") or row.get("levels")
        if pd.notna(levels):
            m = re.findall(r"[-+]?\d*\.\d+|\d+", str(levels))
            if m: return float(m[0]) * 3.5
            
        # 3. Fallback
        return 12.0
    
    def _draw_feasible_region(self, A, b, current_pos):
        """
        Helper to compute and update the halfplane intersection polygon.
        """
        if A is None or b is None:
            if self.feasible_region:
                self.constraint_poly.set_visible(False)
            return

        ix, iy = self.robot.pos_indices
        A = A[:, [ix, iy]]

        R = min(100.0, self.view_range) 
        px, py = current_pos
        bounding_box = np.array([
            [1, 0, px + R],     # x <= px + R
            [-1, 0, -(px - R)], # -x <= -(px - R) -> x >= px - R
            [0, 1, py + R],     # y <= py + R
            [0, -1, -(py - R)]  # -y <= -(py - R) -> y >= py - R
        ])

        # 2. Combine Constraints
        # User input is Ax <= b. Scipy expects Ax + c <= 0.
        # So we stack A and -b.
        user_constraints = np.hstack((A, -b.reshape(-1, 1)))
        
        # Stack bounding box (convert box form to Ax - b <= 0 form)
        # Bounding box is already [A_row, b_val], we need [A_row, -b_val]
        box_constraints = np.hstack((bounding_box[:, :2], -bounding_box[:, 2].reshape(-1, 1)))
        
        halfspaces = np.vstack((box_constraints, user_constraints))

        # 3. Compute Intersection
        # We use the current robot position as the interior point.
        # If the robot is currently violating constraints, this might fail.
        interior_point = np.array([px, py])

        if self.feasible_region:
            try:
                hs = HalfspaceIntersection(halfspaces, interior_point)
                verts = hs.intersections
                
                # 4. Sort Vertices (Angular sort around center)
                center = np.mean(verts, axis=0)
                angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
                verts_sorted = verts[np.argsort(angles)]
                
                # Update Artist
                self.constraint_poly.set_xy(verts_sorted)
                self.constraint_poly.set_visible(True)
                
            except Exception as e:
                # If intersection is empty or interior point is outside, hide polygon
                # print(f"Constraint Viz Error: {e}") 
                self.constraint_poly.set_visible(False)

    def update_altitude(self, new_altitude):
        """Recalculates obstacle detection and pre-computes styling."""
        self.altitude = new_altitude
        if self.buildings is None or self.buildings.empty: return

        df = self.buildings
        df["delta_h"] = df["calc_height"] - self.altitude
        df["is_obstacle"] = df["delta_h"] >= 0

        # Colors
        c_below_near = hex_to_rgb("#616161")
        c_below_far = hex_to_rgb("#f2f2f2")
        c_above_dark = hex_to_rgb("#50473A") 
        c_above_obs = hex_to_rgb("#575757")

        colors = np.zeros((len(df), 3))
        mask_obs = df["is_obstacle"].values
        mask_bg = ~mask_obs
        
        # A. Obstacles (Above Altitude)
        t_obs = np.clip(df.loc[mask_obs, "delta_h"] / 50.0, 0.0, 1.0).values[:, None]
        colors[mask_obs] = interpolate_colors(c_above_obs, c_above_dark, t_obs)
        
        # B. Background (Below Altitude)
        t_bg = np.clip(-df.loc[mask_bg, "delta_h"] / 60.0, 0.0, 1.0).values[:, None]
        colors[mask_bg] = interpolate_colors(c_below_near, c_below_far, t_bg)

        df["face_rgb"] = list(colors)
        df["hatch"] = np.where(mask_obs, "///", None)
        df["edge_color"] = np.where(mask_obs, "#ff4400", "#d0d0d0")
        df["edge_width"] = np.where(mask_obs, 2.0, 0.5)
        
        alpha_obs = 0.4 + (0.5 * t_obs.flatten())
        df.loc[mask_obs, "base_alpha"] = alpha_obs
        df.loc[mask_bg, "base_alpha"] = 0.35
        
        df["zorder"] = (df["calc_height"] / 1000.0) + np.where(mask_obs, 100, 1)

    def _draw_city(self):
        """Draws buildings."""
        
        # Clear old city patches explicitly
        for p in self.city_patches:
            p.remove()
        self.city_patches.clear()

        if self.water is not None:
            for geom in self.water.geometry:
                if geom.is_empty: continue

                # Case A: Lines (Rivers/Streams)
                if geom.geom_type in ['LineString', 'MultiLineString']:
                    geoms = [geom] if geom.geom_type == 'LineString' else geom.geoms
                    for ls in geoms:
                        x, y = ls.xy
                        self.ax.plot(
                            np.array(x) - self.map_center_x, 
                            np.array(y) - self.map_center_y, 
                            color="#aed6f1", linewidth=4, alpha=0.6, zorder=0
                        )

                # Case B: Polygons (Lakes/Bays)
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                    for poly in geoms:
                        x, y = poly.exterior.xy
                        pts = np.column_stack((
                            np.array(x) - self.map_center_x, 
                            np.array(y) - self.map_center_y
                        ))
                        self.ax.add_patch(patches.Polygon(
                            pts, closed=True,
                            facecolor="#aed6f1", 
                            linewidth=0, alpha=0.6, zorder=0
                        ))
        
        if self.buildings is None: return
        df_sorted = self.buildings.sort_values("calc_height")

        dists = df_sorted["center_dist"].values
        dist_fades = np.clip(1.0 - (dists / self.map_radius), 0.25, 1.0)

        iterator = zip(
            df_sorted.geometry, df_sorted["is_obstacle"], df_sorted["face_rgb"],
            df_sorted["hatch"], df_sorted["edge_color"], df_sorted["edge_width"],
            df_sorted["base_alpha"], df_sorted["zorder"], dist_fades
        )

        for geom, is_obs, rgb, hatch, ec, lw, base_alpha, z, d_fade in iterator:
            
            final_fade = (0.2 + 0.8 * d_fade) if is_obs else d_fade
            final_alpha = base_alpha * final_fade
            
            if geom.geom_type == 'Polygon': polys = [geom]
            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
            else: continue

            for poly in polys:
                x, y = poly.exterior.xy
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                
                # Fill
                p1 = patches.Polygon(
                    pts, closed=True,
                    facecolor=rgb,
                    edgecolor="#2b2b2b" if is_obs else None,
                    linewidth=0,
                    hatch=hatch,
                    alpha=final_alpha,
                    zorder=z,
                    antialiased=True
                )
                
                self.ax.add_patch(p1)
                self.city_patches.append(p1)
                
                # Edge
                if is_obs:
                    border_lw = lw * min(final_fade, 0.5)
                    border_alpha = 1.0 
                else:
                    border_lw = lw * final_fade
                    border_alpha = final_alpha
                    
                p2 = patches.Polygon(
                    pts, closed=True,
                    facecolor="none",
                    edgecolor=ec,
                    linewidth=border_lw,
                    alpha=border_alpha,
                    zorder=z + 0.1,
                    antialiased=True
                )
                self.ax.add_patch(p2)
                self.city_patches.append(p2)

                if is_obs and self.safety_margin > 0:
                    # Create buffered geometry (resolution=4 keeps it simple/fast)
                    buffered_poly = poly.buffer(self.safety_margin, resolution=4)
                    bx, by = buffered_poly.exterior.xy
                    b_pts = np.column_stack((np.array(bx) - self.map_center_x, np.array(by) - self.map_center_y))

                    # Create Dashed Outline Patch
                    # We use a slightly lower zorder so it appears "around" the base
                    boundary_style = dict(
                        facecolor='none',
                        edgecolor='red', # Hazard color
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.6,
                        zorder=z - 0.1 # Draw slightly behind the solid building,
                    )

                    # Add to Main
                    boundary = patches.Polygon(b_pts, closed=True, **boundary_style)
                    self.ax.add_patch(boundary)
                    self.city_patches.append(boundary)

    def get_obstacle_rgjs(self):
        """Exports current obstacles to RGJ format for PotentialField."""
        if self.buildings is None: return []
        
        obstacles = self.buildings[self.buildings["is_obstacle"]].geometry
        rgjs = []
        
        for geom in obstacles:
            if geom.geom_type == 'Polygon': polys = [geom]
            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
            else: continue
            
            for poly in polys:
                x, y = poly.convex_hull.exterior.xy
                # Important: Shift coordinates to match the visualization frame (centered at 0,0)
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                
                if not np.allclose(pts[0], pts[-1]):
                    pts = np.vstack([pts, pts[0]])
                    
                rgjs.append({
                    "type": "Polygon",
                    "coordinates": pts.tolist(),
                    "repulsion": [[20.0, 0], [0, 20.0]] # Default repulsion
                })
        return rgjs

    def update(self, t: float, x_cur: np.ndarray, u_cur: np.ndarray, xs_pred=None, traj_history=None, ref_traj=None, A_constraint=None, B_constraint=None):
        """
        Overlays trajectory and HUD data on top of the base robot update.
        """
        # 1. Update Base (Updates the robot artists and camera window)
        super().update(t, x_cur, u_cur)

        # 2. Update Feasible Region
        ix, iy = self.robot.pos_indices
        current_pos = (x_cur[ix], x_cur[iy])
        self._draw_feasible_region(A_constraint, B_constraint, current_pos)

        # 3. Update Trajectory Lines
        if traj_history is not None and len(traj_history) > 0:
            self.traj_line.set_data(traj_history[:, ix], traj_history[:, iy])
        if xs_pred is not None:
            self.pred_line.set_data(xs_pred[:, ix], xs_pred[:, iy])
        if ref_traj is not None:
            self.ref_line.set_data(ref_traj[:, ix], ref_traj[:, iy])
            
        # 4. Update HUD
        telemetry = self.robot.get_telemetry(x_cur, u_cur)
        obs_count = self.buildings["is_obstacle"].sum() if self.buildings is not None else 0
        hud_content = f"Time: {t:.2f} s | Obs: {obs_count}\n{telemetry}"
        self.hud_text.set_text(hud_content)

        # 5. Update Plot Title
        self.ax.set_title(f"Position: [{x_cur[ix]:.1f}, {x_cur[iy]:.1f}]")


class ZoomedCityVisualizer(Visualizer):
    """
    Urban city visualizer with a Zoomed-In Inset (Picture-in-Picture).
    """
    def __init__(self, robot: Robot,
                 location = "NTU, Singapore",
                 altitude = 50,
                 dist = 500,
                 view_range=50.0,        # Main Camera Range
                 zoomed_view_range=10.0,  # Inset Camera Range
                 safety_margin=0.0,
                 feasible_region = False
                 ):
        
        # 1. Initialize Base (Creates self.fig, self.ax)
        super().__init__(robot, view_range)

        self.location_query = location
        self.dist = dist
        self.altitude = altitude
        self.zoom_range = zoomed_view_range
        self.safety_margin = safety_margin
        self.feasible_region = feasible_region

        # 2. Create Inset Axis (Bottom Right, 35% size)
        # Changed loc to "lower right"
        self.ax_zoom = inset_axes(self.ax, width="35%", height="35%", loc="lower right")
        
        # Optional: Add a white background with alpha so map doesn't bleed through perfectly
        self.ax_zoom.patch.set_alpha(0.8) 
        
        self.ax_zoom.set_aspect('equal')
        self.ax.set_xlabel("East-West [m]")
        self.ax.set_ylabel("North-South [m]")
        self.ax_zoom.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_title(f"Location: {self.location_query}")
        
        # Hide tick labels on zoom for cleanliness
        self.ax_zoom.set_xticks([])
        self.ax_zoom.set_yticks([])
        
        # Connect zoom box to main plot (Draws the "zoom lines")
        # For a Bottom-Right box, connecting corner 1 (Top Right) and 3 (Bottom Left) 
        # or 2 (Top Left) usually looks best.
        self.zoom_connector = mark_inset(self.ax, self.ax_zoom, loc1=1, loc2=3, fc="none", ec="0.5", alpha=0.5, zorder=300)

        # 3. Create Painters for BOTH axes
        # We must instantiate separate artists for the main and zoom views
        self.robot_painter_main = self.robot_painter # Created by super()
        self.robot_painter_zoom = self.robot.create_painter(self.ax_zoom)


        if self.feasible_region:
            self.constraint_poly = patches.Polygon(
                np.empty((0, 2)), 
                closed=True, 
                facecolor='#20B2AA', 
                edgecolor='#008B8B', 
                alpha=0.2, 
                label='Feasible Region',
                zorder=50 
            )
            self.ax.add_patch(self.constraint_poly)

        # 4. Trajectory Lines (Dual)
        self.traj_line_main, = self.ax.plot([], [], ".-", color="#3F8CD8", alpha=0.6, markersize=2, label="Trajectory")
        self.ref_line_main, = self.ax.plot([], [], "--", color="#3FB33F", lw=1.0, alpha=0.8, label="Reference")
        self.pred_line_main, = self.ax.plot([], [], "--", color="#0616A5", lw=1.5, alpha=0.99, label="Predicted")

        self.traj_line_zoom, = self.ax_zoom.plot([], [], "-.", color="#3F8CD8", alpha=0.6, markersize=2)
        self.ref_line_zoom, = self.ax_zoom.plot([], [], "--", color="#3FB33F", lw=1.0, alpha=0.8)
        self.pred_line_zoom, = self.ax_zoom.plot([], [], "--", color="#0616A5", lw=1.5, alpha=0.99)

        # 5. Map Data
        self.buildings = None
        self.water = None
        self.center_point = None
        self.map_center_x = 0; self.map_center_y = 0
        self.map_radius = dist * 0.6

        # Store patches for both views
        self.city_patches_main = []
        self.city_patches_zoom = []

        # Graphics Setup
        self.ax.set_facecolor("#f8f9fa")
        self.ax_zoom.set_facecolor("#f8f9fa")
        
        # HUD Text (Top-Left of MAIN axis, stays put)
        self.hud_text = self.ax.text(
            0.02, 0.98, "Initializing...", 
            transform=self.ax.transAxes, 
            verticalalignment='top', 
            family='monospace',
            bbox=dict(boxstyle="round", facecolor="white", alpha=1.0),
            zorder=90000
        )

        # Load & Draw
        ox.settings.use_cache = True
        ox.settings.log_console = False
        print(f"Loading Map Data for {location}...")
        self._load_data()
        self.update_altitude(altitude)
        self._draw_city_complete()
        
        # Move Legend to Upper Left since Zoom is in Lower Right
        self.ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white")

    def _load_data(self):
        """Downloads and prepares geometric data."""
        if isinstance(self.location_query, str):
            self.center_point = ox.geocode(self.location_query)
        else:
            self.center_point = self.location_query

        tags = {"building": True, "building:levels": True, "building:part": True, "structure": True}
        gdf = ox.features_from_point(self.center_point, tags=tags, dist=self.dist)
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        
        if "location" in gdf.columns:
            gdf = gdf[gdf["location"] != "underground"]
            
        utm_crs = gdf.estimate_utm_crs()
        gdf = gdf.to_crs(utm_crs)

        gdf["calc_height"] = gdf.apply(self._extract_height, axis=1)

        self.map_center_x = gdf.geometry.centroid.x.mean()
        self.map_center_y = gdf.geometry.centroid.y.mean()
        
        gdf["center_dist"] = np.sqrt(
            (gdf.geometry.centroid.x - self.map_center_x)**2 + 
            (gdf.geometry.centroid.y - self.map_center_y)**2
        )
        self.buildings = gdf

        tags_w = {
            "natural": "water", 
            "waterway": ["riverbank", "dock", "canal", "river"]
        }
        try:
            water_gdf = ox.features_from_point(self.center_point, tags=tags_w, dist=self.dist)
            if not water_gdf.empty:
                # Crucial: Project to the exact same CRS as the buildings
                water_gdf = water_gdf.to_crs(utm_crs)
                self.water = water_gdf
        except Exception as e:
            print(f"No water found: {e}")
            self.water = None

    def _extract_height(self, row):
        val = row.get("height")
        if pd.notna(val):
            m = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
            if m: return float(m[0])
        levels = row.get("building:levels") or row.get("levels")
        if pd.notna(levels):
            m = re.findall(r"[-+]?\d*\.\d+|\d+", str(levels))
            if m: return float(m[0]) * 3.5
        return 12.0
    
    def _draw_feasible_region(self, A, b, current_pos):
        """
        Helper to compute and update the halfplane intersection polygon.
        """
        if A is None or b is None:
            if self.feasible_region:
                self.constraint_poly.set_visible(False)
            return

        ix, iy = self.robot.pos_indices
        A = A[:, [ix, iy]]

        R = min(100.0, self.view_range) 
        px, py = current_pos
        bounding_box = np.array([
            [1, 0, px + R],     # x <= px + R
            [-1, 0, -(px - R)], # -x <= -(px - R) -> x >= px - R
            [0, 1, py + R],     # y <= py + R
            [0, -1, -(py - R)]  # -y <= -(py - R) -> y >= py - R
        ])

        # 2. Combine Constraints
        # User input is Ax <= b. Scipy expects Ax + c <= 0.
        # So we stack A and -b.
        user_constraints = np.hstack((A, -b.reshape(-1, 1)))
        
        # Stack bounding box (convert box form to Ax - b <= 0 form)
        # Bounding box is already [A_row, b_val], we need [A_row, -b_val]
        box_constraints = np.hstack((bounding_box[:, :2], -bounding_box[:, 2].reshape(-1, 1)))
        
        halfspaces = np.vstack((box_constraints, user_constraints))

        # 3. Compute Intersection
        # We use the current robot position as the interior point.
        # If the robot is currently violating constraints, this might fail.
        interior_point = np.array([px, py])

        if self.feasible_region:
            try:
                hs = HalfspaceIntersection(halfspaces, interior_point)
                verts = hs.intersections
                
                # 4. Sort Vertices (Angular sort around center)
                center = np.mean(verts, axis=0)
                angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
                verts_sorted = verts[np.argsort(angles)]
                
                # Update Artist
                self.constraint_poly.set_xy(verts_sorted)
                self.constraint_poly.set_visible(True)
                
            except Exception as e:
                # If intersection is empty or interior point is outside, hide polygon
                print(f"Constraint Viz Error: {e}") 
                self.constraint_poly.set_visible(False)

    def update_altitude(self, new_altitude):
        self.altitude = new_altitude
        if self.buildings is None or self.buildings.empty: return

        df = self.buildings
        df["delta_h"] = df["calc_height"] - self.altitude
        df["is_obstacle"] = df["delta_h"] >= 0

        c_below_near = hex_to_rgb("#616161")
        c_below_far = hex_to_rgb("#f2f2f2")
        c_above_dark = hex_to_rgb("#50473A") 
        c_above_obs = hex_to_rgb("#575757")

        colors = np.zeros((len(df), 3))
        mask_obs = df["is_obstacle"].values
        mask_bg = ~mask_obs
        
        t_obs = np.clip(df.loc[mask_obs, "delta_h"] / 50.0, 0.0, 1.0).values[:, None]
        colors[mask_obs] = interpolate_colors(c_above_obs, c_above_dark, t_obs)
        
        t_bg = np.clip(-df.loc[mask_bg, "delta_h"] / 60.0, 0.0, 1.0).values[:, None]
        colors[mask_bg] = interpolate_colors(c_below_near, c_below_far, t_bg)

        df["face_rgb"] = list(colors)
        df["hatch"] = np.where(mask_obs, "///", None)
        df["edge_color"] = np.where(mask_obs, "#ff4400", "#d0d0d0")
        df["edge_width"] = np.where(mask_obs, 2.0, 0.5)
        
        alpha_obs = 0.4 + (0.5 * t_obs.flatten())
        df.loc[mask_obs, "base_alpha"] = alpha_obs
        df.loc[mask_bg, "base_alpha"] = 0.35
        df["zorder"] = (df["calc_height"] / 1000.0) + np.where(mask_obs, 100, 1)

    def _draw_city_complete(self):
        """Draws buildings and water on BOTH axes."""
        # 1. Clear old artists
        for p in self.city_patches_main: p.remove()
        for p in self.city_patches_zoom: p.remove()
        self.city_patches_main.clear()
        self.city_patches_zoom.clear()

        # ==========================================
        # DRAW WATER
        # ==========================================
        if self.water is not None:
            # Common style for water
            water_style = dict(color="#aed6f1", linewidth=4, alpha=0.6, zorder=0)
            poly_style = dict(facecolor="#aed6f1", linewidth=0, alpha=0.6, zorder=0)

            for geom in self.water.geometry:
                if geom.is_empty: continue

                # --- 1. Identify Type and Normalize Coordinates ---
                geoms_to_process = []
                is_line = False

                if geom.geom_type == 'LineString':
                    geoms_to_process = [geom]
                    is_line = True
                elif geom.geom_type == 'MultiLineString':
                    geoms_to_process = geom.geoms
                    is_line = True
                elif geom.geom_type == 'Polygon':
                    geoms_to_process = [geom]
                elif geom.geom_type == 'MultiPolygon':
                    geoms_to_process = geom.geoms

                # --- 2. Process & Draw on BOTH axes ---
                for g in geoms_to_process:
                    # OPTIMIZATION: Calculate coordinates ONCE
                    if is_line:
                        x, y = g.xy
                        pts_x = np.array(x) - self.map_center_x
                        pts_y = np.array(y) - self.map_center_y
                        
                        # Draw Main
                        l1, = self.ax.plot(pts_x, pts_y, **water_style)
                        self.city_patches_main.append(l1)
                        
                        # Draw Zoom (Must create new artist)
                        l2, = self.ax_zoom.plot(pts_x, pts_y, **water_style)
                        self.city_patches_zoom.append(l2)
                    else:
                        # Polygon
                        x, y = g.exterior.xy
                        pts = np.column_stack((
                            np.array(x) - self.map_center_x, 
                            np.array(y) - self.map_center_y
                        ))
                        
                        # Draw Main
                        p1 = patches.Polygon(pts, closed=True, **poly_style)
                        self.ax.add_patch(p1)
                        self.city_patches_main.append(p1)
                        
                        # Draw Zoom (Must create new artist)
                        p2 = patches.Polygon(pts, closed=True, **poly_style)
                        self.ax_zoom.add_patch(p2)
                        self.city_patches_zoom.append(p2)

        # ==========================================
        # DRAW BUILDINGS
        # ==========================================
        if self.buildings is None: return
        
        # To optimize, we sort once and iterate once
        df_sorted = self.buildings.sort_values("calc_height")
        dists = df_sorted["center_dist"].values
        dist_fades = np.clip(1.0 - (dists / self.map_radius), 0.25, 1.0)

        iterator = zip(
            df_sorted.geometry, df_sorted["is_obstacle"], df_sorted["face_rgb"],
            df_sorted["hatch"], df_sorted["edge_color"], df_sorted["edge_width"],
            df_sorted["base_alpha"], df_sorted["zorder"], dist_fades
        )

        for geom, is_obs, rgb, hatch, ec, lw, base_alpha, z, d_fade in iterator:
            
            final_fade = (0.2 + 0.8 * d_fade) if is_obs else d_fade
            final_alpha = base_alpha * final_fade
            
            if geom.geom_type == 'Polygon': polys = [geom]
            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
            else: continue

            for poly in polys:
                x, y = poly.exterior.xy
                # OPTIMIZATION: Calculate pts ONCE
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                
                # --- Create Patch for MAIN Axis ---
                p_main = self._create_building_patch(pts, rgb, is_obs, hatch, final_alpha, z, ec, lw, final_fade)
                self.ax.add_patch(p_main)
                self.city_patches_main.append(p_main)
                
                # --- Create Patch for ZOOM Axis ---
                # We must create a new object (Matplotlib requirement)
                p_zoom = self._create_building_patch(pts, rgb, is_obs, hatch, final_alpha, z, ec, lw, final_fade)
                self.ax_zoom.add_patch(p_zoom)
                self.city_patches_zoom.append(p_zoom)

                if is_obs and self.safety_margin > 0:
                    # Create buffered geometry (resolution=4 keeps it simple/fast)
                    buffered_poly = poly.buffer(self.safety_margin, resolution=4)
                    bx, by = buffered_poly.exterior.xy
                    b_pts = np.column_stack((np.array(bx) - self.map_center_x, np.array(by) - self.map_center_y))

                    # Create Dashed Outline Patch
                    boundary_style = dict(
                        facecolor='none',
                        edgecolor='red', 
                        linestyle='--',
                        linewidth=1.0,
                        alpha=0.6,
                        zorder=z - 0.1 
                    )

                    # Add to Main
                    b_main = patches.Polygon(b_pts, closed=True, **boundary_style)
                    self.ax.add_patch(b_main)
                    self.city_patches_main.append(b_main)

                    # Add to Zoom
                    b_zoom = patches.Polygon(b_pts, closed=True, **boundary_style)
                    self.ax_zoom.add_patch(b_zoom)
                    self.city_patches_zoom.append(b_zoom)

    def _create_building_patch(self, pts, rgb, is_obs, hatch, alpha, z, ec, lw, fade):
        return patches.Polygon(
            pts, closed=True,
            facecolor=rgb,
            edgecolor=ec,
            linewidth=lw * (min(fade, 0.8) if is_obs else fade),
            hatch=hatch,
            alpha=alpha,
            zorder=z,
            antialiased=True
        )

    def get_obstacle_rgjs(self):
        if self.buildings is None: return []
        obstacles = self.buildings[self.buildings["is_obstacle"]].geometry
        rgjs = []
        for geom in obstacles:
            if geom.geom_type == 'Polygon': polys = [geom]
            elif geom.geom_type == 'MultiPolygon': polys = geom.geoms
            else: continue
            for poly in polys:
                x, y = poly.convex_hull.exterior.xy
                pts = np.column_stack((np.array(x) - self.map_center_x, np.array(y) - self.map_center_y))
                if not np.allclose(pts[0], pts[-1]): pts = np.vstack([pts, pts[0]])
                rgjs.append({"type": "Polygon", "coordinates": pts.tolist(), "repulsion": [[20.0, 0], [0, 20.0]]})
        return rgjs

    def update(self, t: float, x_cur: np.ndarray, u_cur: np.ndarray, xs_pred=None, traj_history=None, ref_traj=None, A_constraint=None, B_constraint=None, **kwargs):
        """
        Updates both Main and Zoom views.
        """
        # 1. Update Robot Painters (Main & Zoom)
        self.robot.update_painter(self.robot_painter_main, t, x_cur, u_cur)
        self.robot.update_painter(self.robot_painter_zoom, t, x_cur, u_cur)

        # 2. Update Indices
        ix, iy = self.robot.pos_indices
        #px, py = xs_pred[0, [ix, iy]]
        px, py = x_cur[ix], x_cur[iy]

        self._draw_feasible_region(A_constraint, B_constraint, (px, py))

        # 3. Update Trajectory Lines (Both)
        if traj_history is not None and len(traj_history) > 0:
            self.traj_line_main.set_data(traj_history[:, ix], traj_history[:, iy])
            self.traj_line_zoom.set_data(traj_history[:, ix], traj_history[:, iy])
            
        if xs_pred is not None:
            self.pred_line_main.set_data(xs_pred[:, ix], xs_pred[:, iy])
            self.pred_line_zoom.set_data(xs_pred[:, ix], xs_pred[:, iy])
            
        if ref_traj is not None:
            self.ref_line_main.set_data(ref_traj[:, ix], ref_traj[:, iy])
            self.ref_line_zoom.set_data(ref_traj[:, ix], ref_traj[:, iy])

        # 4. Update Cameras
        # Main Camera
        self.ax.set_xlim(px - self.view_range, px + self.view_range)
        self.ax.set_ylim(py - self.view_range, py + self.view_range)
        
        # Zoom Camera
        self.ax_zoom.set_xlim(px - self.zoom_range, px + self.zoom_range)
        self.ax_zoom.set_ylim(py - self.zoom_range, py + self.zoom_range)

        # 5. HUD & Title
        telemetry = self.robot.get_telemetry(x_cur, u_cur)
        obs_count = self.buildings["is_obstacle"].sum() if self.buildings is not None else 0
        hud_content = f"Time: {t:.2f} s | Obs: {obs_count}\n{telemetry}"
        self.hud_text.set_text(hud_content)
        #self.ax.set_title(f"Pos: [{px:.1f}, {py:.1f}]")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()