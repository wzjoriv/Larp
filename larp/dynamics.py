from typing import Dict, List, Optional, Tuple
import numpy as np
from itertools import chain

"""
Author: Josue N Rivera
"""
        
class Dynamics():

    def __init__(self, 
                 constants:Optional[Dict] = None,
                 state_derivative_orders:List[int] = [1],
                 control_derivative_orders:List[int] = [0]) -> None:

        self.constants = constants

        # Maximum derivitive order for each primitive state needed to represent the system's state vector (same for control)
        
        self.state_derivative_orders = np.array(state_derivative_orders, dtype=int)
        self.control_derivative_orders = np.array(control_derivative_orders, dtype=int)
        
        # Privitive state devitive order for each system state
        self.first_state_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.state_derivative_orders])
        self.first_control_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.control_derivative_orders])
        
        # One order higher of the max derivitive order for each primitive state needed to form the ode
        self.highest_state_order = max(self.state_derivative_orders)
        self.highest_control_order = max(self.control_derivative_orders)
        
        self.state_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in state_derivative_orders])), dtype=bool).reshape(-1)
        self.control_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in control_derivative_orders])), dtype=bool).reshape(-1)

        self.primitive_state_n, self.primitive_control_n = (len(state_derivative_orders), len(control_derivative_orders))

        self.first_order_state_n, self.first_order_control_n = (sum(state_derivative_orders) + self.primitive_state_n, sum(control_derivative_orders) + self.primitive_control_n)
    
    def split_first(self, first:np.ndarray):
        return tuple([first[:, i:i+1] for i in range(first.shape[1])])
    
    def f(self,
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdx(self,
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdu(self,
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def linearize(self,
             x0: np.ndarray,
             u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearizes the dynamics around the reference points (x0, u0).

        Returns:
            A: Jacobian of f with respect to x at (x0, u0)
            B: Jacobian of f with respect to u at (x0, u0)
            f0: f(x0, u0)
        """
        
        A = self.dfdx(x0, u0)
        B = self.dfdu(x0, u0)
        f0 = self.f(x0, u0)

        return A, B, f0
    
    def discretize_linear(self, x0, u0, dt:float=0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearizes and discretize the dynamics around the reference point (x0, u0).

        Returns:
            Ad: Jacobian of f with respect to x at (x0, u0)
            Bd: Jacobian of f with respect to u at (x0, u0)
        """
        
        A, B, f0 = self.linearize(x0, u0)
        
        Ad = np.eye(self.first_order_state_n) + dt*A
        Bd = dt*B

        return Ad, Bd
        
    def first_state_names(self) -> List[str]:
        "Returns a text label for state names"

        orders = self.state_derivative_orders

        return [f'x_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_control_names(self) -> List[str]:
    
        orders = self.control_derivative_orders
        return [f'u_{{{orders[o_idx]}}}^{{[{i}]}}' for o_idx in range(orders) for i in range(orders[o_idx]+1)]
        
    def first_names(self) -> Tuple[List[str], List[str]]:    
        return self.first_state_names(), self.first_control_names()
    
class WMRDynamics(Dynamics):

    """
    Dynamics for Wheeled Mobile Robot

    - Kuhne, F., Lages, W. F., & da Silva Jr, J. G. (2004, September). Model predictive control of a mobile robot using linearization. In Proceedings of mechatronics and robotics (Vol. 4, No. 4, pp. 525-530).
    - 
        
    Constants:
        wheels distance (wd; default: 1.0 m)

    """

    def __init__(self, wheels_distance=1.0) -> None:

        constants = {
            'wheels distance': wheels_distance
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[0, 0, 0],
                         control_derivative_orders=[0, 0])
        
        self.wd = self.constants['wheels distance']
        
    def extract_wheel_speed(self, first_order_control):
        v, w = self.split_first(first_order_control)
        
        v_l = v - self.wd*w
        v_r = v + self.wd*w

        return np.concatenate([v_l, v_r], axis=0)

    def f(self,
          
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        _, _, theta = self.split_first(first_order_state)

        v, w = self.split_first(first_order_control)
        
        dx = v*np.cos(theta)
        dy = v*np.sin(theta)
        dtheta = w

        return np.concatenate([dx, dy, dtheta], axis=1)
    
    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:

        _, _, theta = self.split_first(first_order_state)
        v, w = self.split_first(first_order_control)

        df1 = np.concatenate([np.cos(theta), np.zeros_like(w)], axis=1)
        df2 = np.concatenate([np.sin(theta), np.zeros_like(w)], axis=1)
        df3 = np.concatenate([np.zeros_like(v), np.ones_like(w)], axis=1)

        return np.stack([df1, df2, df3], axis=1)
    
    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:

        x, _, theta = self.split_first(first_order_state)
        v, _ = self.split_first(first_order_control)

        zeros = np.zeros_like(x)

        df1 = np.concatenate([zeros, zeros, -v*np.sin(theta)], axis=1)
        df2 = np.concatenate([zeros, zeros, v*np.cos(theta)], axis=1)
        df3 = np.concatenate([zeros, zeros, zeros], axis=1)

        return np.stack([df1, df2, df3], axis=1)

class Quadcopter2DDynamics(Dynamics):
    """
    2D Quadcopter Dynamics
    
    States:
        x      - horizontal position
        y      - vertical position
        theta  - orientation angle
        dx     - horizontal velocity
        dy     - vertical velocity
        dtheta - angular velocity
    
    Controls:
        u1     - left rotor thrust
        u2     - right rotor thrust

    Constants:
        m  - mass
        g  - gravitational acceleration
        L  - distance from center to rotor
        I  - moment of inertia
    """

    def __init__(self, mass=1.0, gravity=9.81, arm_length=0.5, inertia=0.01) -> None:
        constants = {
            'mass': mass,
            'gravity': gravity,
            'arm_length': arm_length,
            'inertia': inertia
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[1, 1, 1],
                         control_derivative_orders=[0, 0])

        self.m = mass
        self.g = gravity
        self.L = arm_length
        self.I = inertia

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        x, y, theta, dx, dy, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        ddx = -(u1 + u2) * np.sin(theta) / self.m
        ddy = (u1 + u2) * np.cos(theta) / self.m - self.g
        ddtheta = self.L * (u1 - u2) / self.I

        return np.concatenate([dx, dy, dtheta, ddx, ddy, ddtheta], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        x, y, theta, dx, dy, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        zero = np.zeros_like(x)
        one = np.ones_like(x)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        total_u = u1 + u2

        dddx_dtheta = -total_u * cos_theta / self.m
        dddy_dtheta = -total_u * sin_theta / self.m

        # Each row is ∂f[i]/∂x_j
        df = [
            np.concatenate([zero, zero, zero, one, zero, zero], axis=1),  # ∂dx/∂x
            np.concatenate([zero, zero, zero, zero, one, zero], axis=1),  # ∂dy/∂x
            np.concatenate([zero, zero, zero, zero, zero, one], axis=1),  # ∂dtheta/∂x
            np.concatenate([zero, zero, dddx_dtheta, zero, zero, zero], axis=1),  # ∂ddx/∂x
            np.concatenate([zero, zero, dddy_dtheta, zero, zero, zero], axis=1),  # ∂ddy/∂x
            np.concatenate([zero, zero, zero, zero, zero, zero], axis=1),         # ∂ddtheta/∂x
        ]
        return np.stack(df, axis=1)

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        x, y, theta, dx, dy, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        zero = np.zeros_like(u1)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        dddx_du = -sin_theta / self.m
        dddy_du = cos_theta / self.m
        ddtheta_du1 = self.L / self.I
        ddtheta_du2 = -self.L / self.I

        df = [
            np.concatenate([zero, zero], axis=1),  # ∂dx/∂u
            np.concatenate([zero, zero], axis=1),  # ∂dy/∂u
            np.concatenate([zero, zero], axis=1),  # ∂dtheta/∂u
            np.concatenate([dddx_du, dddx_du], axis=1),  # ∂ddx/∂u
            np.concatenate([dddy_du, dddy_du], axis=1),  # ∂ddy/∂u
            np.concatenate([np.full_like(u1, ddtheta_du1), np.full_like(u2, ddtheta_du2)], axis=1),  # ∂ddtheta/∂u
        ]
        return np.stack(df, axis=1)

class QuadcopterV1Dynamics(Dynamics):
    """
    Quadcopter Dynamics V2 model.
    
    States:
        - x, y, z (position)
        - vx, vy, vz (linear velocity)
        - phi, theta, psi (roll, pitch, yaw)
        - p, q, r (angular velocity)

    Controls:
        - u1: total thrust
        - u2, u3, u4: body torques (roll, pitch, yaw)
    
    Constants:
        - m: mass
        - g: gravitational acceleration
        - Ix, Iy, Iz: moments of inertia
    """

    def __init__(self, mass=1.0, gravity=9.81, Ix=0.01, Iy=0.01, Iz=0.02):
        constants = {
            'mass': mass,
            'gravity': gravity,
            'Ix': Ix,
            'Iy': Iy,
            'Iz': Iz
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[1]*6 + [1]*6,  # 12 states: pos(3), angles(3), ang vel(3)
                         control_derivative_orders=[0]*4)       # 4 controls: thrust, 3 torques

        self.m = mass
        self.g = gravity
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = self.split_first(first_order_state)
        u1, u2, u3, u4 = self.split_first(first_order_control)

        m, g, Ix, Iy, Iz = self.m, self.g, self.Ix, self.Iy, self.Iz

        # Derivatives
        dx = vx
        dy = vy
        dz = vz

        dvx = u1 / m * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
        dvy = u1 / m * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
        dvz = u1 / m * (np.cos(phi) * np.cos(theta)) - g

        dphi = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)

        dp = (u2 + (Iy - Iz) * q * r) / Ix
        dq = (u3 + (Iz - Ix) * p * r) / Iy
        dr = (u4 + (Ix - Iy) * p * q) / Iz

        return np.concatenate([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:

        _, _, _, _, _, _, phi, theta, psi, p, q, r = self.split_first(first_order_state)
        u1, _, _, _ = self.split_first(first_order_control)

        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz

        dfdx = np.zeros((first_order_state.shape[0], 12, 12))

        # Partial derivatives wrt theta, phi, psi for acceleration
        dfdx[:, 3, 6] = -u1[:, 0] / m * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi))
        dfdx[:, 3, 7] =  u1[:, 0] / m * (np.cos(phi) * np.cos(theta) * np.cos(psi))
        dfdx[:, 3, 8] = -u1[:, 0] / m * (np.cos(phi) * np.sin(theta) * np.sin(psi) + np.sin(phi) * np.cos(psi))

        dfdx[:, 4, 6] = -u1[:, 0] / m * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi))
        dfdx[:, 4, 7] =  u1[:, 0] / m * (np.cos(phi) * np.cos(theta) * np.sin(psi))
        dfdx[:, 4, 8] =  u1[:, 0] / m * (np.cos(phi) * np.sin(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi))

        dfdx[:, 5, 6] = -u1[:, 0] / m * (np.sin(phi) * np.cos(theta))
        dfdx[:, 5, 7] = -u1[:, 0] / m * (np.cos(phi) * np.sin(theta))

        # Angular dynamics Jacobian (simplified, no cross terms)
        dfdx[:, 6, 9] = 1
        dfdx[:, 6, 10] = np.sin(phi) * np.tan(theta)
        dfdx[:, 6, 11] = np.cos(phi) * np.tan(theta)

        dfdx[:, 7, 10] = np.cos(phi)
        dfdx[:, 7, 11] = -np.sin(phi)

        dfdx[:, 8, 10] = np.sin(phi) / np.cos(theta)
        dfdx[:, 8, 11] = np.cos(phi) / np.cos(theta)

        dfdx[:, 9, 10] = (Iy - Iz) * r / Ix
        dfdx[:, 9, 11] = (Iy - Iz) * q / Ix
        dfdx[:, 10, 9] = (Iz - Ix) * r / Iy
        dfdx[:, 10, 11] = (Iz - Ix) * p / Iy
        dfdx[:, 11, 9] = (Ix - Iy) * q / Iz
        dfdx[:, 11, 10] = (Ix - Iy) * p / Iz

        # Derivatives of x,y,z wrt vx,vy,vz
        dfdx[:, 0, 3] = 1
        dfdx[:, 1, 4] = 1
        dfdx[:, 2, 5] = 1

        return dfdx

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        phi, theta, psi = self.split_first(first_order_state)[6:9]
        u1, u2, u3, u4 = self.split_first(first_order_control)
        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz

        dfdu = np.zeros((first_order_state.shape[0], 12, 4))

        # ∂f/∂u1 (affects acceleration)
        dfdu[:, 3, 0] = 1 / m * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))[:, 0]
        dfdu[:, 4, 0] = 1 / m * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))[:, 0]
        dfdu[:, 5, 0] = 1 / m * (np.cos(phi) * np.cos(theta))[:, 0]

        # ∂f/∂u2..u4 (affects angular acceleration)
        dfdu[:, 9, 1] = 1 / Ix
        dfdu[:, 10, 2] = 1 / Iy
        dfdu[:, 11, 3] = 1 / Iz

        return dfdu
    
class QuadcopterV2Dynamics(Dynamics):
    """
    Quadcopter dynamics using individual rotor speeds as control inputs.
    
    """

    def __init__(self, 
                 inertia = [3.8e-3, 3.8e-3, 7.1e-3],
                 mass = 5.2,
                 gravity = 9.807,
                 arm_length = 0.32,
                 thrust_constant = 3.13e-5,
                 translational_drag = [0.1, 0.1, 0.15],
                 torque_constant = 7.5e-7,
                 rotational_drag = [0.1, 0.1, 0.15],
                 motor_inertia = 6e-5) -> None:

        constants = {
            'inertia': inertia,
            'mass': mass,
            'gravity': gravity,
            'arm length': arm_length,
            'thrust constant': thrust_constant,
            'translational drag': translational_drag,
            'torque constant': torque_constant,
            'rotational drag': rotational_drag,
            'motor inertia': motor_inertia
        }

        super().__init__(constants,
                         state_derivative_orders=[1]*6,
                         control_derivative_orders=[0]*4)
        
        self.m = mass
        self.I = inertia
        self.g = gravity
        self.l = arm_length
        self.b = thrust_constant
        self.d = torque_constant
        self.Ct = translational_drag
        self.Cr = rotational_drag
        self.Jr = motor_inertia

        self.M = np.array([
            [],
            [],
            [],
            []
        ])

    def extract_force(self, first_order_control):

        """ Convert vector of controls to """

        w_2 = first_order_control**2
        
        return w_2@self.M

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        # State
        _, dx, _, dy, _, dz, phi, dphi, theta, dtheta, psi, dpsi = self.split_first(first_order_state)
        # Rotor speeds
        w1, w2, w3, w4 = self.split_first(first_order_control)

        # Precompute square of speeds
        w1_2, w2_2, w3_2, w4_2 = w1**2, w2**2, w3**2, w4**2

        # Compute forces/torques
        u1 = self.b * (w1_2 + w2_2 + w3_2 + w4_2)
        u2 = self.b * self.l * (w4_2 - w2_2)
        u3 = self.b * self.l * (w3_2 - w1_2)
        u4 = self.d * (w1_2 - w2_2 + w3_2 - w4_2)

        # Intermediates
        cos, sin = np.cos, np.sin
        ux = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        uy = cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)

        omega_r = w1 - w2 + w3 - w4

        # Dynamics
        ddx = (u1 * ux - self.Ct[0] * dx) / self.m
        ddy = (u1 * uy - self.Ct[1] * dy) / self.m
        ddz = (u1 * cos(theta) * cos(phi) - self.Ct[2] * dz) / self.m - self.g

        Ix, Iy, Iz = self.I
        ddphi = (u2 - self.Cr[0] * dphi**2 - self.Jr * omega_r * dtheta - (Iz - Iy) * dtheta * dpsi) / Ix
        ddtheta = (u3 - self.Cr[1] * dtheta**2 + self.Jr * omega_r * dphi - (Ix - Iz) * dphi * dpsi) / Iy
        ddpsi = (u4 - self.Cr[2] * dpsi**2 - (Iy - Ix) * dphi * dtheta) / Iz

        return np.concatenate([
            dx,
            ddx,
            dy,
            ddy,
            dz,
            ddz,
            dphi,
            ddphi,
            dtheta,
            ddtheta,
            dpsi,
            ddpsi
        ], axis=1)

    def dfdx(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        dfdx = np.zeros((batch, 12, 12))

        x_pos, dx, y_pos, dy, z_pos, dz, phi, dphi, theta, dtheta, psi, dpsi = [
            x[:, i] for i in range(12)
        ]
        w1, w2, w3, w4 = [u[:, i] for i in range(4)]
        omega_r = w1 - w2 + w3 - w4

        m = self.m
        Ix, Iy, Iz = self.I
        Jr = self.Jr
        Cx, Cy, Cz = self.Ct
        Crx, Cry, Crz = self.Cr

        cos, sin = np.cos, np.sin

        ux = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        uy = cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)
        u1 = self.b * (w1**2 + w2**2 + w3**2 + w4**2)

        # dx/dx
        dfdx[:, 0, 1] = 1.0
        dfdx[:, 1, 1] = -Cx / m
        dfdx[:, 1, 6] = (u1 / m) * (-sin(phi)*cos(psi)*sin(theta) + sin(psi)*cos(phi))
        dfdx[:, 1, 8] = (u1 / m) * (cos(phi)*cos(psi)*cos(theta))
        dfdx[:, 1,10] = (u1 / m) * (-sin(psi)*sin(theta)*cos(phi) + cos(psi)*sin(phi))

        # dy/dy
        dfdx[:, 2, 3] = 1.0
        dfdx[:, 3, 3] = -Cy / m
        dfdx[:, 3, 6] = (u1 / m) * (-sin(psi)*cos(phi) - cos(psi)*sin(theta)*sin(phi))
        dfdx[:, 3, 8] = (u1 / m) * (cos(phi)*sin(psi)*cos(theta))
        dfdx[:, 3,10] = (u1 / m) * (cos(psi)*cos(phi)*sin(theta) + sin(psi)*sin(phi))

        # dz/dz
        dfdx[:, 4, 5] = 1.0
        dfdx[:, 5, 5] = -Cz / m
        dfdx[:, 5, 6] = -(u1 / m) * cos(theta) * sin(phi)
        dfdx[:, 5, 8] = -(u1 / m) * cos(phi) * sin(theta)

        # dφ/dt, dθ/dt, dψ/dt
        dfdx[:, 6, 7] = 1.0
        dfdx[:, 8, 9] = 1.0
        dfdx[:,10,11] = 1.0

        # Rotational dynamics
        dfdx[:, 7, 7] = -2 * Crx * dphi / Ix
        dfdx[:, 7, 9] = -Jr * omega_r / Ix - (Iz - Iy) * dpsi / Ix
        dfdx[:, 7,11] = -(Iz - Iy) * dtheta / Ix

        dfdx[:, 9, 7] = Jr * omega_r / Iy - (Ix - Iz) * dpsi / Iy
        dfdx[:, 9, 9] = -2 * Cry * dtheta / Iy
        dfdx[:, 9,11] = -(Ix - Iz) * dphi / Iy

        dfdx[:,11, 7] = -(Iy - Ix) * dtheta / Iz
        dfdx[:,11, 9] = -(Iy - Ix) * dphi / Iz
        dfdx[:,11,11] = -2 * Crz * dpsi / Iz

        return dfdx

    def dfdu(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        dfdu = np.zeros((batch, 12, 4))

        dx, dy, dz, phi, theta, psi, dphi, dtheta, dpsi = [
            x[:, i] for i in [1, 3, 5, 6, 8, 10, 7, 9, 11]
        ]
        w1, w2, w3, w4 = [u[:, i] for i in range(4)]

        m = self.m
        Ix, Iy, Iz = self.I
        b, d, l = self.b, self.d, self.l

        cos, sin = np.cos, np.sin
        ux = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        uy = cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)

        dfdu[:, 1, 0] = (2 * b * w1 * ux) / m
        dfdu[:, 1, 1] = (2 * b * w2 * ux) / m
        dfdu[:, 1, 2] = (2 * b * w3 * ux) / m
        dfdu[:, 1, 3] = (2 * b * w4 * ux) / m

        dfdu[:, 3, 0] = (2 * b * w1 * uy) / m
        dfdu[:, 3, 1] = (2 * b * w2 * uy) / m
        dfdu[:, 3, 2] = (2 * b * w3 * uy) / m
        dfdu[:, 3, 3] = (2 * b * w4 * uy) / m

        common_z = cos(theta) * cos(phi)
        dfdu[:, 5, 0] = (2 * b * w1 * common_z) / m
        dfdu[:, 5, 1] = (2 * b * w2 * common_z) / m
        dfdu[:, 5, 2] = (2 * b * w3 * common_z) / m
        dfdu[:, 5, 3] = (2 * b * w4 * common_z) / m

        # ∂φ̈
        dfdu[:, 7, 1] = -2 * b * l * w2 / Ix
        dfdu[:, 7, 3] =  2 * b * l * w4 / Ix
        dfdu[:, 7, 0] = -dtheta * self.Jr / Ix
        dfdu[:, 7, 1] = dfdu[:, 7, 1] + dtheta * self.Jr / Ix
        dfdu[:, 7, 2] = dfdu[:, 7, 2] - dtheta * self.Jr / Ix
        dfdu[:, 7, 3] = dfdu[:, 7, 3] + dtheta * self.Jr / Ix

        # ∂θ̈
        dfdu[:, 9, 2] = 2 * b * l * w3 / Iy
        dfdu[:, 9, 0] = -2 * b * l * w1 / Iy
        dfdu[:, 9, 0] += dphi * self.Jr / Iy
        dfdu[:, 9, 1] += -dphi * self.Jr / Iy
        dfdu[:, 9, 2] += dphi * self.Jr / Iy
        dfdu[:, 9, 3] += -dphi * self.Jr / Iy

        # ∂ψ̈
        dfdu[:,11,0] =  2 * d * w1 / Iz
        dfdu[:,11,1] = -2 * d * w2 / Iz
        dfdu[:,11,2] =  2 * d * w3 / Iz
        dfdu[:,11,3] = -2 * d * w4 / Iz

        return dfdu