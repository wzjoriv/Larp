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
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdx(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
    
    def dfdu(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        raise NotImplementedError
        
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
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        _, _, theta = self.split_first(first_order_state)

        v, w = self.split_first(first_order_control)
        
        dx = v*np.cos(theta)
        dy = v*np.sin(theta)
        dtheta = w

        return np.concatenate([dx, dy, dtheta], axis=1)
    
    def dfdu(self, time: np.ndarray, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:

        _, _, theta = self.split_first(first_order_state)
        v, w = self.split_first(first_order_control)

        df1 = np.concatenate([np.cos(theta), np.zeros_like(w)], axis=1)
        df2 = np.concatenate([np.sin(theta), np.zeros_like(w)], axis=1)
        df3 = np.concatenate([np.zeros_like(v), np.ones_like(w)], axis=1)

        return np.stack([df1, df2, df3], axis=1)
    
    def dfdx(self, time: np.ndarray, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:

        x, y, theta = self.split_first(first_order_state)
        v, _ = self.split_first(first_order_control)

        df1 = np.concatenate([np.zeros_like(x), np.zeros_like(y), -v*np.sin(theta)], axis=1)
        df2 = np.concatenate([np.zeros_like(x), np.zeros_like(y), v*np.cos(theta)], axis=1)
        df3 = np.concatenate([np.zeros_like(x), np.zeros_like(y), np.zeros_like(theta)], axis=1)

        return np.stack([df1, df2, df3], axis=1)
    
class QuadcopterDynamics(Dynamics):

    """
    Quadcopter

    - Abdelhay, S., & Zakriti, A. (2019). Modeling of a quadcopter trajectory tracking system using PID controller. Procedia Manufacturing, 32, 564-571.
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
        
        """
        Dynamics for a quadcopter
        
        Constants:
            Inertia (I; default: [3.8*10^-3, 3.8*10^-3, 7.1*10^-3] kg m^2)
            Mass (m; default: 5.2 kg)
            Gravity (g; default: 9.807 m^2/s)
            Arm length (l; 0.32 m)

            Thrust constant (k_t; default: 3.13*10^-5 kg m)
            Net translational drag/friction coefficients (C_t; default: [0.1, 0.1, 0.15] kg s^-1)

            Torque constant (k_r; default: 7.5*10^-7 kg m)
            Net rotational drag/friction coefficients (C_r; default: [0.1, 0.1, 0.15] kg m)

            Inertia of motor (J_r; default: 6*10^-5 kg m^2) (For gyroscopic effect)

        """

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
                         control_derivative_orders=[0]*4,)
        
        self.m = self.constants['mass']
        self.Ct = self.constants['translational drag']
        self.g = self.constants['gravity']
        self.I = self.constants['inertia']
        self.Cr = self.constants['rotational drag']
        self.Jr = self.constants['motor inertia']

        self.b = self.constants['thrust constant']
        self.d = self.constants['torque constant']
        self.l = self.constants['arm length']

        self.w2_to_control = np.array([
            [self.b, self.b, self.b, self.b],
            [0, -self.b * self.l, 0, self.b * self.l],
            [-self.b * self.l, 0, self.b * self.l, 0],
            [self.d, -self.d, self.d, -self.d]
        ])

        self.control_to_w2 = np.linalg.inv(self.w2_to_control)

    def extract_w(self, first_order_control:np.ndarray) -> np.ndarray:

        w_squared = first_order_control @ self.control_to_w2.T

        w_squared = np.clip(w_squared, a_min=0.0, a_max=None)
        return np.sqrt(w_squared)
    
    def f(self,
          time: np.ndarray, 
          first_order_state: np.ndarray,
          first_order_control: np.ndarray) -> np.ndarray:
        
        x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi = tuple([first_order_state[:, i:i+1] for i in range(first_order_state.shape[1])])

        u1, u2, u3, u4 = tuple([first_order_control[:, i:i+1] for i in range(first_order_control.shape[1])])

        ux = np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)

        uy = np.cos(phi)*np.sin(psi)*np.sin(theta) - np.cos(psi)*np.sin(phi)

        ws = self.extract_w(first_order_control)
        w1, w2, w3, w4 = tuple([ws[:, i:i+1] for i in range(ws.shape[1])])

        omega_r = w1 - w2 + w3 - w4

        return np.concatenate([
            dx,
            (u1/self.m)*ux - (self.Ct[0]/self.m)*dx,
            dy,
            (u1/self.m)*uy - (self.Ct[1]/self.m)*dy,
            dz,
            (u1/self.m)*np.cos(theta)*np.cos(phi) - (self.Ct[2]/self.m)*dz - self.g,
            dphi,
            (u2 - self.Cr[0]*dphi**2 - self.Jr*omega_r*dtheta - (self.I[2] - self.I[1])*dtheta*dpsi)/self.I[0],
            dtheta,
            (u3 - self.Cr[1]*dtheta**2 + self.Jr*omega_r*dphi - (self.I[0] - self.I[2])*dphi*dpsi)/self.I[1],
            dpsi,
            (u4 - self.Cr[2]*dpsi**2 - (self.I[1] - self.I[0])*dphi*dtheta)/self.I[2]
        ], axis=1)
    
