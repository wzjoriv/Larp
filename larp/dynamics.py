from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from itertools import chain

"""
Author: Josue N Rivera
"""
        
class Dynamics():

    """
    Base class for representing general dynamical systems with support for
    higher-order derivatives of states and controls.

    This class assumes the system is represented in terms of its *first-order form*:

    .. math::

        \\dot{x} = f(x, u)

    where:

    - :math:`x \\in \\mathbb{R}^n` is the **state vector**
    - :math:`u \\in \\mathbb{R}^m` is the **control vector**

    For higher-order systems (e.g., involving acceleration, jerk), this class uses the notion
    of *primitive states* and builds up the first-order representation accordingly.

    Attributes
    ----------
    constants : dict, optional
        Dictionary of any physical or system constants needed for the dynamics.

    state_derivative_orders : list of int
        List of maximum derivative orders for each primitive state variable.

    control_derivative_orders : list of int
        List of maximum derivative orders for each primitive control variable.

    first_state_orders : np.ndarray
        The derivative order of each element in the first-order state vector.

    first_control_orders : np.ndarray
        The derivative order of each element in the first-order control vector.

    highest_state_order : int
        The maximum order of state derivatives.

    highest_control_order : int
        The maximum order of control derivatives.

    state_primitive_mask : np.ndarray of bool
        Mask indicating positions of primitive state variables.

    control_primitive_mask : np.ndarray of bool
        Mask indicating positions of primitive control variables.

    primitive_state_n : int
        Number of primitive state variables.

    primitive_control_n : int
        Number of primitive control variables.

    first_order_state_n : int
        Total length of first-order state vector.

    first_order_control_n : int
        Total length of first-order control vector.
    """

    def __init__(self, 
                 constants:Optional[Dict] = None,
                 state_derivative_orders:List[int] = [1],
                 control_derivative_orders:List[int] = [0]) -> None:
        
        """
        Initialize the base dynamics model.

        Parameters
        ----------
        constants : dict, optional
            Physical parameters or constants used in dynamics computations.

        state_derivative_orders : list of int
            Derivative orders for each primitive state (e.g., `[2]` → position, velocity, and acceleration are needed to represent f).

        control_derivative_orders : list of int
            Derivative orders for each primitive control. Most time it would be 0.
        """

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
    
    def split_first(self, first: np.ndarray) -> Tuple[np.ndarray]:
        """
        Splits the concatenated first-order vector into primitive segments.

        Parameters
        ----------
        first : np.ndarray
            Batched vector of first-order states or controls, shape (batch_size, dim)

        Returns
        -------
        tuple of np.ndarray
            Tuple of shape (batch_size, 1) per primitive component.
        """
        return tuple([first[:, i:i+1] for i in range(first.shape[1])])

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Defines the time derivative of the state: \\( \\dot{x} = f(x, u) \\)

        Parameters
        ----------
        first_order_state : np.ndarray
            First-order state vector, shape (batch_size, state_dim)

        first_order_control : np.ndarray
            First-order control vector, shape (batch_size, control_dim)

        Returns
        -------
        np.ndarray
            Time derivative of the state.
        """
        raise NotImplementedError

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Jacobian of \\( f(x, u) \\) with respect to \\( x \\)

        Returns
        -------
        np.ndarray
            Partial derivatives \\( \\frac{\\partial f}{\\partial x} \\)
        """
        raise NotImplementedError

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Jacobian of \\( f(x, u) \\) with respect to \\( u \\)

        Returns
        -------
        np.ndarray
            Partial derivatives \\( \\frac{\\partial f}{\\partial u} \\)
        """
        raise NotImplementedError

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearizes the nonlinear dynamics \\( f(x, u) \\) about the reference point \\( (x_0, u_0) \\).

        Parameters
        ----------
        x0 : np.ndarray
            Batch of reference states, shape (batch_size, state_dim)

        u0 : np.ndarray
            Batch of reference controls, shape (batch_size, control_dim)

        Returns
        -------
        A : np.ndarray
            Jacobian matrix \\( A = \\frac{\\partial f}{\\partial x} \\)

        B : np.ndarray
            Jacobian matrix \\( B = \\frac{\\partial f}{\\partial u} \\)
        """
        A = self.dfdx(x0, u0)
        B = self.dfdu(x0, u0)
        return A, B

    def discretize(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretizes the linearized system at point (x0, u0) using zero-order hold (ZOH):

        .. math::

            \\begin{bmatrix} A_d & B_d \\\\ 0 & I \\end{bmatrix} =
            \\exp \\left( dt \\cdot \\begin{bmatrix} A & B \\\\ 0 & 0 \\end{bmatrix} \\right)

        Parameters
        ----------
        x0 : np.ndarray
            Batched state input, shape (batch_size, state_dim)

        u0 : np.ndarray
            Batched control input, shape (batch_size, control_dim)

        dt : float
            Time step for discretization

        estimate : bool
            Whether to estimate discretization

        Returns
        -------
        Ad : np.ndarray
            Discrete-time state matrix

        Bd : np.ndarray
            Discrete-time control matrix
        """
        state_dim = self.first_order_state_n

        A, B = self.linearize(x0, u0)

        if estimate:
            Ad = np.eye(state_dim) + A*dt
            Bd = B*dt

        else:
            batch_size = x0.shape[0]
            control_dim = self.first_order_control_n

            M = np.zeros((batch_size, state_dim + control_dim, state_dim + control_dim))
            M[:, :state_dim, :state_dim] = A
            M[:, :state_dim, state_dim:] = B

            AdBd = expm(dt * M)[:, :state_dim, :]
            Ad, Bd = AdBd[:, :, :state_dim], AdBd[:, :, state_dim:]

        return Ad, Bd

    def first_state_names(self) -> List[str]:
        """
        Returns symbolic names for each component of the first-order state vector.

        Returns
        -------
        List[str]
            Labels such as `x_0^{[0]}`, `x_0^{[1]}`, etc.
        """
        orders = self.state_derivative_orders
        return [f'x_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    def first_control_names(self) -> List[str]:
        """
        Returns symbolic names for each component of the first-order control vector.

        Returns
        -------
        List[str]
            Labels such as `u_0^{[0]}`, `u_0^{[1]}`, etc.
        """
        orders = self.control_derivative_orders
        return [f'u_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    def first_names(self) -> Tuple[List[str], List[str]]:
        """
        Returns symbolic names for both state and control vectors.

        Returns
        -------
        Tuple[List[str], List[str]]
            Tuple of lists (state_names, control_names)
        """
        return self.first_state_names(), self.first_control_names()
    
class WMRDynamics(Dynamics):
    """
    Dynamics for a differential-drive Wheeled Mobile Robot (WMR).

    This class implements the unicycle model commonly used for differential-drive robots:

    .. math::

        \\begin{align}
        \\dot{x} &= v \\cos(\\theta) \\\\
        \\dot{y} &= v \\sin(\\theta) \\\\
        \\dot{\\theta} &= \\omega
        \\end{align}

    where:

    - :math:`(x, y)` is the position of the robot in 2D space
    - :math:`\\theta` is the robot's orientation
    - :math:`v` is the linear velocity
    - :math:`\\omega` is the angular velocity

    References
    ----------
    Kuhne, F., Lages, W. F., & da Silva Jr, J. G. (2004, September).
    Model predictive control of a mobile robot using linearization.
    In *Proceedings of Mechatronics and Robotics* (Vol. 4, No. 4, pp. 525-530).

    Parameters
    ----------
    wheels_distance : float, optional
        The distance between the left and right wheels (default: 1.0)

    Attributes
    ----------
    wd : float
        The wheelbase, used to compute left/right wheel speeds.
    """

    def __init__(self, wheels_distance=1.0) -> None:
        """
        Initializes the Wheeled Mobile Robot (WMR) dynamics with physical parameters.

        Parameters
        ----------
        wheels_distance : float
            Distance between the wheels.
        """

        constants = {
            'wheels distance': wheels_distance
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[0, 0, 0],   # x, y, theta
                         control_derivative_orders=[0, 0])    # v, omega

        self.wd = self.constants['wheels distance']

    def extract_wheel_speed(self, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the left and right wheel speeds from the control input.

        Parameters
        ----------
        first_order_control : np.ndarray
            Control input of shape (batch_size, 2), where each row is (v, ω)

        Returns
        -------
        np.ndarray
            Concatenated array of left and right wheel speeds,
            shape (2 * batch_size, 1)
        """
        v, w = self.split_first(first_order_control)
        wd2 = self.wd/2

        v_l = v - wd2 * w
        v_r = v + wd2 * w

        return np.concatenate([v_l, v_r], axis=0)

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative \\( \\dot{x} = f(x, u) \\) for the WMR model.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector of shape (batch_size, 3): [x, y, θ]

        first_order_control : np.ndarray
            Control vector of shape (batch_size, 2): [v, ω]

        Returns
        -------
        np.ndarray
            Derivative of the state, shape (batch_size, 3)
        """
        _, _, theta = self.split_first(first_order_state)
        v, w = self.split_first(first_order_control)

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w

        return np.concatenate([dx, dy, dtheta], axis=1)

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial u} \\) of the WMR dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape (batch_size, 3)

        first_order_control : np.ndarray
            Control vector, shape (batch_size, 2)

        Returns
        -------
        np.ndarray
            Jacobian matrix, shape (batch_size, 3, 2)
        """
        _, _, theta = self.split_first(first_order_state)
        v, w = self.split_first(first_order_control)

        zeros = np.zeros_like(v)

        df1 = np.concatenate([np.cos(theta), zeros], axis=1)
        df2 = np.concatenate([np.sin(theta), zeros], axis=1)
        df3 = np.concatenate([zeros, np.ones_like(w)], axis=1)

        return np.stack([df1, df2, df3], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial x} \\) of the WMR dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape (batch_size, 3)

        first_order_control : np.ndarray
            Control vector, shape (batch_size, 2)

        Returns
        -------
        np.ndarray
            Jacobian matrix, shape (batch_size, 3, 3)
        """
        x, _, theta = self.split_first(first_order_state)
        v, _ = self.split_first(first_order_control)

        zeros = np.zeros_like(x)

        df1 = np.concatenate([zeros, zeros, -v * np.sin(theta)], axis=1)
        df2 = np.concatenate([zeros, zeros, v * np.cos(theta)], axis=1)
        df3 = np.concatenate([zeros, zeros, zeros], axis=1)

        return np.stack([df1, df2, df3], axis=1)

class Quadcopter2DDynamics(Dynamics):
    """
    Dynamics for a planar (2D) quadcopter system.

    The model assumes a rigid body with two vertically-oriented thrust-producing rotors.
    The dynamics are derived from Newton-Euler equations in 2D space, with the state ordered as:

    .. math::

        x = [x, \\dot{x}, y, \\dot{y}, \\theta, \\dot{\\theta}]^\\top

    The continuous-time dynamics are given by:

    .. math::

        \\begin{align}
        \\dot{x} &= \\dot{x} \\\\
        \\dot{\\dot{x}} &= -\\frac{u_1 + u_2}{m} \\sin(\\theta) \\\\
        \\dot{y} &= \\dot{y} \\\\
        \\dot{\\dot{y}} &= \\frac{u_1 + u_2}{m} \\cos(\\theta) - g \\\\
        \\dot{\\theta} &= \\dot{\\theta} \\\\
        \\dot{\\dot{\\theta}} &= \\frac{L(u_1 - u_2)}{I}
        \\end{align}

    Parameters
    ----------
    mass : float, optional
        Mass of the quadcopter in kilograms (default: 1.0)

    gravity : float, optional
        Gravitational acceleration in m/s² (default: 9.81)

    arm_length : float, optional
        Distance from the center of mass to each rotor in meters (default: 0.5)

    inertia : float, optional
        Moment of inertia around the out-of-plane axis (default: 0.01)

    Attributes
    ----------
    m : float
        Mass of the quadcopter

    g : float
        Gravitational acceleration

    L : float
        Arm length (distance from center to rotor)

    I : float
        Moment of inertia
    """

    def __init__(self, mass=1.0, gravity=9.81, arm_length=0.5, inertia=0.01) -> None:
        """
        Initializes the 2D quadcopter dynamics with physical parameters.

        Parameters
        ----------
        mass : float
            Mass of the quadcopter.

        gravity : float
            Gravitational acceleration.

        arm_length : float
            Distance from the center to each rotor.

        inertia : float
            Moment of inertia about the out-of-plane axis.
        """
        constants = {
            'mass': mass,
            'gravity': gravity,
            'arm_length': arm_length,
            'inertia': inertia,
        }

        super().__init__(
            constants=constants,
            state_derivative_orders=[1, 1, 1],
            control_derivative_orders=[0, 0],
        )

        self.m = mass
        self.g = gravity
        self.L = arm_length
        self.I = inertia

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative \\( \\dot{x} = f(x, u) \\) for the 2D quadcopter.

        Parameters
        ----------
        first_order_state : np.ndarray
            State array of shape (batch_size, 6): [x, dx, y, dy, θ, dθ]

        first_order_control : np.ndarray
            Control input of shape (batch_size, 2): [u₁, u₂]

        Returns
        -------
        np.ndarray
            Time derivative of the state, shape (batch_size, 6)
        """
        x, dx, y, dy, theta, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        total_u = u1 + u2

        ddx = -total_u * np.sin(theta) / self.m
        ddy = total_u * np.cos(theta) / self.m - self.g
        ddtheta = self.L * (u1 - u2) / self.I

        return np.concatenate([dx, ddx, dy, ddy, dtheta, ddtheta], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial x} \\) of the dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State array of shape (batch_size, 6): [x, dx, y, dy, θ, dθ]

        first_order_control : np.ndarray
            Control input array of shape (batch_size, 2): [u₁, u₂]

        Returns
        -------
        np.ndarray
            Jacobian tensor of shape (batch_size, 6, 6)
        """
        x, dx, y, dy, theta, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        zero = np.zeros_like(x)
        one = np.ones_like(x)

        total_u = u1 + u2
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        dddx_dtheta = -total_u * cos_theta / self.m
        dddy_dtheta = -total_u * sin_theta / self.m

        df = [
            np.concatenate([zero, one, zero, zero, zero, zero], axis=1),             # ∂ẋ/∂x
            np.concatenate([zero, zero, zero, zero, dddx_dtheta, zero], axis=1),     # ∂ẍ/∂x
            np.concatenate([zero, zero, zero, one, zero, zero], axis=1),             # ∂ẏ/∂x
            np.concatenate([zero, zero, zero, zero, dddy_dtheta, zero], axis=1),     # ∂ÿ/∂x
            np.concatenate([zero, zero, zero, zero, zero, one], axis=1),             # ∂θ̇/∂x
            np.concatenate([zero, zero, zero, zero, zero, zero], axis=1),            # ∂θ̈/∂x
        ]
        return np.stack(df, axis=1)

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial u} \\) of the dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State array of shape (batch_size, 6): [x, dx, y, dy, θ, dθ]

        first_order_control : np.ndarray
            Control input array of shape (batch_size, 2): [u₁, u₂]

        Returns
        -------
        np.ndarray
            Jacobian tensor of shape (batch_size, 6, 2)
        """
        x, dx, y, dy, theta, dtheta = self.split_first(first_order_state)
        u1, u2 = self.split_first(first_order_control)

        zero = np.zeros_like(u1)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        dddx_du = -sin_theta / self.m
        dddy_du = cos_theta / self.m
        ddtheta_du1 = self.L / self.I
        ddtheta_du2 = -self.L / self.I

        df = [
            np.concatenate([zero, zero], axis=1),  # ∂ẋ/∂u
            np.concatenate([dddx_du, dddx_du], axis=1),  # ∂ẍ/∂u
            np.concatenate([zero, zero], axis=1),  # ∂ẏ/∂u
            np.concatenate([dddy_du, dddy_du], axis=1),  # ∂ÿ/∂u
            np.concatenate([zero, zero], axis=1),  # ∂θ̇/∂u
            np.concatenate([
                np.full_like(u1, ddtheta_du1),
                np.full_like(u2, ddtheta_du2)
            ], axis=1),  # ∂θ̈/∂u
        ]
        return np.stack(df, axis=1)

class QuadcopterV1Dynamics(Dynamics):
    """
    Quadcopter Dynamics V1 model.

    This model describes a rigid-body quadcopter with translational and rotational dynamics.
    It uses the Newton-Euler equations for a 6-DOF system.

    **State vector (12 elements)**:
        - Positions: x, y, z
        - Linear velocities: vx, vy, vz
        - Euler angles: phi (roll), theta (pitch), psi (yaw)
        - Angular velocities: p, q, r

    **Control vector (4 elements)**:
        - u1: total thrust (N)
        - u2: roll torque (Nm)
        - u3: pitch torque (Nm)
        - u4: yaw torque (Nm)

    **Constants**:
        - m: Mass of the quadcopter (kg)
        - g: Gravitational acceleration (m/s²)
        - Ix, Iy, Iz: Moments of inertia along x, y, z axes (kg·m²)

    The translational dynamics are governed by:

    .. math::
        \\dot{v} = \\frac{1}{m} R \\begin{bmatrix} 0 \\\\ 0 \\\\ u_1 \\end{bmatrix} - \\begin{bmatrix} 0 \\\\ 0 \\\\ g \\end{bmatrix}

    where \\( R \\) is the rotation matrix from body to world frame using Euler angles (phi, theta, psi).

    The rotational dynamics are governed by:

    .. math::
        \\dot{\\omega} = I^{-1} (\\tau - \\omega \\times I \\omega)

    where \\( \\omega = [p, q, r]^T \\), \\( I = diag(Ix, Iy, Iz) \\), and \\( \\tau = [u_2, u_3, u_4]^T \\).

    Parameters
    ----------
    mass : float, optional
        Mass of the quadcopter [kg] (default: 1.0)

    gravity : float, optional
        Gravitational acceleration [m/s²] (default: 9.81)

    Ix : float, optional
        Moment of inertia about x-axis [kg·m²] (default: 0.01)

    Iy : float, optional
        Moment of inertia about y-axis [kg·m²] (default: 0.01)

    Iz : float, optional
        Moment of inertia about z-axis [kg·m²] (default: 0.02)

    Attributes
    ----------
    m : float
        Mass of the quadcopter

    g : float
        Gravitational acceleration

    Ix : float
        Inertia around x-axis

    Iy : float
        Inertia around y-axis

    Iz : float
        Inertia around z-axis
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
                         state_derivative_orders=[1] * 3 + [0]*6,   # states
                         control_derivative_orders=[0] * 4)  # 4 control inputs

        self.m = mass
        self.g = gravity
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative \\( \\dot{x} = f(x, u) \\) of the quadcopter state.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector of shape (batch_size, 12)

        first_order_control : np.ndarray
            Control vector of shape (batch_size, 4)

        Returns
        -------
        np.ndarray
            Derivative of the state, shape (batch_size, 12)
        """
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = self.split_first(first_order_state)
        u1, u2, u3, u4 = self.split_first(first_order_control)

        m, g, Ix, Iy, Iz = self.m, self.g, self.Ix, self.Iy, self.Iz

        dx = vx
        dy = vy
        dz = vz

        sin, cos = np.sin, np.cos

        dvx = u1 / m * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi))
        dvy = u1 / m * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi))
        dvz = u1 / m * (cos(phi) * cos(theta)) - g

        dphi = p + q * sin(phi) * np.tan(theta) + r * cos(phi) * np.tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        dp = (u2 + (Iy - Iz) * q * r) / Ix
        dq = (u3 + (Iz - Ix) * p * r) / Iy
        dr = (u4 + (Ix - Iy) * p * q) / Iz

        return np.concatenate([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial x} \\) of the dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape (batch_size, 12)

        first_order_control : np.ndarray
            Control vector, shape (batch_size, 4)

        Returns
        -------
        np.ndarray
            Jacobian matrix, shape (batch_size, 12, 12)
        """
        _, _, _, _, _, _, phi, theta, psi, p, q, r = self.split_first(first_order_state)
        u1, _, _, _ = self.split_first(first_order_control)

        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz
        sin, cos = np.sin, np.cos
        tan = np.tan

        dfdx = np.zeros((first_order_state.shape[0], 12, 12))

        # Linear velocity to position
        dfdx[:, 0, 3] = 1
        dfdx[:, 1, 4] = 1
        dfdx[:, 2, 5] = 1

        # Acceleration w.r.t. angles
        dfdx[:, 3, 6] = -u1[:, 0] / m * (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi))
        dfdx[:, 3, 7] =  u1[:, 0] / m * (cos(phi) * cos(theta) * cos(psi))
        dfdx[:, 3, 8] = -u1[:, 0] / m * (cos(phi) * sin(theta) * sin(psi) + sin(phi) * cos(psi))

        dfdx[:, 4, 6] = -u1[:, 0] / m * (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi))
        dfdx[:, 4, 7] =  u1[:, 0] / m * (cos(phi) * cos(theta) * sin(psi))
        dfdx[:, 4, 8] =  u1[:, 0] / m * (cos(phi) * sin(theta) * cos(psi) - sin(phi) * sin(psi))

        dfdx[:, 5, 6] = -u1[:, 0] / m * sin(phi) * cos(theta)
        dfdx[:, 5, 7] = -u1[:, 0] / m * cos(phi) * sin(theta)

        # Euler angle kinematics
        dfdx[:, 6, 9]  = 1
        dfdx[:, 6, 10] = sin(phi) * tan(theta)
        dfdx[:, 6, 11] = cos(phi) * tan(theta)

        dfdx[:, 7, 10] = cos(phi)
        dfdx[:, 7, 11] = -sin(phi)

        dfdx[:, 8, 10] = sin(phi) / cos(theta)
        dfdx[:, 8, 11] = cos(phi) / cos(theta)

        # Angular acceleration terms
        dfdx[:, 9, 10] = (Iy - Iz) * r / Ix
        dfdx[:, 9, 11] = (Iy - Iz) * q / Ix
        dfdx[:, 10, 9] = (Iz - Ix) * r / Iy
        dfdx[:, 10, 11] = (Iz - Ix) * p / Iy
        dfdx[:, 11, 9] = (Ix - Iy) * q / Iz
        dfdx[:, 11, 10] = (Ix - Iy) * p / Iz

        return dfdx

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial u} \\) of the dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape (batch_size, 12)

        first_order_control : np.ndarray
            Control vector, shape (batch_size, 4)

        Returns
        -------
        np.ndarray
            Jacobian matrix, shape (batch_size, 12, 4)
        """
        phi, theta, psi = self.split_first(first_order_state)[6:9]
        u1, u2, u3, u4 = self.split_first(first_order_control)

        m, Ix, Iy, Iz = self.m, self.Ix, self.Iy, self.Iz
        sin, cos = np.sin, np.cos

        dfdu = np.zeros((first_order_state.shape[0], 12, 4))

        # ∂acceleration/∂u1
        dfdu[:, 3, 0] = 1 / m * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi))[:, 0]
        dfdu[:, 4, 0] = 1 / m * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi))[:, 0]
        dfdu[:, 5, 0] = 1 / m * (cos(phi) * cos(theta))[:, 0]

        # ∂angular_acceleration/∂u2, u3, u4
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

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        dfdx = np.zeros((first_order_state.shape[0], 12, 12))

        _, _, _, _, _, _, phi, dphi, theta, dtheta, psi, dpsi = self.split_first(first_order_state)
        w1, w2, w3, w4 = self.split_first(first_order_control)
        omega_r = w1 - w2 + w3 - w4

        m = self.m
        Ix, Iy, Iz = self.I
        Jr = self.Jr
        Cx, Cy, Cz = self.Ct
        Crx, Cry, Crz = self.Cr

        cos, sin = np.cos, np.sin

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

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        dfdu = np.zeros((first_order_state.shape[0], 12, 4))

        _, _, _, _, _, _, phi, dphi, theta, dtheta, psi, dpsi = self.split_first(first_order_state)
    
        w1, w2, w3, w4 = self.split_first(first_order_control)

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