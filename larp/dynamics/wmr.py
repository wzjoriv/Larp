import importlib
from types import ModuleType
from typing import List, Optional, Tuple
import numpy as np

from larp.const import JAX_INSTALLED, MJX_INSTALLED


class WMRDynamics(Dynamics):
    """
    Dynamics for a differential-drive Wheeled Mobile Robot (WMR).
    Includes physical dimensions for visualization.

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

    Attributes
    ----------
    wd : float
        The wheelbase, used to compute left/right wheel speeds.
    body_radius : float
        Radius of the robot's circular body (visualization).
    wheel_len : float
        Length of the wheels (visualization).
    wheel_width : float
        Width of the wheels (visualization).
    """

    def __init__(self, 
                 wheels_distance: float = 0.6, 
                 body_radius: float = 0.3,
                 wheel_dims: Tuple[float, float] = (0.2, 0.05), # (Length, Width)
                 jax_backend: bool = False) -> None:
        """
        Initializes the Wheeled Mobile Robot (WMR) dynamics with physical parameters.

        Parameters
        ----------
        wheels_distance : float
            Distance between the wheels.
        body_radius : float
            Radius of the robot body for visualization.
        wheel_dims : tuple
            Dimensions of the wheels (length, width) for visualization.
        jax_backend : bool
            Enable JAX backend for faster computation.
        """

        constants = {
            'wheels_distance': wheels_distance,
            'body_radius': body_radius,
            'wheel_length': wheel_dims[0],
            'wheel_width': wheel_dims[1]
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[0, 0, 0],   # x, y, theta
                         control_derivative_orders=[0, 0],    # v, omega
                         holonomic=False,
                         jax_backend=jax_backend)

        # Physics
        self.wd = self.constants['wheels_distance']
        
        self.body_radius = self.constants['body_radius']
        self.wheel_len = self.constants['wheel_length']
        self.wheel_width = self.constants['wheel_width']

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
            shape (batch_size, 2)
        """
        v, w = self.split_first(first_order_control)
        wd2 = self.wd/2
        v_l = v - wd2 * w
        v_r = v + wd2 * w
        return np.concatenate([v_l, v_r], axis=1)
    
    @property
    def angle_indices(self) -> List[int]:
        return [2]

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        """
        Computes the time derivative \\( \\dot{x} = f(x, u) \\) for the WMR model.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector of shape (batch_size, 3): [x, y, θ]

        first_order_control : np.ndarray
            Control vector of shape (batch_size, 2): [v, ω]

        np: Module
            Numpy backend. Numpy or Jax.numpy

        Returns
        -------
        np.ndarray
            Derivative of the state, shape (batch_size, 3)
        """
        if np is None: np = importlib.import_module("numpy")

        _, _, theta = self.split_first(first_order_state)
        v, w = self.split_first(first_order_control)

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = w

        return np.concatenate([dx, dy, dtheta], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian :math:`\left(\frac{\partial f}{\partial x}\right)` of the WMR dynamics.

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

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian :math:`\left(\frac{\partial f}{\partial u}\right)` of the WMR dynamics.

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

    def dfdxx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Analytical second-order state Hessian of the WMR dynamics.

        :math:`f_{xx} = \frac{\partial^2 f}{\partial x^2}`

        For the unicycle model :math:`f = [v\cos\theta,\ v\sin\theta,\ \omega]`, only
        the :math:`\partial^2 / \partial\theta^2` terms are non-zero:

        .. math::
            f_{xx}^{(0)}[2,2] = -v\cos\theta, \quad f_{xx}^{(1)}[2,2] = -v\sin\theta

        Parameters
        ----------
        first_order_state : (B, 3)
        first_order_control : (B, 2)

        Returns
        -------
        (B, 3, 3, 3)  — tensor indexed as [batch, output, x_i, x_j]
        """
        _, _, theta = self.split_first(first_order_state)
        v, _        = self.split_first(first_order_control)
        B           = first_order_state.shape[0]

        H = np.zeros((B, self.first_order_state_n, self.first_order_state_n, self.first_order_state_n))
        H[:, 0, 2, 2] = (-v * np.cos(theta)).reshape(-1)
        H[:, 1, 2, 2] = (-v * np.sin(theta)).reshape(-1)
        return H

    def dfduu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Analytical second-order control Hessian of the WMR dynamics.

        :math:`f_{uu} = \frac{\partial^2 f}{\partial u^2}`

        The unicycle dynamics are **linear** in :math:`u = [v, \omega]`, so all
        second-order control derivatives are identically zero.

        Returns
        -------
        (B, 3, 2, 2)  — all zeros
        """
        B = first_order_state.shape[0]
        return np.zeros((B, self.first_order_state_n, self.first_order_control_n, self.first_order_control_n))

    def dfdux(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Analytical mixed control-state Hessian of the WMR dynamics.

        :math:`f_{ux} = \frac{\partial^2 f}{\partial u \partial x}`

        Only the cross terms between :math:`v` and :math:`\theta` are non-zero:

        .. math::
            f_{ux}^{(0)}[v, \theta] = -\sin\theta, \quad
            f_{ux}^{(1)}[v, \theta] =  \cos\theta

        Parameters
        ----------
        first_order_state : (B, 3)
        first_order_control : (B, 2)

        Returns
        -------
        (B, 3, 2, 3)  — tensor indexed as [batch, output, u_i, x_j]
        """
        _, _, theta = self.split_first(first_order_state)
        B           = first_order_state.shape[0]

        H = np.zeros((B, self.first_order_state_n, self.first_order_control_n, self.first_order_state_n))
        H[:, 0, 0, 2] = (-np.sin(theta)).reshape(-1)
        H[:, 1, 0, 2] = ( np.cos(theta)).reshape(-1)
        return H