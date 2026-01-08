from abc import ABC, abstractmethod
import importlib.util
import inspect
from types import ModuleType
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from itertools import chain

from larp.fn import bmatvec

JAX_INSTALLED = importlib.util.find_spec("jax") is not None

if JAX_INSTALLED:
    import jax.numpy as jnp
    from jax import jacfwd, jit, vmap
    from jax.scipy.linalg import expm as jexpm

"""
Author: Josue N Rivera

Dynamics classes
"""
        
class Dynamics(ABC):

    r"""
    Base class for representing general dynamical systems with support for
    higher-order derivatives, batched operations, and automatic differentiation via JAX.

    This class assumes the system can be represented in a **first-order form**:

    .. math::
        \dot{x} = f(x, u)

    where :math:`x \in \mathbb{R}^{n}` is the flattened state vector and 
    :math:`u \in \mathbb{R}^{m}` is the flattened control vector.

    **Key Concepts:**
    
    1.  **Primitives vs. First-Order:** The class maps "primitive" configuration variables 
        (e.g., position $p$) to a flattened first-order vector containing derivatives 
        (e.g., $[p, \dot{p}, \ddot{p}]$).
    2.  **Batched Operations:** All methods expect inputs with a leading batch dimension 
        (e.g., shape ``(batch_size, state_dim)``).
    3.  **Auto-Differentiation:** If JAX is installed and the subclass does not implement 
        analytical ``dfdx``/``dfdu``, this class attempts to generate them automatically.

    Attributes
    ----------
    state_dim : int
        Total dimension of the flattened first-order state vector.
    control_dim : int
        Total dimension of the flattened first-order control vector.
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
        
    first_angle_state_mask : np.ndarray
        Boolean mask of shape ``(state_dim,)``. True at indices corresponding to
        states that require angular wrapping.

    Examples
    --------
    .. code-block:: python
        # 1 primitive state (pos) with order 2 (pos, vel, acc)
        # 1 primitive control (jerk) with order 0
        >>> dyn = Dynamics(state_derivative_orders=[2], control_derivative_orders=[0])
        
        # State vector x corresponds to [pos, vel, acc]
        # Control vector u corresponds to [jerk]
        >>> dyn.first_order_state_n
        3
    """

    def __init__(self,
                 constants: Optional[Dict] = None, 
                 state_derivative_orders: List[int] = [1],
                 control_derivative_orders: List[int] = [0],
                 wrapable_primitive_state: Optional[List[int]] = None,
                 holonomic: Optional[bool] = None,
                 jax_backend=False) -> None:
        
        r"""
        Initializes the Dynamics instance.

        Parameters
        ----------
        constants : dict, optional
            Physical parameters or constants used in the model (e.g., mass, length).
            Defaults to an empty dict.
        state_derivative_orders : list of int, default=[1]
            Highest derivative order for each primitive state dynamics function.
            
            * Example: ``[2, 1]`` implies:
                * Primitive 0: :math:`[x_0, \dot{x}_0, \ddot{x}_0]`
                * Primitive 1: :math:`[x_1, \dot{x}_1]`
        control_derivative_orders : list of int, default=[0]
            Same structure as ``state_derivative_orders`` but for controls.
        wrapable_primitive_state : list of int, optional
            Indices of primitive states that represent angles (require wrapping to $[-\pi, \pi]$).
        holonomic : bool, optional
            If True, indicates the system is holonomic; otherwise non-holonomic.
        """

        self.constants = {} if constants is None else constants 
        self.holonomic = holonomic
        
        # --- Dimensions and Orders ---
        self.state_derivative_orders = np.array(state_derivative_orders, dtype=int)
        self.control_derivative_orders = np.array(control_derivative_orders, dtype=int)
        
        self.primitive_state_n = len(state_derivative_orders)
        self.primitive_control_n = len(control_derivative_orders)

        self.state_block_sizes = self.state_derivative_orders + 1
        self.control_block_sizes = self.control_derivative_orders + 1

        self.first_order_state_n = int(np.sum(self.state_block_sizes))
        self.first_order_control_n = int(np.sum(self.control_block_sizes))
        
        # --- Masks and Indices ---
        # Privitive state devitive order for each system state
        self.first_state_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.state_derivative_orders])
        self.first_control_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.control_derivative_orders])
        
        # One order higher of the max derivitive order for each primitive state needed to form the ode
        self.highest_state_order = max(self.state_derivative_orders)
        self.highest_control_order = max(self.control_derivative_orders)
        
        self.state_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in state_derivative_orders])), dtype=bool).reshape(-1)
        self.control_primitive_mask = np.array(list(chain(*[[True]+[False]*i for i in control_derivative_orders])), dtype=bool).reshape(-1)

        self.primitive_to_first_map = np.cumsum(
            np.concatenate(([0], self.state_derivative_orders[:-1] + 1))
        )
        
        self.first_wrapped_state_mask = np.zeros(self.first_order_state_n, dtype=bool)
        if wrapable_primitive_state:
            self.first_wrapped_state_mask[self.primitive_to_first_map[wrapable_primitive_state]] = True

        # --- Auto-JAX Jacobians ---
        self.dfdx_jax = None
        self.dfdu_jax = None
        self._linearize_jit = None
        self._discretize_jit = None

        self.jax_backend = jax_backend and "np" in inspect.signature(self.f).parameters and JAX_INSTALLED

        if self.jax_backend:
            self._setup_jax_functions()

    def _setup_jax_functions(self):
        """
        Compiles JAX functions for derivatives, linearization, and discretization.
        This enables end-to-end vectorization on the GPU/TPU.
        """
        # 1. Setup Auto-Diff for dfdx/dfdu
        # Wrapper to ensure f returns unbatched output for jacfwd
        def f_wrapper(x, u):
            return self.f(x[None, :], u[None, :], np=jnp)[0]

        # Auto-differentiation
        self.dfdx_jax = jit(vmap(jacfwd(f_wrapper, argnums=0)))
        self.dfdu_jax = jit(vmap(jacfwd(f_wrapper, argnums=1)))

        # 2. Setup Linearization JIT
        def _linearize_pure(x, u):
            """Pure JAX implementation of linearization"""
            f0 = self.f(x, u, np=jnp)
            A = self.dfdx_jax(x, u)
            B = self.dfdu_jax(x, u)
            return A, B, f0
        
        self._linearize_jit = jit(_linearize_pure)

        # 3. Setup Discretization JIT (Exact ZOH)
        state_dim = self.first_order_state_n
        control_dim = self.first_order_control_n

        def _discretize_pure(x0, u0, dt):
            """Pure JAX implementation of Matrix Exponential Discretization"""
            A, B, f = self._linearize_jit(x0, u0)
            
            # Helper for batch matrix-vector mult (JAX version of bmatvec)
            def jax_bmatvec(Mat, vec):
                return jnp.einsum('bnm,bm->bn', Mat, vec)
            
            batch_size = x0.shape[0]
            control_dim = self.first_order_control_n

            # --- Construct M Matrix for ZOH ---
            # M = [A  B  f]
            #     [0  0  0]
            
            # Because JAX arrays are immutable, we construct parts and concatenate
            # Top row components:
            # A shape: (B, N, N)
            # B shape: (B, N, M)
            # f shape: (B, N, 1)

            M = np.zeros((batch_size, state_dim + control_dim + 1, state_dim + control_dim + 1))
            M[:, :state_dim, :state_dim] = A
            M[:, :state_dim, state_dim:state_dim+control_dim] = B
            M[:, :state_dim, state_dim+control_dim:] = f[..., None]

            AdBd = expm(dt * M)[:, :state_dim, :]
            Ad, Bd = AdBd[:, :, :state_dim], AdBd[:, :, state_dim:state_dim+control_dim]
            c = AdBd[:, :, state_dim+control_dim:]
            c = np.squeeze(c, axis=-1)

            fd = c - jax_bmatvec(Ad, x0) - jax_bmatvec(Bd, u0) + x0
            
            return Ad, Bd, fd

        self._discretize_jit = jit(_discretize_pure)
    
    def split_first(self, first: np.ndarray) -> List[np.ndarray]:
        """
        Splits the concatenated first-order vector into individual segments.

        Parameters
        ----------
        first : np.ndarray
            Batched vector of first-order states or controls, shape (batch_size, dim)

        Returns
        -------
        List of np.ndarray
            List of shape (batch_size, 1) per component.
        """
        return [first[:, i:i+1] for i in range(first.shape[1])]

    @abstractmethod
    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Computes the time derivative of the state vector.
        
        .. math:: \dot{x} = f(x, u)

        Parameters
        ----------
        first_order_state : np.ndarray
            Batched state vector :math:`x`, shape ``(batch_size, state_dim)``.
        first_order_control : np.ndarray
            Batched control vector :math:`u`, shape ``(batch_size, control_dim)``.
        np : module, optional
            Numerical backend (numpy or jax.numpy). Should be used for all mathematical 
            operations inside the method to support auto-differentiation.

        Returns
        -------
        np.ndarray
            The time derivative :math:`\dot{x}`, shape ``(batch_size, state_dim)``.
        """
        raise NotImplementedError

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian of the dynamics with respect to the state.

        .. math:: A = \frac{\partial f(x, u)}{\partial x}
        
        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape ``(batch_size, state_dim)``.
        first_order_control : np.ndarray
            Control vector, shape ``(batch_size, control_dim)``.

        Returns
        -------
        np.ndarray
            Jacobian matrix :math:`A`, shape ``(batch_size, state_dim, state_dim)``.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass does not implement this method analytically.
        """

        if self.jax_backend is None:
            raise NotImplementedError(
                "dfdx is not implemented and JAX is not installed. "
                "Install JAX to enable automatic Jacobians."
            )
        
        x = jnp.asarray(first_order_state)
        u = jnp.asarray(first_order_control)
        return np.asarray(self.dfdx_jax(x, u))

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian of the dynamics with respect to the control input.

        .. math:: B = \frac{\partial f(x, u)}{\partial u}
        
        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape ``(batch_size, state_dim)``.
        first_order_control : np.ndarray
            Control vector, shape ``(batch_size, control_dim)``.

        Returns
        -------
        np.ndarray
            Jacobian matrix :math:`B`, shape ``(batch_size, state_dim, control_dim)``.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass does not implement this method analytically.
        """

        if self.dfdu_jax is None:
            raise NotImplementedError(
                "dfdu is not implemented and JAX is not installed. "
                "Install JAX to enable automatic Jacobians."
            )
        
        x = jnp.asarray(first_order_state)
        u = jnp.asarray(first_order_control)
        return np.asarray(self.dfdu_jax(x, u))


    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Linearizes the nonlinear dynamics about a reference point :math:`(x_0, u_0)`.

        Expands the dynamics using a first-order Taylor series:
        
        .. math:: \dot{x} \approx f(x_0, u_0) + A(x - x_0) + B(u - u_0)

        Parameters
        ----------
        x0 : np.ndarray
            Reference state batch, shape ``(batch_size, state_dim)``.
        u0 : np.ndarray
            Reference control batch, shape ``(batch_size, control_dim)``.

        Returns
        -------
        A : np.ndarray
            State Jacobian :math:`\frac{\partial f}{\partial x}`, shape ``(batch_size, state_dim, state_dim)``.
        B : np.ndarray
            Control Jacobian :math:`\frac{\partial f}{\partial u}`, shape ``(batch_size, state_dim, control_dim)``.
        f0 : np.ndarray
            Nominal dynamics :math:`f(x_0, u_0)`, shape ``(batch_size, state_dim)``.
        """
        if self.jax_backend and self._linearize_jit:
            A, B, f0 = self._linearize_jit(jnp.asarray(x0), jnp.asarray(u0))
            return np.array(A), np.array(B), np.array(f0)
        
        f0 = self.f(x0, u0)
        A = self.dfdx(x0, u0)
        B = self.dfdu(x0, u0)
        return A, B, f0

    def discretize(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Discretizes the linearized system dynamics over a time step :math:`dt`.

        Uses the Matrix Exponential (Zero-Order Hold) for exact discretization of the 
        linear system, or a first-order Euler approximation.

        **Zero-Order Hold (ZOH):**
        
        .. math::

            \exp \left( \begin{bmatrix} A & B & f_0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} dt \right) 
            \approx \begin{bmatrix} A_d & B_d & f_d \\ 0 & I & 0 \\ 0 & 0 & 1 \end{bmatrix}

        Parameters
        ----------
        x0 : np.ndarray
            Reference state, shape ``(batch_size, state_dim)``.
        u0 : np.ndarray
            Reference control, shape ``(batch_size, control_dim)``.
        dt : float, default=0.1
            Time step duration in seconds.
        estimate : bool, default=True
            If True, uses Euler integration (:math:`A_d = I + A dt`).
            If False, uses exact matrix exponential (slower but more accurate).

        Returns
        -------
        Ad : np.ndarray
            Discrete state matrix.
        Bd : np.ndarray
            Discrete control matrix.
        fd : np.ndarray
            Discrete affine term (accounting for the linearization offset).
        """
        if self.jax_backend and self._discretize_jit:
            Ad, Bd, fd = self._discretize_jit(jnp.asarray(x0), jnp.asarray(u0), dt)
            return np.array(Ad), np.array(Bd), np.array(fd)

        state_dim = self.first_order_state_n

        A, B, f = self.linearize(x0, u0)

        if estimate:
            Ad = np.eye(state_dim) + A*dt
            Bd = B*dt
            fd = (f - bmatvec(A, x0) - bmatvec(B, u0))*dt

        else:
            batch_size = x0.shape[0]
            control_dim = self.first_order_control_n

            M = np.zeros((batch_size, state_dim + control_dim + 1, state_dim + control_dim + 1))
            M[:, :state_dim, :state_dim] = A
            M[:, :state_dim, state_dim:state_dim+control_dim] = B
            M[:, :state_dim, state_dim+control_dim:] = f[..., None]

            AdBd = expm(dt * M)[:, :state_dim, :]
            Ad, Bd = AdBd[:, :, :state_dim], AdBd[:, :, state_dim:state_dim+control_dim]
            c = AdBd[:, :, state_dim+control_dim:]
            c = np.squeeze(c, axis=-1)

            fd = c - bmatvec(Ad, x0) - bmatvec(Bd, u0) + x0

        return Ad, Bd, fd

    @property
    def first_state_names(self) -> List[str]:
        r"""
        List[str]: Symbolic names for the first-order state vector elements.
        
        Format: ``x_{primitive_idx}^{[derivative_order]}``
        """
        orders = self.state_derivative_orders
        return [f'x_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_control_names(self) -> List[str]:
        r"""
        List[str]: Symbolic names for the first-order control vector elements.

        Format: ``u_{primitive_idx}^{[derivative_order]}``
        """
        orders = self.control_derivative_orders
        return [rf'u_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_names(self) -> Tuple[List[str], List[str]]:
        """
        Tuple[List[str], List[str]]: Tuple containing (state names, control names).
        """
        return self.first_state_names, self.first_control_names
    
    def __repr__(self, verbose: bool = False) -> str:
        """
        Return a string representation of the Dynamics object.

        Provides either a concise single-line summary or a verbose detailed
        summary including derivative orders, symbolic names, and constants.

        Parameters
        ----------
        verbose : bool, optional
            If True, returns a detailed multi-line description including 
            derivative orders, symbolic state/control names, and constants. 
            Default is False (concise single-line summary).

        Returns
        -------
        str
            String representation of the Dynamics instance, formatted for
            easy inspection in notebooks or console output.
        """

        cls_name = self.__class__.__name__
        state_names, control_names = self.first_names
        
        if verbose:
            return (
                f"<{cls_name}>\n"
                f"Primitive states: {self.primitive_state_n}, Primitive controls: {self.primitive_control_n}\n"
                f"First-order state size: {self.first_order_state_n}, First-order control size: {self.first_order_control_n}\n"
                f"State derivative orders: {self.state_derivative_orders.tolist()}\n"
                f"Control derivative orders: {self.control_derivative_orders.tolist()}\n"
                f"State names: {state_names}\n"
                f"Control names: {control_names}\n"
                f"Holonomic: {self.holonomic}\n"
                f"Constants: {self.constants}\n"
            )
        else:
            return (
                f"<{cls_name} | "
                f"Constants: {self.constants}, "
                f"Primitive states={self.primitive_state_n}, primitive_controls={self.primitive_control_n}, "
                f"First-order state size: {self.first_order_state_n}, First-order control size: {self.first_order_control_n}, "
                f"holonomic={self.holonomic}>"
            )
    
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
            'wheels_distance': wheels_distance
        }

        super().__init__(constants=constants,
                         state_derivative_orders=[0, 0, 0],   # x, y, theta
                         control_derivative_orders=[0, 0],    # v, omega
                         holonomic=False)

        self.wd = self.constants['wheels_distance']

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

        return np.concatenate([v_l, v_r], axis=1)

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

class CarDynamics(Dynamics):
    r"""
    Dynamics for a rear-wheeled drive car model.

    This class implements kinematics for car-like vechicle based-on ackerman steering:

    .. math::
        \begin{align*}
        \dot{x} &= v \cos(\theta) \\
        \dot{y} &= v \sin(\theta) \\
        \dot{v} &= a \\
        \dot{\theta} &= v tan(\omega)/L
        \end{align*}

    where:

    - :math:`(x, y)` is the position of the robot in 2D space
    - :math:`\\theta` is the robot's orientation
    - :math:`v` is the linear velocity
    - :math:`a` is the linear acceleration
    - :math:`\\omega` is the angular velocity
    - :math:`l` is the distance between front and real wheels

    References
    ----------
    - https://github.com/AtsushiSakai/PythonRobotics
    - https://grauonline.de/wordpress/?page_id=3244

    Parameters
    ----------
    wheels_distance : float, optional
        The distance between the left and right wheels (default: 1.0)

    front_rear_length : float, optional
        The distance between the front and real wheels (default: 1.0)

    Attributes
    ----------
    wd : float
        The wheelbase, used to compute left/right wheel speeds.

    frd : float
        The car length, used to compute steering rate.
    """

    def __init__(self, wheels_distance=1.0, front_rear_length=1.0) -> None:
        """
        Initializes the Wheeled Mobile Robot (WMR) dynamics with physical parameters.

        Parameters
        ----------
        wheels_distance : float
            Distance between the wheels.
        """

        constants = {
            'wheels distance': wheels_distance,
            'front rear length': front_rear_length
        }
        super().__init__(constants=constants,
                         holonomic=False,
                         state_derivative_orders=[0, 0, 0, 0],   # x, y, v, theta
                         control_derivative_orders=[0, 0])    # a, omega

        self.wd = self.constants['wheels distance']
        self.frd = self.constants['front rear length']

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        if np is None: np = importlib.import_module("numpy")

        v = first_order_state[:, 2:3]
        theta = first_order_state[:, 3:4]
        a = first_order_control[:, 0:1]
        w = first_order_control[:, 1:2]

        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dv = a
        dtheta = v * np.tan(w) / self.frd

        return np.concatenate([dx, dy, dv, dtheta], axis=1)

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
        _, _, v, _ = self.split_first(first_order_state)
        _, w = self.split_first(first_order_control)

        zeros = np.zeros_like(v)
        ones = np.ones_like(v)

        df1 = np.concatenate([zeros, zeros], axis=1)
        df2 = np.concatenate([zeros, zeros], axis=1)
        df3 = np.concatenate([ ones, zeros], axis=1)
        df4 = np.concatenate([zeros, v/(self.frd*np.cos(w)**2)], axis=1)

        return np.stack([df1, df2, df3, df4], axis=1)

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        """
        Computes the Jacobian \\( \\frac{\\partial f}{\\partial x} \\) of the car dynamics.

        Parameters
        ----------
        first_order_state : np.ndarray
            State vector, shape (batch_size, 4)

        first_order_control : np.ndarray
            Control vector, shape (batch_size, 2)

        Returns
        -------
        np.ndarray
            Jacobian matrix, shape (batch_size, 4, 4)
        """
        x, _, v, theta = self.split_first(first_order_state)
        _, w = self.split_first(first_order_control)

        zeros = np.zeros_like(x)

        df1 = np.concatenate([zeros, zeros, np.cos(theta), -v * np.sin(theta)], axis=1)
        df2 = np.concatenate([zeros, zeros, np.sin(theta), v * np.cos(theta)], axis=1)
        df3 = np.concatenate([zeros, zeros, zeros, zeros], axis=1)
        df4 = np.concatenate([zeros, zeros, np.tan(w)/self.frd, zeros], axis=1)

        return np.stack([df1, df2, df3, df4], axis=1)

class Quadcopter2DDynamics(Dynamics):
    r"""
    Dynamics for a planar (2D) quadcopter system.

    The model assumes a rigid body with two vertically-oriented thrust-producing rotors.
    The dynamics are derived from Newton-Euler equations in 2D space, with the state ordered as:

    .. math::

        x = [x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}]^\top

    The continuous-time dynamics are given by:

    .. math::

        \begin{align}
        \dot{x} &= \dot{x} \\
        \d\dot{x} &= -\frac{u_1 + u_2}{m} \sin(\theta) \\
        \dot{y} &= \dot{y} \\
        \ddot{y} &= \frac{u_1 + u_2}{m} \cos(\theta) - g \\
        \dot{\theta} &= \dot{\theta} \\
        \ddot{\theta} &= \frac{L(u_1 - u_2)}{I}
        \end{align}

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
        mass : float, optional
            Mass of the quadcopter in kilograms (default: 1.0)

        gravity : float, optional
            Gravitational acceleration in m/s² (default: 9.81)

        arm_length : float, optional
            Distance from the center of mass to each rotor in meters (default: 0.5)

        inertia : float, optional
            Moment of inertia around the out-of-plane axis (default: 0.01)
        """
        constants = {'mass': mass, 'gravity': gravity, 'arm_length': arm_length, 'inertia': inertia}
        super().__init__(constants=constants, state_derivative_orders=[1, 1, 1], control_derivative_orders=[0, 0])
        self.m, self.g, self.L, self.I = mass, gravity, arm_length, inertia

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
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
        if np is None: np = importlib.import_module("numpy")

        dx = first_order_state[:, 1:2]
        theta = first_order_state[:, 4:5]
        dtheta = first_order_state[:, 5:6]
        dy = first_order_state[:, 3:4]
        
        u1 = first_order_control[:, 0:1]
        u2 = first_order_control[:, 1:2]

        total_u = u1 + u2
        ddx = -total_u * np.sin(theta) / self.m
        ddy = total_u * np.cos(theta) / self.m - self.g
        ddtheta = self.L * (u1 - u2) / self.I

        return np.concatenate([dx, ddx, dy, ddy, dtheta, ddtheta], axis=1)

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
                         control_derivative_orders=[0] * 4, # 4 control inputs
                         wrapable_primitive_state=[3, 4, 5])

        self.m = mass
        self.g = gravity
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

    def __init__(self, mass=1.0, gravity=9.81, Ix=0.01, Iy=0.01, Iz=0.02):
        constants = {'mass': mass, 'gravity': gravity, 'Ix': Ix, 'Iy': Iy, 'Iz': Iz}
        super().__init__(constants=constants,
                         state_derivative_orders=[1] * 3 + [0]*6,
                         control_derivative_orders=[0] * 4,
                         wrapable_primitive_state=[3, 4, 5])
        self.m, self.g, self.Ix, self.Iy, self.Iz = mass, gravity, Ix, Iy, Iz

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        if np is None: np = importlib.import_module("numpy")
        
        # Slicing state: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # x=0, y=1, z=2 are unused in deriv calc directly
        vx = first_order_state[:, 3:4]
        vy = first_order_state[:, 4:5]
        vz = first_order_state[:, 5:6]
        phi = first_order_state[:, 6:7]
        theta = first_order_state[:, 7:8]
        psi = first_order_state[:, 8:9]
        p = first_order_state[:, 9:10]
        q = first_order_state[:, 10:11]
        r = first_order_state[:, 11:12]

        u1 = first_order_control[:, 0:1]
        u2 = first_order_control[:, 1:2]
        u3 = first_order_control[:, 2:3]
        u4 = first_order_control[:, 3:4]

        sin, cos = np.sin, np.cos
        tan = np.tan

        dvx = u1 / self.m * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi))
        dvy = u1 / self.m * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi))
        dvz = u1 / self.m * (cos(phi) * cos(theta)) - self.g

        dphi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)

        dp = (u2 + (self.Iy - self.Iz) * q * r) / self.Ix
        dq = (u3 + (self.Iz - self.Ix) * p * r) / self.Iy
        dr = (u4 + (self.Ix - self.Iy) * p * q) / self.Iz

        return np.concatenate([vx, vy, vz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr], axis=1)
    
class QuadcopterV2Dynamics(Dynamics):
    """
    Quadcopter dynamics using individual rotor speeds as control inputs.
    
    """

    """
    Quadcopter dynamics using individual rotor speeds.
    """
    def __init__(self, inertia = [3.8e-3, 3.8e-3, 7.1e-3], mass = 5.2, gravity = 9.807,
                 arm_length = 0.32, thrust_constant = 3.13e-5, translational_drag = [0.1, 0.1, 0.15],
                 torque_constant = 7.5e-7, rotational_drag = [0.1, 0.1, 0.15], motor_inertia = 6e-5) -> None:
        constants = {
            'inertia': inertia, 'mass': mass, 'gravity': gravity, 'arm length': arm_length,
            'thrust constant': thrust_constant, 'translational drag': translational_drag,
            'torque constant': torque_constant, 'rotational drag': rotational_drag, 'motor inertia': motor_inertia
        }
        super().__init__(constants=constants, state_derivative_orders=[1]*6, control_derivative_orders=[0]*4)
        
        self.m, self.I, self.g = mass, inertia, gravity
        self.l, self.b, self.d = arm_length, thrust_constant, torque_constant
        self.Ct, self.Cr, self.Jr = translational_drag, rotational_drag, motor_inertia

        # --- Mixing Matrix (M) ---
        # Maps squared rotor speeds [w1^2, w2^2, w3^2, w4^2] to [Thrust, Roll_Torque, Pitch_Torque, Yaw_Torque]
        # u = M * w^2
        
        # Row 1: Thrust = b * (w1^2 + w2^2 + w3^2 + w4^2)
        # Row 2: Roll   = b * l * (w4^2 - w2^2)
        # Row 3: Pitch  = b * l * (w3^2 - w1^2)
        # Row 4: Yaw    = d * (w1^2 - w2^2 + w3^2 - w4^2)
        
        b, l, d = self.b, self.l, self.d
        
        self.M = np.array([
            [b,      b,      b,      b],      # Thrust (u1)
            [0,     -b*l,    0,      b*l],    # Roll Torque (u2)
            [-b*l,   0,      b*l,    0],      # Pitch Torque (u3)
            [d,     -d,      d,     -d]       # Yaw Torque (u4)
        ])

    def extract_force(self, first_order_control: np.ndarray) -> np.ndarray:
        """ 
        Converts vector of rotor speeds (w) to body frame forces/torques (u).
        
        u = M @ (w^2)

        Parameters
        ----------
        first_order_control : np.ndarray
            Rotor speeds [w1, w2, w3, w4], shape (Batch, 4)

        Returns
        -------
        np.ndarray
            Forces [Thrust, Roll, Pitch, Yaw], shape (Batch, 4)
        """
        # Element-wise square of rotor speeds
        w_2 = first_order_control**2
        
        # Batch matrix multiplication: (B, 4) @ (4, 4).T -> (B, 4)
        # We use self.M.T because w_2 is a row vector per batch item
        return w_2 @ self.M.T

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        if np is None: np = importlib.import_module("numpy")

        # --- State Extraction ---
        # Indices: x(0), dx(1), y(2), dy(3), z(4), dz(5), phi(6), dphi(7), theta(8), dtheta(9), psi(10), dpsi(11)
        # Using slicing [:, i:i+1] to keep dimensions (Batch, 1)
        dx     = first_order_state[:, 1:2]
        dy     = first_order_state[:, 3:4]
        dz     = first_order_state[:, 5:6]
        phi    = first_order_state[:, 6:7]
        dphi   = first_order_state[:, 7:8]
        theta  = first_order_state[:, 8:9]
        dtheta = first_order_state[:, 9:10]
        psi    = first_order_state[:, 10:11]
        dpsi   = first_order_state[:, 11:12]

        # --- Control Extraction & Force Calculation ---
        # Note: We implement the mixing logic manually here for JIT efficiency
        # instead of calling self.extract_force (which uses numpy/self.M) to avoid
        # capturing large constants in the JAX trace if not strictly necessary.
        
        w1 = first_order_control[:, 0:1]
        w2 = first_order_control[:, 1:2]
        w3 = first_order_control[:, 2:3]
        w4 = first_order_control[:, 3:4]

        w1_2, w2_2, w3_2, w4_2 = w1**2, w2**2, w3**2, w4**2

        # Forces/Torques
        u1 = self.b * (w1_2 + w2_2 + w3_2 + w4_2)
        u2 = self.b * self.l * (w4_2 - w2_2)
        u3 = self.b * self.l * (w3_2 - w1_2)
        u4 = self.d * (w1_2 - w2_2 + w3_2 - w4_2)

        # --- Dynamics ---
        cos, sin = np.cos, np.sin
        
        # Rotation matrix components
        ux = cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)
        uy = cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi)
        
        # Residual rotor angular momentum
        omega_r = w1 - w2 + w3 - w4

        # Linear Accelerations
        ddx = (u1 * ux - self.Ct[0] * dx) / self.m
        ddy = (u1 * uy - self.Ct[1] * dy) / self.m
        ddz = (u1 * cos(theta) * cos(phi) - self.Ct[2] * dz) / self.m - self.g

        # Angular Accelerations
        Ix, Iy, Iz = self.I
        ddphi = (u2 - self.Cr[0] * dphi**2 - self.Jr * omega_r * dtheta - (Iz - Iy) * dtheta * dpsi) / Ix
        ddtheta = (u3 - self.Cr[1] * dtheta**2 + self.Jr * omega_r * dphi - (Ix - Iz) * dphi * dpsi) / Iy
        ddpsi = (u4 - self.Cr[2] * dpsi**2 - (Iy - Ix) * dphi * dtheta) / Iz

        return np.concatenate([
            dx, ddx,
            dy, ddy,
            dz, ddz,
            dphi, ddphi,
            dtheta, ddtheta,
            dpsi, ddpsi
        ], axis=1)