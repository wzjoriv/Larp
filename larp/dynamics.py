from abc import ABC, abstractmethod
import importlib
import inspect
from types import ModuleType
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from itertools import chain
import warnings

from larp import const
from larp.fn import bmatvec
from larp.const import JAX_INSTALLED, MJX_INSTALLED

if JAX_INSTALLED:
    import jax.numpy as jnp
    from jax import jacfwd, jit, vmap, lax
    from jax.scipy.linalg import expm as jexpm

if MJX_INSTALLED:
    import mujoco
    from mujoco import mjx

"""
Author: Josue N Rivera

Dynamics classes
"""
        
class Dynamics(ABC):

    r"""
    Base class for representing general dynamical systems with support for
    higher-order derivatives, batched operations, and automatic differentiation via JAX.

    This class models systems in the first-order form:

    :math:`\dot{x} = f(x, u)`

    where :math:`x \in \mathbb{R}^{n}` is the flattened state vector and 
    :math:`u \in \mathbb{R}^{m}` is the flattened control vector.

    Key Features
    ------------

    * **Primitives vs. First-Order:** Maps "primitive" configuration variables (e.g., position :math:`p`) 
      to a flattened first-order vector containing derivatives (e.g., :math:`[p, \dot{p}, \ddot{p}]`).
    * **Auto-Differentiation:** If JAX is installed and analytical gradients are not provided, 
      Jacobians (:math:`\frac{\partial f}{\partial x}`, :math:`\frac{\partial f}{\partial u}`) 
      are computed automatically.
    * **Batched Operations:** All core methods (``f``, ``linearize``, ``discretize``) 
      support batched inputs for parallel simulation or trajectory optimization.
    * **JIT Compilation:** Can selectively compile linearization and discretization steps 
      using JAX for significant performance speedups on GPU/TPU.

    Attributes
    ----------
    :ivar constants: Dictionary of physical parameters or constants (e.g., mass, length).
    :ivar holonomic: Boolean flag indicating if the system is holonomic.
    :ivar state_derivative_orders: Array of highest derivative orders for each primitive state.
    :ivar control_derivative_orders: Array of highest derivative orders for each primitive control.
    :ivar primitive_state_n: Number of primitive state variables.
    :ivar primitive_control_n: Number of primitive control variables.
    :ivar first_order_state_n: Total dimension (:math:`n`) of the flattened first-order state vector.
    :ivar first_order_control_n: Total dimension (:math:`m`) of the flattened first-order control vector.
    :ivar first_state_orders: Array indicating the derivative order of each element in the flattened state vector.
    :ivar first_control_orders: Array indicating the derivative order of each element in the flattened control vector.
    :ivar state_primitive_mask: Boolean mask identifying indices of primitive variables (0-th order) in the state vector.
    :ivar control_primitive_mask: Boolean mask identifying indices of primitive variables in the control vector.
    :ivar first_wrapped_state_mask: Boolean mask of shape ``(first_order_state_n,)``. True at indices corresponding to
        states that require angular wrapping (e.g., :math:`[-\pi, \pi]`).
    :ivar jax_backend: Boolean indicating if JAX JIT compilation is enabled for this instance.

    Example
    -------

    Defining a simple system and performing linearization:

    .. code-block:: python

        # Define 1 primitive state (Order 1: pos, vel) and 1 primitive control (Order 0: force)
        dyn = Dynamics(state_derivative_orders=[1], control_derivative_orders=[0])
        
        # Dummy input batch (Batch size 1)
        x0 = np.zeros((1, 2)) # [pos, vel]
        u0 = np.zeros((1, 1)) # [force]
        
        # Compute continuous Jacobians (Linearization)
        A, B, f0 = dyn.linearize(x0, u0)
        print(A.shape) 
        
        # prints (1, 2, 2)
    """

    def __init__(self,
                 constants: Optional[Dict] = None, 
                 state_derivative_orders: List[int] = [1],
                 control_derivative_orders: List[int] = [0],
                 wrapable_primitive_state: Optional[List[int]] = None,
                 holonomic: Optional[bool] = None,
                 jax_backend = False) -> None:
        
        r"""
        Initializes the Dynamics instance.

        :param constants: Dictionary of physical parameters or constants used in the model (e.g., mass, length).
            Defaults to an empty dict.
        :type constants: dict, optional
        :param state_derivative_orders: A list specifying the highest derivative order for each primitive state variable.
            Example: ``[2, 1]`` implies:
            
            * Primitive 0 (e.g., Position): Has derivatives up to acceleration (Order 2). 
              State entries: :math:`[p_0, \dot{p}_0, \ddot{p}_0]`.
            * Primitive 1 (e.g., Heading): Has derivatives up to angular velocity (Order 1).
              State entries: :math:`[\theta, \dot{\theta}]`.
        :type state_derivative_orders: list[int]
        :param control_derivative_orders: Similar to ``state_derivative_orders`` but for control variables.
            Example: ``[0]`` implies control is the 0-th derivative (e.g., force/torque directly).
        :type control_derivative_orders: list[int]
        :param wrapable_primitive_state: Indices of primitive states that represent angles and require wrapping 
            to :math:`[-\pi, \pi]`. These indices refer to the *primitive* list, not the flattened state vector.
        :type wrapable_primitive_state: list[int], optional
        :param holonomic: Flag indicating if the system is holonomic.
        :type holonomic: bool, optional
        :param jax_backend: If ``True`` and JAX is installed, enables JIT compilation for step, rollout, linearization, 
            and discretization methods. This requires the subclass to implement ``f`` 
            in a way that accepts the ``np`` module argument for backend injection.
            Independently, if jacobians and hessians are not specified JAX will be used to compute them if JAX is installed.
        :type jax_backend: bool
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
        self.jax_backend = jax_backend

        self._dfdx_jit = None
        self._dfdu_jit = None
        self._dfdxx_jit = None
        self._dfduu_jit = None
        self._dfdux_jit = None
        self._dfdxu_jit = None
        self._linearize_jit = None
        self._discretize_zoh_jit = None
        self._discretize_euler_jit = None
        self._discretize_hessian_zoh_jit   = None
        self._discretize_hessian_euler_jit = None

        self._step_euler_jit = None
        self._step_rk4_jit = None

        self._rollout_euler_jit = None
        self._rollout_rk4_jit = None

        self._setup_jax_functions()

    @property
    @abstractmethod
    def angle_indices(self) -> List[int]:
        """Indices in the flattened state vector that represent angles (need wrapping)."""
        return []

    @property
    def heading_convention_offset(self) -> float:
        """
        Offset from the standard path-direction angle (``arctan2(dy, dx)``,
        measured from the ``+X`` world axis) to this model's yaw state slot.

        Planners that derive a heading from a 2-D path direction should add
        this offset before placing the value into the reference state vector.
        This property belongs on Dynamics because the dynamics model is the
        authoritative source of its own state-variable conventions — consistent
        with how ROS2/tf2, Drake, and ALTRO handle frame semantics.

        Default ``0.0`` (WMR, Car).
        Override to ``-pi/2`` for ``QuadcopterDynamics`` whose ``psi=0`` means
        nose→+Y, not nose→+X.
        """
        return 0.0

    def _setup_jax_functions(self):
        """
        Compiles JAX functions for derivatives, linearization, and discretization.
        """

        if "np" not in inspect.signature(self.f).parameters or not JAX_INSTALLED:
            self.jax_backend = False
            return

        # 1. Setup Wrapper
        def f_jax_wrapper(x, u):
            return self.f(x[None, :], u[None, :], np=jnp)[0]

        # 2. Detect if we are using AutoDiff (if subclass didn't override)
        using_autodiff_dfdx = (self.__class__.dfdx == Dynamics.dfdx)
        using_autodiff_dfdu = (self.__class__.dfdu == Dynamics.dfdu)
        using_autodiff_dfdxx = (self.__class__.dfdxx == Dynamics.dfdxx)
        using_autodiff_dfduu = (self.__class__.dfduu == Dynamics.dfduu)
        using_autodiff_dfdxu = (self.__class__.dfdxu == Dynamics.dfdxu)
        using_autodiff_dfdux = (self.__class__.dfdux == Dynamics.dfdux)

        # 3. Setup Jacobians (First Order)
        if using_autodiff_dfdx:
            self._dfdx_jit = jit(vmap(jacfwd(f_jax_wrapper, argnums=0)))
        
        if using_autodiff_dfdu:
            self._dfdu_jit = jit(vmap(jacfwd(f_jax_wrapper, argnums=1)))

        # 4. Setup Hessians (Second Order) - JAX Only
        # Computed as Jacobian of Jacobian: (B, n, n, n), (B, n, m, m), (B, n, n, m), and (B, n, m, n)
        if using_autodiff_dfdxx:
            self._dfdxx_jit = jit(vmap(jacfwd(jacfwd(f_jax_wrapper, argnums=0), argnums=0)))

        if using_autodiff_dfduu:
            self._dfduu_jit = jit(vmap(jacfwd(jacfwd(f_jax_wrapper, argnums=1), argnums=1)))

        if using_autodiff_dfdxu:
            self._dfdxu_jit = jit(vmap(jacfwd(jacfwd(f_jax_wrapper, argnums=0), argnums=1)))
        
        if using_autodiff_dfdux:
            self._dfdux_jit = jit(vmap(jacfwd(jacfwd(f_jax_wrapper, argnums=1), argnums=0)))


        #4. Linearization and discritization
        if self.jax_backend:
            
            _internal_dfdx = self._dfdx_jit if using_autodiff_dfdx else jit(vmap(jacfwd(f_jax_wrapper, argnums=0)))
            _internal_dfdu = self._dfdu_jit if using_autodiff_dfdu else jit(vmap(jacfwd(f_jax_wrapper, argnums=1)))

            def _linearize_pure(x, u):
                f0 = self.f(x, u, np=jnp)
                A = _internal_dfdx(x, u)
                B = _internal_dfdu(x, u)
                return A, B, f0
            
            self._linearize_jit = jit(_linearize_pure)

            state_dim = self.first_order_state_n
            control_dim = self.first_order_control_n

            # --- JIT A: Exact ZOH ---
            def _discretize_zoh(x0, u0, dt):
                A, B, f0 = _linearize_pure(x0, u0)
                
                # Affine Augmentation ZOH
                term_Ax = jnp.einsum('bij,bj->bi', A, x0)
                term_Bu = jnp.einsum('bij,bj->bi', B, u0)
                K = f0 - term_Ax - term_Bu

                top_row = jnp.concatenate([A, B, K[..., None]], axis=2)
                zeros_bottom = jnp.zeros((x0.shape[0], control_dim + 1, state_dim + control_dim + 1))
                M = jnp.concatenate([top_row, zeros_bottom], axis=1)
                
                Expm = jexpm(M * dt) 
                
                Ad = Expm[:, :state_dim, :state_dim]
                Bd = Expm[:, :state_dim, state_dim : state_dim + control_dim]
                fd = Expm[:, :state_dim, -1] 

                return Ad, Bd, fd
            
            # --- JIT B: Forward Euler Approximation ---
            def _discretize_euler(x0, u0, dt):
                A, B, f0 = _linearize_pure(x0, u0)
                
                # Ad = I + A*dt
                Ad = jnp.eye(state_dim) + A * dt
                
                # Bd = B*dt
                Bd = B * dt
                
                # fd = (f0 - A*x0 - B*u0) * dt
                term_Ax = jnp.einsum('bij,bj->bi', A, x0)
                term_Bu = jnp.einsum('bij,bj->bi', B, u0)
                fd = (f0 - term_Ax - term_Bu) * dt
                
                return Ad, Bd, fd

            self._discretize_zoh_jit = jit(_discretize_zoh)
            self._discretize_euler_jit = jit(_discretize_euler)

            # --- JIT C: Exact ZOH Discretized Hessians ---
            def _step_zoh_single(x, u, dt):
                A, B, f0 = _linearize_pure(x[None], u[None])
                A, B, f0 = A[0], B[0], f0[0]
                K = f0 - A @ x - B @ u
                
                top_row = jnp.concatenate([A, B, K[:, None]], axis=1)
                zeros_bottom = jnp.zeros((control_dim + 1, state_dim + control_dim + 1))
                M = jnp.concatenate([top_row, zeros_bottom], axis=0)
                
                Expm = jexpm(M * dt)
                Ad = Expm[:state_dim, :state_dim]
                Bd = Expm[:state_dim, state_dim : state_dim + control_dim]
                fd = Expm[:state_dim, -1]
                return Ad @ x + Bd @ u + fd
                
            _zoh_Fxx = vmap(jacfwd(jacfwd(_step_zoh_single, argnums=0), argnums=0), in_axes=(0, 0, None))
            _zoh_Fuu = vmap(jacfwd(jacfwd(_step_zoh_single, argnums=1), argnums=1), in_axes=(0, 0, None))
            _zoh_Fux = vmap(jacfwd(jacfwd(_step_zoh_single, argnums=1), argnums=0), in_axes=(0, 0, None))


            def _discretize_hessian_zoh(x, u, dt):
                return _zoh_Fxx(x, u, dt), _zoh_Fuu(x, u, dt), _zoh_Fux(x, u, dt)

            # --- JIT D: Forward Euler Discretized Hessians ---
            def _discretize_hessian_euler(x, u, dt):
                return self._dfdxx_jit(x, u) * dt, self._dfduu_jit(x, u) * dt, self._dfdux_jit(x, u) * dt

            self._discretize_hessian_zoh_jit   = jit(_discretize_hessian_zoh)
            self._discretize_hessian_euler_jit = jit(_discretize_hessian_euler)

            # --- JIT F: Forward Euler & RK4 Step (Batched) ---
            def _step_euler_pure(x, u, dt):
                return x + self.f(x, u, np=jnp) * dt

            def _step_rk4_pure(x, u, dt):
                k1 = self.f(x, u, np=jnp)
                k2 = self.f(x + 0.5 * dt * k1, u, np=jnp)
                k3 = self.f(x + 0.5 * dt * k2, u, np=jnp)
                k4 = self.f(x + dt * k3, u, np=jnp)
                return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            self._step_euler_jit = jit(vmap(_step_euler_pure, in_axes=(0, 0, None)))
            self._step_rk4_jit   = jit(vmap(_step_rk4_pure, in_axes=(0, 0, None)))

            # --- JIT G: Rollout (Batched Trajectories) ---
            def _make_rollout(step_fn):
                def rollout_fn(x0_batch, us_time_major, dt):
                    def scan_op(x_prev, u_curr):
                        x_next = step_fn(x_prev, u_curr, dt)
                        return x_next, x_next

                    _, xs_traj = lax.scan(scan_op, x0_batch, us_time_major)
                    return xs_traj
                return rollout_fn

            self._rollout_euler_jit = jit(_make_rollout(_step_euler_pure))
            self._rollout_rk4_jit   = jit(_make_rollout(_step_rk4_pure))

    
    def split_first(self, first: np.ndarray) -> List[np.ndarray]:
        """
        Splits a batched first-order vector into a list of individual components.

        This method effectively "unzips" the vector columns. If the input has shape 
        ``(batch_size, k)``, this returns a list of ``k`` arrays, each with shape ``(batch_size, 1)``.

        This is commonly used in ``f()`` to unpack the state vector into named variables 
        for cleaner physics equations.

        :param first: Batched vector of first-order states or controls. 
            Shape: ``(batch_size, total_dim)``.
        :type first: np.ndarray
        :return: A list containing one array for each element in the state/control vector.
        :rtype: List[np.ndarray]

        Example
        -------

        .. code-block:: python

            # State vector contains [x, y, theta]
            # Batch size = 1
            state = np.array([[1.0, 2.0, 3.14]])
            
            # Unpack into individual variables
            x, y, theta = dyn.split_first(state)
            
            # x is now [[1.0]], y is [[2.0]], theta is [[3.14]]
        """
        return [first[:, i:i+1] for i in range(first.shape[1])]

    @abstractmethod
    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Computes the continuous-time derivative of the state vector.

        :math:`\dot{x} = f(x, u)`

        :param first_order_state: Batched state vector :math:`x`. Shape: ``(batch_size, state_dim)``.
        :type first_order_state: np.ndarray
        :param first_order_control: Batched control vector :math:`u`. Shape: ``(batch_size, control_dim)``.
        :type first_order_control: np.ndarray
        :param np: Numerical backend to use (e.g., ``numpy`` or ``jax.numpy``). 
            If ``None``, defaults to standard NumPy. 
            **Note:** Implementations must use this argument for all math operations 
            (e.g., ``np.sin``, ``np.cos``) to support JAX auto-differentiation.
        :type np: module, optional
        :return: The time derivative :math:`\dot{x}`. Shape: ``(batch_size, state_dim)``.
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian of the dynamics with respect to the state vector.

        :math:`A = \frac{\partial f(x, u)}{\partial x}`

        If the subclass does not override this method and JAX is enabled, this will 
        use automatic differentiation.

        :param first_order_state: Batched state vector :math:`x`. Shape: ``(batch_size, state_n)``.
        :type first_order_state: np.ndarray
        :param first_order_control: Batched control vector :math:`u`. Shape: ``(batch_size, control_n)``.
        :type first_order_control: np.ndarray
        :return: Jacobian matrix :math:`A`. Shape: ``(batch_size, state_n, state_n)``.
        :rtype: np.ndarray
        :raises NotImplementedError: If JAX is not installed/enabled and the subclass provides no implementation.
        """

        if JAX_INSTALLED and self._dfdx_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdx_jit(x, u))
        
        raise NotImplementedError(
            "dfdx is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def dfdu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Jacobian of the dynamics with respect to the control vector.

        :math:`B = \frac{\partial f(x, u)}{\partial u}`

        If the subclass does not override this method and JAX is enabled, this will 
        use automatic differentiation.

        :param first_order_state: Batched state vector :math:`x`. Shape: ``(batch_size, state_n)``.
        :type first_order_state: np.ndarray
        :param first_order_control: Batched control vector :math:`u`. Shape: ``(batch_size, control_n)``.
        :type first_order_control: np.ndarray
        :return: Jacobian matrix :math:`B`. Shape: ``(batch_size, state_n, control_n)``.
        :rtype: np.ndarray
        :raises NotImplementedError: If JAX is not installed/enabled and the subclass provides no implementation.
        """

        if JAX_INSTALLED and self._dfdu_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdu_jit(x, u))

        raise NotImplementedError(
            "dfdu is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )
    
    def dfdxx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Hessian of the dynamics with respect to the state vector (Tensor).

        :math:`f_{xx} = \frac{\partial^2 f}{\partial x^2}`

        :return: Hessian Tensor. Shape: ``(batch_size, state_n, state_n, state_n)``.
        """
        if JAX_INSTALLED and self._dfdxx_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdxx_jit(x, u))
        
        raise NotImplementedError(
            "dfdxx is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def dfduu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Hessian of the dynamics with respect to the control vector (Tensor).

        :math:`f_{uu} = \frac{\partial^2 f}{\partial u^2}`

        :return: Hessian Tensor. Shape: ``(batch_size, state_n, control_n, control_n)``.
        """
        if JAX_INSTALLED and self._dfduu_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfduu_jit(x, u))
        
        raise NotImplementedError(
            "dfduu is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def dfdxu(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Cross-Hessian of the dynamics with respect to the state and control vectors (Tensor).

        :math:`f_{xu} = \frac{\partial^2 f}{\partial x \partial u}`

        :return: Hessian Tensor. Shape: ``(batch_size, state_n, state_n, control_n)``.
        """
        if JAX_INSTALLED and self._dfdxu_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdxu_jit(x, u))
        
        raise NotImplementedError(
            "dfdxx is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def dfdux(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Computes the Cross-Hessian of the dynamics with respect to the control and state vector (Tensor).

        :math:`f_{ux} = \frac{\partial^2 f}{\partial u \partial x}`

        :return: Hessian Tensor. Shape: ``(batch_size, state_n, control_n, state_n)``.
        """
        if JAX_INSTALLED and self._dfdux_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdux_jit(x, u))
        
        raise NotImplementedError(
            "dfdux is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def linearize(self, x0: np.ndarray, u0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Linearizes the system dynamics about a reference point :math:`(x_0, u_0)` using a first-order Taylor expansion.

        :math:`\dot{x} \approx f(x_0, u_0) + A(x - x_0) + B(u - u_0)`

        where:
        
        * :math:`A = \frac{\partial f}{\partial x} \big|_{x_0, u_0}` (State Jacobian)
        * :math:`B = \frac{\partial f}{\partial u} \big|_{x_0, u_0}` (Control Jacobian)

        :param x0: Reference state batch. Shape: ``(batch_size, state_dim)``.
        :type x0: np.ndarray
        :param u0: Reference control batch. Shape: ``(batch_size, control_dim)``.
        :type u0: np.ndarray
        :return: A tuple ``(A, B, f0)`` where:
            
            * **A** is the State Jacobian matrix.
            * **B** is the Control Jacobian matrix.
            * **f0** is the nominal dynamics :math:`f(x_0, u_0)`.
        :rtype: tuple

        Example
        -------

        .. code-block:: python

            # Linearize around origin for 10 batch items
            x_ref = np.zeros((10, 12))
            u_ref = np.zeros((10, 4))
            A, B, f0 = dyn.linearize(x_ref, u_ref)

        """
        if self.jax_backend and self._linearize_jit:
            A, B, f0 = self._linearize_jit(jnp.asarray(x0), jnp.asarray(u0))
            return np.asarray(A), np.asarray(B), np.asarray(f0)
        
        f0 = self.f(x0, u0)
        A = self.dfdx(x0, u0)
        B = self.dfdu(x0, u0)
        return A, B, f0

    def discretize(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Discretizes the continuous linearized dynamics over a time step :math:`dt`.

        Converts the affine system :math:`\dot{x} = Ax + Bu + K` into the discrete form:

        :math:`x_{k+1} = A_d x_k + B_d u_k + f_d`

        Methods
        -------

        * **Exact Zero-Order Hold (ZOH):** Used if ``estimate=False``. Computes the matrix exponential 
          of an augmented matrix to exactly integrate the linear system. This handles the drift term :math:`K` accurately.
        * **Forward Euler:** Used if ``estimate=True``. Approximates :math:`A_d \approx I + A dt`.

        :param x0: Reference state for linearization. Shape: ``(batch_size, state_dim)``.
        :type x0: np.ndarray
        :param u0: Reference control for linearization. Shape: ``(batch_size, control_dim)``.
        :type u0: np.ndarray
        :param dt: Time step duration in seconds. Default is 0.1.
        :type dt: float
        :param estimate: If ``True``, uses fast Euler integration. 
            If ``False``, uses exact Matrix Exponential (more accurate, computationally heavier).
        :type estimate: bool
        :return: A tuple ``(Ad, Bd, fd)`` containing:
            
            * **Ad**: Discrete state transition matrix.
            * **Bd**: Discrete control input matrix.
            * **fd**: Discrete affine offset term.
        :rtype: tuple

        Example
        -------

        .. code-block:: python
            Ad, Bd, fd = dyn.discretize(x_curr, u_curr, dt=0.05, estimate=False)
            
            # Predict next state
            x_next = Ad @ x_curr.T + Bd @ u_curr.T + fd
        """
        if self.jax_backend: #TODO: check if functions are not None
            if estimate:
                Ad, Bd, fd = self._discretize_euler_jit(jnp.asarray(x0), jnp.asarray(u0), dt)
            else:
                Ad, Bd, fd = self._discretize_zoh_jit(jnp.asarray(x0), jnp.asarray(u0), dt)

            return np.array(Ad), np.array(Bd), np.array(fd)

        state_dim = self.first_order_state_n
        control_dim = self.first_order_control_n

        A, B, f = self.linearize(x0, u0)

        if estimate:
            Ad = np.eye(state_dim) + A*dt
            Bd = B*dt
            fd = (f - bmatvec(A, x0) - bmatvec(B, u0))*dt

        else:
            batch_size = x0.shape[0]
            K = f - bmatvec(A, x0) - bmatvec(B, u0)

            M = np.zeros((batch_size, state_dim + control_dim + 1, state_dim + control_dim + 1))
            M[:, :state_dim, :state_dim] = A
            M[:, :state_dim, state_dim:state_dim+control_dim] = B
            M[:, :state_dim, -1] = K

            Expm = expm(M*dt)

            Ad = Expm[:, :state_dim, :state_dim]
            Bd = Expm[:, :state_dim, state_dim : state_dim + control_dim]
            fd = Expm[:, :state_dim, -1]

        return Ad, Bd, fd
    
    def discretize_hessian(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Discretise the second-order dynamics Hessians over a time step *dt*.
 
        Returns the discrete 3D tensors ``(F_xx, F_uu, F_ux)``.
 
        Methods
        -------
        **ZOH-exact** (``estimate=False``, requires ``jax_backend=True``):
            Differentiates the full ZOH step map :math:`x_{k+1}(x_k, u_k)`
            twice via JAX auto-diff, yielding the exact discrete Hessian of
            the one-step transition.  This is the correct quantity for
            second-order DDP and avoids the :math:`O(dt)` truncation error of
            the Euler approximation.
 
        **Euler approximation** (``estimate=True`` or no JAX):
            :math:`F_{xx} \approx f_{xx} \cdot dt`.  Fast, but introduces
            first-order truncation error.
 
        Parameters
        ----------
        x0 : (B, n)  Reference state batch.
        u0 : (B, m)  Reference control batch.
        dt : float   Time step [s].
        estimate : bool
            If ``True`` (default), use fast Euler scaling.
            If ``False``, use exact ZOH differentiation (JAX required).
 
        Returns
        -------
        F_xx : (B, n, n, n)   Discrete state-state Hessian.
        F_uu : (B, n, m, m)   Discrete control-control Hessian.
        F_ux : (B, n, m, n)   Discrete control-state cross-Hessian.
 
        Raises
        ------
        NotImplementedError
            If ``estimate=False`` and neither ``jax_backend`` nor an analytical
            override of ``dfdxx / dfduu / dfdux`` is available.
        """
        if self.jax_backend:
            if estimate:
                F_xx, F_uu, F_ux = self._discretize_hessian_euler_jit(jnp.asarray(x0), jnp.asarray(u0), dt)
            else:
                F_xx, F_uu, F_ux = self._discretize_hessian_zoh_jit(jnp.asarray(x0), jnp.asarray(u0), dt)

            return np.asarray(F_xx), np.asarray(F_uu), np.asarray(F_ux)
 
        if not estimate:
            warnings.warn(
                "discretize_hessian: ZOH-exact Hessians require jax_backend=True. "
                "Falling back to Euler approximation.",
                RuntimeWarning, stacklevel=2,
            )
        F_xx = self.dfdxx(x0, u0) * dt
        F_uu = self.dfduu(x0, u0) * dt
        F_ux = self.dfdux(x0, u0) * dt
        return F_xx, F_uu, F_ux

    def step(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate=True) -> np.ndarray:
        r"""
        Advances the system dynamics forward by one time step :math:`dt`.

        Supports batched operations and can use either a fast Forward Euler 
        approximation or a more accurate 4th-order Runge-Kutta (RK4) integration.
        If JAX is enabled, this operation is JIT-compiled and vectorized over the batch.

        :param x0: Current state batch :math:`x_k`. Shape: ``(batch_size, state_dim)``.
        :type x0: np.ndarray
        :param u0: Current control batch :math:`u_k`. Shape: ``(batch_size, control_dim)``.
        :type u0: np.ndarray
        :param dt: Time step duration in seconds. Default is 0.1.
        :type dt: float
        :param estimate: If ``True``, uses 1st-order Forward Euler integration. 
            If ``False``, uses 4th-order Runge-Kutta (RK4) integration.
        :type estimate: bool
        :return: The next state batch :math:`x_{k+1}`. Shape: ``(batch_size, state_dim)``.
        :rtype: np.ndarray

        Example
        -------

        .. code-block:: python

            # Step forward a batch of 10 states using exact RK4 integration
            x_curr = np.zeros((10, 4))  # [Batch, State_Dim]
            u_curr = np.ones((10, 2))   # [Batch, Control_Dim]
            
            x_next = dyn.step(x_curr, u_curr, dt=0.05, estimate=False)
            # x_next shape: (10, 4)
        """

        if self.jax_backend:
            if estimate:
                x0 = self._step_euler_jit(jnp.asarray(x0), jnp.asarray(u0), dt)
            else:
                x0 = self._step_rk4_jit(jnp.asarray(x0), jnp.asarray(u0), dt)

            return np.array(x0)
        
        if estimate:
            x = x0 + self.f(x0, u0) * dt
        else:
            k1 = self.f(x0, u0)
            k2 = self.f(x0 + 0.5 * dt * k1, u0)
            k3 = self.f(x0 + 0.5 * dt * k2, u0)
            k4 = self.f(x0 + dt * k3, u0)
            x  = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return x
    
    def rollout(self, x0: np.ndarray, us: np.ndarray, dt: float = 0.1, estimate:bool = True) -> np.ndarray:
        r"""
        Simulates a forward rollout of the system dynamics over a sequence of controls.

        This method expects a **time-major** control sequence (time is Axis 0). 
        It returns the resulting state trajectory starting from the first integration 
        step (i.e., it excludes the initial state ``x0`` from the returned array).

        :param x0: Initial state batch. Shape: ``(batch_size, state_dim)``.
        :type x0: np.ndarray
        :param us: Sequence of control batches over time. 
            Shape: ``(N_steps, batch_size, control_dim)``.
        :type us: np.ndarray
        :param dt: Time step duration in seconds. Default is 0.1.
        :type dt: float
        :param estimate: If ``True``, uses 1st-order Forward Euler integration. 
            If ``False``, uses 4th-order Runge-Kutta (RK4) integration.
        :type estimate: bool
        :return: The state trajectory, excluding the initial state. 
            Shape: ``(N_steps, batch_size, state_dim)``.
        :rtype: np.ndarray

        Example
        -------

        .. code-block:: python

            # Rollout 20 time steps for a batch of 5 agents
            x0 = np.zeros((5, 4))        # [Batch, State_Dim]
            us = np.ones((20, 5, 2))     # [Time, Batch, Control_Dim]
            
            # Predict future states using fast Euler estimation
            xs = dyn.rollout(x0, us, dt=0.05, estimate=True)
            
            # xs shape: (20, 5, 4)
            # xs[0] is the state at t=0.05s
            # xs[-1] is the state at t=1.0s
        """

        if self.jax_backend:
            if estimate:
                xs = self._rollout_euler_jit(jnp.asarray(x0), jnp.asarray(us), dt)
            else:
                xs = self._rollout_rk4_jit(jnp.asarray(x0), jnp.asarray(us), dt)

            return np.array(xs)

        N = us.shape[0]
        xs = np.zeros((N+1, *x0.shape))
        xs[0] = x0

        for k in range(N):
            xs[k+1] = self.step(xs[k], us[k], dt=dt, estimate=estimate)
        return xs[1:]

    @property
    def first_state_names(self) -> List[str]:
        r"""
        Generates symbolic names for each element of the flattened first-order state vector.

        Format: ``x_{primitive_idx}^{[derivative_order]}``

        :return: List of string labels.
        :rtype: List[str]

        **Example:**
        
        >>> dyn = Dynamics(state_derivative_orders=[1])
        >>> dyn.first_state_names
        ['x_0^{[0]}', 'x_0^{[1]}']
        """
        orders = self.state_derivative_orders
        return [f'x_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_control_names(self) -> List[str]:
        r"""
        Generates symbolic names for each element of the flattened first-order control vector.

        Format: ``u_{primitive_idx}^{[derivative_order]}``

        :return: List of string labels.
        :rtype: List[str]
        """
        orders = self.control_derivative_orders
        return [rf'u_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_names(self) -> Tuple[List[str], List[str]]:
        """
        Convenience property to get both state and control names.

        :return: A tuple containing ``(first_state_names, first_control_names)``.
        :rtype: Tuple[List[str], List[str]]
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

    Attributes
    ----------
    L : float
        The wheelbase (distance between front and rear axles).
    W : float
        The track width (distance between left and right wheels).
    body_length : float
        Visual length of the car chassis.
    body_width : float
        Visual width of the car chassis.
    wheel_len : float
        Visual length of the wheels.
    wheel_width : float
        Visual width of the wheels.
    """

    def __init__(self, 
                 wheelbase: float = 2.5,          
                 track_width: float = 1.5,        
                 body_dims: Tuple[float, float] = (3.5, 1.6), # (Length, Width)
                 wheel_dims: Tuple[float, float] = (0.5, 0.2), # (Length, Width)
                 jax_backend: bool = True) -> None:
        """
        Initializes the Car dynamics with physical and visual parameters.

        Parameters
        ----------
        wheelbase : float
            Distance between the front and rear axles (meters). Default 2.5.
        track_width : float
            Distance between the left and right wheels (meters). Default 1.5.
        body_dims : Tuple[float, float]
            Dimensions (length, width) of the chassis for visualization. Default (3.5, 1.6).
        wheel_dims : Tuple[float, float]
            Dimensions (length, width) of the wheels for visualization. Default (0.5, 0.2).
        jax_backend : bool
            If True, enables JAX JIT compilation for this system.
        """

        constants = {
            'wheelbase': wheelbase,
            'track_width': track_width,
            'body_length': body_dims[0],
            'body_width': body_dims[1],
            'wheel_length': wheel_dims[0],
            'wheel_width': wheel_dims[1]
        }
        super().__init__(constants=constants,
                         holonomic=False,
                         state_derivative_orders=[0, 0, 0, 0],   # x, y, v, theta
                         control_derivative_orders=[0, 0],       # a, omega
                         jax_backend=jax_backend)    

        self.wd  = self.constants['track_width']
        self.frd = self.constants['wheelbase']

    @property
    def angle_indices(self) -> List[int]:
        return [3]   # theta is at index 3 in [x, y, v, theta]

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

class QuadcopterDynamics(Dynamics):
    """
    Quadcopter Dynamics V1 (Updated)

    A numerically robust 6-DOF rigid body model for a quadcopter using Newton-Euler equations.
    This model explicitly tracks body rates (p, q, r) to avoid singularities associated with
    second derivatives of Euler angles.

    **State Vector (12-dim)**:
        [x, vx, y, vy, z, vz, phi, theta, psi, p, q, r]

        - x, y, z: Position (World frame)
        - vx, vy, vz: Linear Velocity (World frame)
        - phi, theta, psi: Euler Angles (Pitch, Roll, Yaw)
        - p, q, r: Angular Velocity (Body frame)

    **Control Vector (4-dim)**:
        [w1, w2, w3, w4] - Rotor speeds (rad/s)

    **Frame Convention (Custom)**:
        - x: Right (Lateral)
        - y: Forward (Longitudinal)
        - z: Up (Vertical)

    **Rotation Mapping:**
        - phi (Rot about X)   -> Pitch (Nose Up/Down)
        - theta (Rot about Y) -> Roll (Bank Left/Right)
        - psi (Rot about Z)   -> Yaw (Heading)

    **Motor Order ('x' config)**:
        1: Back-Right (-45 deg) | CW
        2: Back-Left (-135 deg) | CCW
        3: Front-Left (+135 deg)| CW
        4: Front-Right (+45 deg)| CCW

    **Motor Order ('+' config)**:
        1: Right (0 deg)   | CW
        2: Back (-90 deg)  | CCW
        3: Left (180 deg)  | CW
        4: Front (90 deg)  | CCW

    Based on the model by:
        Islam, M., Okasha, M., & Idres, M. M. (2017, December). Dynamics and control of quadcopter using linear model predictive control approach. In IOP conference series: materials science and engineering (Vol. 270, No. 1, p. 012007). IOP Publishing.
    """

    def __init__(self,
                 frame: str = 'x',
                 inertia = [3.8e-3, 3.8e-3, 7.1e-3], # Ix, Iy, Iz
                 mass = 1.0,
                 gravity = 9.807,
                 arm_length = 0.32,
                 thrust_constant = 3.13e-5, # kf
                 translational_drag = [0.1, 0.1, 0.15], # kt
                 torque_constant = 7.5e-7,  # km
                 rotational_drag = [0.1, 0.1, 0.15], # kr
                 motor_inertia = 6e-5 # Ir
                 ) -> None:

        # Save all local variables to constants dict automatically
        constants = {k: v for k,v in locals().items() if k != 'self'}

        # Structure:
        #   - 3 Pos (Order 1: x, vx)
        #   - 3 Angles (Order 0: phi, theta, psi -> integrated via kinematics)
        #   - 3 Rates (Order 0: p, q, r -> integrated via dynamics)
        super().__init__(constants=constants,
                         state_derivative_orders=[1, 1, 1] + [0, 0, 0] + [0, 0, 0],
                         control_derivative_orders=[0] * 4,
                         wrapable_primitive_state=[3, 4, 5],
                         jax_backend=True)

        self.frame = frame
        self.m, self.g, self.l = float(mass), float(gravity), float(arm_length)
        self.I = inertia
        self.Ix, self.Iy, self.Iz = self.I
        self.Ir, self.kf, self.km = float(motor_inertia), float(thrust_constant), float(torque_constant)
        self.kt, self.kr = np.array(translational_drag), np.array(rotational_drag)

        # --- Mixing Matrix Setup ---
        # Frame: X=Right, Y=Forward
        if frame == 'x':
            # 1: Back-Right (X>0, Y<0) -> -45 deg
            # 2: Back-Left  (X<0, Y<0) -> -135 deg
            # 3: Front-Left (X<0, Y>0) -> +135 deg
            # 4: Front-Right(X>0, Y>0) -> +45 deg
            angles = np.array([-np.pi/4, -3*np.pi/4, 3*np.pi/4, np.pi/4])
        elif frame == '+':
            # 1: Right (X>0, Y=0) -> 0 deg
            # 2: Back  (X=0, Y<0) -> -90 deg
            # 3: Left  (X<0, Y=0) -> 180 deg
            # 4: Front (X=0, Y>0) -> 90 deg
            angles = np.array([0.0, -np.pi/2, np.pi, np.pi/2])
        else:
            raise ValueError(f"Unknown frame: {frame}")

        # Motor positions (x, y) relative to center
        self.motor_pos = np.stack([self.l * np.cos(angles), self.l * np.sin(angles)], axis=1)

        # Build Mixing Matrix M
        # 1. Thrust
        row_thrust = np.ones(4) * self.kf

        # 2. Torque X (Pitch): Force * Y_dist
        # Positive Y (Front) -> Positive Torque X (Nose Up)
        row_pitch = self.kf * self.motor_pos[:, 1]

        # 3. Torque Y (Roll): Force * -X_dist
        # Positive X (Right) -> Negative Torque Y (Right Wing Up / Roll Left)
        # We define +Theta as rolling LEFT (Right wing up) to match Right Hand Rule on Y axis
        row_roll = self.kf * (-self.motor_pos[:, 0])

        # 4. Torque Z (Yaw)
        # 1(CW), 2(CCW), 3(CW), 4(CCW)
        yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])
        row_yaw = self.km * yaw_signs
        self.yaw_signs = yaw_signs

        # Stack: [Thrust, Pitch(Phi), Roll(Theta), Yaw(Psi)]
        # Note the order change to match state [phi, theta, psi]
        self.M = np.stack([row_thrust, row_pitch, row_roll, row_yaw], axis=0)
        self.inv_M = np.linalg.inv(self.M)
    def to_rotor_speed(self, u_thrust, u_pitch, u_roll, u_yaw):
        """
        Calculates rotor speeds from [Thrust, Pitch(X), Roll(Y), Yaw(Z)].
        """
        u_vec = np.concatenate([u_thrust, u_pitch, u_roll, u_yaw], axis=1)
        w2 = u_vec @ self.inv_M.T
        return np.sqrt(np.maximum(w2, 0.0))

    def to_force_torque(self, w):
        """Returns [Thrust, Pitch_Tau, Roll_Tau, Yaw_Tau]"""
        return (w**2) @ self.M.T

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        if np is None: np = importlib.import_module("numpy")

        # Unpack
        vel = first_order_state[:, 1:6:2]  # [vx, vy, vz]
        ang = first_order_state[:, 6:9]    # [phi, theta, psi]
        rate = first_order_state[:, 9:12]  # [p, q, r]

        phi, theta, psi = ang[:, 0:1], ang[:, 1:2], ang[:, 2:3]
        p, q, r = rate[:, 0:1], rate[:, 1:2], rate[:, 2:3]
        w = first_order_control

        # Forces & Torques 
        # u_vec index: 0=Thrust, 1=Pitch(X), 2=Roll(Y), 3=Yaw(Z)
        u_vec = self.to_force_torque(w)
        u1 = u_vec[:, 0:1]
        tau = u_vec[:, 1:4] # [Tx, Ty, Tz]

        omega_r = np.sum(w * self.yaw_signs, axis=1, keepdims=True)

        # Translational (R_bw)
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        # Z-axis column of R (Body Z expressed in World)
        R13 = sph * sps + cph * cps * sth
        R23 = cph * sps * sth - cps * sph
        R33 = cph * cth
        z_axis = np.concatenate([R13, R23, R33], axis=1)

        drag = self.kt * vel
        accel = (1.0/self.m) * (u1 * z_axis - drag) - np.array([[0,0,self.g]])

        # Rotational
        tx, ty, tz = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]
        
        # Cross product coupling
        cp_p = (self.Iy - self.Iz) * q * r
        cp_q = (self.Iz - self.Ix) * p * r
        cp_r = (self.Ix - self.Iy) * p * q
        
        gyro_p = self.Ir * q * omega_r
        gyro_q = -self.Ir * p * omega_r
        drag_rot = self.kr * rate

        dp = (tx + cp_p - gyro_p - drag_rot[:, 0:1]) / self.Ix
        dq = (ty + cp_q - gyro_q - drag_rot[:, 1:2]) / self.Iy
        dr = (tz + cp_r          - drag_rot[:, 2:3]) / self.Iz

        # Kinematics
        tt = np.tan(theta)
        dphi   = p + r * cph * tt + q * sph * tt
        dtheta = q * cph - r * sph
        dpsi   = r * (cph / cth) + q * (sph / cth)

        return np.concatenate([
            vel[:, 0:1], accel[:, 0:1],
            vel[:, 1:2], accel[:, 1:2],
            vel[:, 2:3], accel[:, 2:3],
            dphi, dtheta, dpsi,
            dp, dq, dr
        ], axis=1)

    @property
    def angle_indices(self) -> List[int]:
        return [6, 7, 8]

    @property
    def heading_convention_offset(self) -> float:
        """psi=0 means nose→+Y (Forward); arctan2 uses +X=0 → offset is -pi/2."""
        return -np.pi / 2

class HighFidelityQuadcopterDynamics(Dynamics):
    r"""
    A high-fidelity 6-DOF rigid-body dynamics model for a quadcopter.

    This class models the physical behavior of a quadrotor UAV, including advanced aerodynamic
    effects, actuator dynamics, and electrical system constraints. It is designed for
    Model Predictive Control (MPC) and realistic flight simulation.

    **State Vector (17-Dimensional):**

    * **Position** (World Frame): :math:`[x, y, z]`
    * **Velocity** (World Frame): :math:`[v_x, v_y, v_z]`
    * **Attitude** (Unit Quaternion, Body-to-World): :math:`[q_w, q_x, q_y, q_z]`
    * **Angular Velocity** (Body Frame): :math:`[p, q, r]`
    * **Rotor Speeds**: :math:`[\omega_1, \omega_2, \omega_3, \omega_4]`

    **Control Vector (4-Dimensional):**

    * **PWM Inputs**: :math:`[u_1, u_2, u_3, u_4]` normalized to :math:`[0.0, 1.0]`.

    **Physical Model & Equations:**

    1.  **Motor Dynamics & Electrical System:**
        Motors are modeled as first-order lag systems subject to voltage constraints.
        
        .. math::
            V_{load} = V_{batt} - R_{int} \sum (k_i \omega^2) \\
            \omega_{max}(t) = \omega_{nom} \frac{V_{load}}{V_{max}} \\
            \dot{\omega} = \frac{\omega_{target}(u) - \omega}{\tau_{motor}}

    2.  **Aerodynamic Forces:**
        Includes quadratic drag, relative airflow (wind), and ground effect.

        .. math::
            F_{drag} = - (C_{lin} v_{air} + C_{quad} v_{air} |v_{air}|) \\
            T_{total} = \sum (k_f \omega_i^2) \cdot \left(1 + k_{ge} \left(\frac{h_{ge}}{z}\right)^2 \right)

    3.  **Rotational Dynamics (Euler Equations):**
        Includes gyroscopic precession, blade flapping, and inertial coupling.

        .. math::
            \dot{\Omega} = I^{-1} \left( \tau_{motor} + \tau_{flap} - \Omega \times (I \Omega) - \sum I_r (\Omega \times \omega_i \hat{z}) \right)

    **Coordinate System:**
    
    * **World Frame:** Inertial, Z-up (Gravity = -9.81 on Z).
    * **Body Frame:** x-Forward, y-Left, z-Up (Standard ROS/ENU convention).
        *Note: The mixing matrix implementation assumes 'x' config with specific motor angles.*

    :ivar m: Mass of the vehicle [kg].
    :ivar g: Gravitational acceleration [m/s^2].
    :ivar I: Inertia tensor (diagonal) [kg m^2].
    :ivar com: Center of Mass offset from geometric center [m].
    """

    def __init__(self,
                 frame: str = 'x',
                 mass: float = 5.2,
                 gravity: float = 9.81,
                 inertia: List[float] = [0.04, 0.04, 0.08],
                 arm_length: float = 0.35,
                 motor_time_constant: float = 0.05,
                 max_rpm: float = 12000,
                 min_rpm: float = 1000,
                 thrust_constant: float = 6.0e-5,
                 torque_constant: float = 1.0e-6,
                 motor_inertia: float = 1e-4,
                 drag_coeffs_linear: List[float] = [0.1, 0.1, 0.1],
                 drag_coeffs_quadratic: List[float] = [0.4, 0.4, 0.6],
                 rotational_drag: List[float] = [0.2, 0.2, 0.2],
                 blade_flapping_coeff: float = 0.05,
                 ground_effect_coeff: float = 0.1,
                 ground_effect_height: float = 0.5,
                 battery_voltage_max: float = 25.2,
                 battery_resistance: float = 0.02,
                 current_draw_coeff: float = 0.00015,
                 com_offset: List[float] = [0.0, 0.0, 0.0]) -> None:
        """
        Initialize the High Fidelity Dynamics model parameters.

        :param frame: Multirotor configuration. Options: ``'x'`` or ``'+'``.
        :param mass: Total mass of the vehicle [kg].
        :param gravity: Gravitational acceleration (positive scalar) [m/s^2].
        :param inertia: Principal moments of inertia ``[Ix, Iy, Iz]`` [kg m^2].
        :param arm_length: Distance from geometric center to motor hub [m].
        
        **Actuator Parameters:**
        
        :param motor_time_constant: Time constant :math:`\\tau` for motor response (63% rise time) [s].
        :param max_rpm: Maximum propeller RPM at full voltage (no load).
        :param min_rpm: Idle RPM when armed (PWM = 0.0).
        :param thrust_constant: Coefficient :math:`k_f` where :math:`F = k_f \\omega^2` [N/(rad/s)^2].
        :param torque_constant: Coefficient :math:`k_m` where :math:`\\tau = k_m \\omega^2` [Nm/(rad/s)^2].
        :param motor_inertia: Rotational inertia of the motor bell + propeller :math:`I_r` [kg m^2].

        **Aerodynamic Parameters:**
        
        :param drag_coeffs_linear: Linear drag coefficients ``[C_x, C_y, C_z]`` [Ns/m].
        :param drag_coeffs_quadratic: Quadratic drag coefficients ``[C_x, C_y, C_z]`` [Ns^2/m^2].
        :param rotational_drag: Drag torque coefficients on body rotation ``[C_{rp}, C_{rq}, C_{rr}]``.
        :param blade_flapping_coeff: Coefficient for H-force/Blade flapping moment [Ns].
        :param ground_effect_coeff: Maximum thrust increase percentage (e.g., 0.1 = 10%) at z=0.
        :param ground_effect_height: Height [m] at which ground effect becomes negligible.

        **Electrical & Physical Imperfections:**
        
        :param battery_voltage_max: Fully charged battery voltage [V].
        :param battery_resistance: Internal resistance of the battery [Ohms].
        :param current_draw_coeff: Factor relating RPM to current :math:`I = k_i \\omega^2` [A/(rad/s)^2].
        :param com_offset: Displacement of Center of Mass from Geometric Center ``[dx, dy, dz]`` [m].
        """

        constants = {k: v for k, v in locals().items() if k != 'self'}

        super().__init__(constants=constants,
                         state_derivative_orders=[1]*3 + [0]*4 + [0]*3 + [0]*4,
                         control_derivative_orders=[0]*4,
                         wrapable_primitive_state=[],
                         jax_backend=True)

        self.m = float(mass)
        self.g = float(gravity)
        self.l = float(arm_length)
        self.I = np.array(inertia)
        self.Ix, self.Iy, self.Iz = inertia
        
        self.tau_m = float(motor_time_constant)
        self.max_w_nominal = float(max_rpm * 2 * np.pi / 60.0)
        self.min_w = float(min_rpm * 2 * np.pi / 60.0)
        
        self.kf = float(thrust_constant)
        self.km = float(torque_constant)
        self.Ir = float(motor_inertia)

        self.Cd_lin = np.array(drag_coeffs_linear)
        self.Cd_quad = np.array(drag_coeffs_quadratic)
        self.Cr = np.array(rotational_drag)
        self.K_flap = float(blade_flapping_coeff)
        
        self.ge_k = float(ground_effect_coeff)
        self.ge_h = float(ground_effect_height)
        
        # Electrical
        self.v_max = float(battery_voltage_max)
        self.r_int = float(battery_resistance)
        self.k_i = float(current_draw_coeff)
        
        # Physical
        self.com = np.array(com_offset)
        
        # Frame Mixing
        if frame == 'x':
            angles = np.array([-np.pi/4, -3*np.pi/4, 3*np.pi/4, np.pi/4])
        elif frame == '+':
            angles = np.array([0.0, -np.pi/2, np.pi, np.pi/2])
        else:
            raise ValueError(f"Unknown frame: {frame}")

        # Geometric Positions (x, y)
        pos_x = self.l * np.cos(angles)
        pos_y = self.l * np.sin(angles)
        
        # Effective Moment Arms (Position relative to CoM)
        self.eff_rx = pos_x - self.com[0]
        self.eff_ry = pos_y - self.com[1]
        
        # Used for helper torque calcs
        self.motor_pos = np.stack([pos_x, pos_y], axis=1)
        self.yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])

    # =========================================================================
    #                               CORE DYNAMICS
    # =========================================================================

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, 
          wind_world: np.ndarray = None, np: Optional[ModuleType] = None) -> np.ndarray:
        
        if np is None: np = importlib.import_module("numpy")
        if wind_world is None: wind_world = np.zeros((first_order_state.shape[0], 3))

        # --- Unpack ---
        vel = first_order_state[:, 1:6:2]  # [vx, vy, vz]
        quat = first_order_state[:, 6:10]  # [qw, qx, qy, qz]
        rate = first_order_state[:, 10:13] # [p, q, r]
        w_curr = first_order_state[:, 13:17]
        z = first_order_state[:, 4:5]
        
        qw, qx, qy, qz = quat[:,0:1], quat[:,1:2], quat[:,2:3], quat[:,3:4]
        
        # --- Relative Airspeed ---
        v_air_world = vel - wind_world
        
        r11 = 1.0 - 2.0*(qy**2 + qz**2); r12 = 2.0*(qx*qy - qw*qz); r13 = 2.0*(qx*qz + qw*qy)
        r21 = 2.0*(qx*qy + qw*qz);     r22 = 1.0 - 2.0*(qx**2 + qz**2); r23 = 2.0*(qy*qz - qw*qx)
        r31 = 2.0*(qx*qz - qw*qy);     r32 = 2.0*(qy*qz + qw*qx);     r33 = 1.0 - 2.0*(qx**2 + qy**2)
        
        vx, vy, vz = v_air_world[:, 0:1], v_air_world[:, 1:2], v_air_world[:, 2:3]
        u_a = r11*vx + r21*vy + r31*vz
        v_a = r12*vx + r22*vy + r32*vz
        w_a = r13*vx + r23*vy + r33*vz

        # --- Electrical & Motor Dynamics ---
        u_pwm = np.clip(first_order_control, 0.0, 1.0)
        
        # 1. Estimate Load Current (Sum of Squares approx for Power)
        total_w2 = np.sum(w_curr**2, axis=1, keepdims=True)
        I_total = self.k_i * total_w2
        
        # 2. Voltage Sag: V_load = V_max - I * R
        v_load = self.v_max - (I_total * self.r_int)
        
        # 3. Dynamic RPM Limit
        max_w_dynamic = np.maximum(
            self.max_w_nominal * (v_load / self.v_max),
            self.min_w,
        )
        
        # 4. Target Speed
        w_target = self.min_w + u_pwm * (max_w_dynamic - self.min_w)
        dw = (w_target - w_curr) / self.tau_m

        # --- Forces & Torques ---
        T_i = self.kf * (w_curr ** 2)
        ge_factor = 1.0 + self.ge_k * (self.ge_h / np.maximum(z, 0.1))**2
        T_total = np.sum(T_i, axis=1, keepdims=True) * ge_factor
        
        # [Updated] Moment Calculation using Effective positions (relative to CoM)
        # Torque X = Sum(T_i * y_dist)
        # Torque Y = Sum(T_i * -x_dist)
        Mx = np.sum(T_i * self.eff_ry[None, :], axis=1, keepdims=True)
        My = np.sum(T_i * (-self.eff_rx[None, :]), axis=1, keepdims=True)
        Mz = np.sum(self.yaw_signs[None, :] * self.km * (w_curr**2), axis=1, keepdims=True)
        
        M_flap_x = -self.K_flap * v_a
        M_flap_y = +self.K_flap * u_a
        
        tau_body = np.concatenate([Mx + M_flap_x, My + M_flap_y, Mz], axis=1)

        # --- Translational ---
        D_bx = -(self.Cd_lin[0] * u_a + self.Cd_quad[0] * u_a * np.abs(u_a))
        D_by = -(self.Cd_lin[1] * v_a + self.Cd_quad[1] * v_a * np.abs(v_a))
        D_bz = -(self.Cd_lin[2] * w_a + self.Cd_quad[2] * w_a * np.abs(w_a))
        
        F_bx, F_by, F_bz = D_bx, D_by, T_total + D_bz
        
        F_wx = r11*F_bx + r12*F_by + r13*F_bz
        F_wy = r21*F_bx + r22*F_by + r23*F_bz
        F_wz = r31*F_bx + r32*F_by + r33*F_bz
        
        accel = np.concatenate([F_wx/self.m, F_wy/self.m, F_wz/self.m - self.g], axis=1)

        # --- Rotational ---
        p, q, r = rate[:,0:1], rate[:,1:2], rate[:,2:3]
        omega_r = np.sum(w_curr * self.yaw_signs[None, :], axis=1, keepdims=True)
        
        gyro_p = self.Ir * q * omega_r
        gyro_q = -self.Ir * p * omega_r
        drag_rot = self.Cr * rate
        
        cp_p = (self.Iz - self.Iy) * q * r
        cp_q = (self.Ix - self.Iz) * p * r
        cp_r = (self.Iy - self.Ix) * p * q
        
        dp = (tau_body[:, 0:1] - cp_p - gyro_p - drag_rot[:, 0:1]) / self.Ix
        dq = (tau_body[:, 1:2] - cp_q - gyro_q - drag_rot[:, 1:2]) / self.Iy
        dr = (tau_body[:, 2:3] - cp_r          - drag_rot[:, 2:3]) / self.Iz

        # --- Kinematics ---
        dqw = 0.5 * (-qx*p - qy*q - qz*r)
        dqx = 0.5 * ( qw*p + qy*r - qz*q)
        dqy = 0.5 * ( qw*q - qx*r + qz*p)
        dqz = 0.5 * ( qw*r + qx*q - qy*p)

        # Baumgarte quaternion normalisation stabilisation.
        q_err = qw**2 + qx**2 + qy**2 + qz**2 - 1.0
        dqw -= const.QUAT_BAUMGARTE_FACTOR * q_err * qw
        dqx -= const.QUAT_BAUMGARTE_FACTOR * q_err * qx
        dqy -= const.QUAT_BAUMGARTE_FACTOR * q_err * qy
        dqz -= const.QUAT_BAUMGARTE_FACTOR * q_err * qz

        return np.concatenate([
            vel[:, 0:1], accel[:, 0:1],
            vel[:, 1:2], accel[:, 1:2],
            vel[:, 2:3], accel[:, 2:3],
            dqw, dqx, dqy, dqz,
            dp, dq, dr,
            dw
        ], axis=1)

    @property
    def angle_indices(self) -> List[int]:
        return []

    # =========================================================================
    #                            HELPER METHODS
    # =========================================================================

    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Returns [x, y, z] from state."""
        return state[..., [0, 2, 4]]

    def get_velocity(self, state: np.ndarray) -> np.ndarray:
        """Returns [vx, vy, vz] from state."""
        return state[..., [1, 3, 5]]

    def get_quaternion(self, state: np.ndarray) -> np.ndarray:
        """Returns [qw, qx, qy, qz] from state."""
        return state[..., 6:10]

    def get_euler_angles(self, state: np.ndarray) -> np.ndarray:
        """
        Returns [phi, theta, psi] (Roll, Pitch, Yaw) from Quaternion state.
        Useful for logging and visualization.
        """
        q = self.get_quaternion(state)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Clamp out of bounds for numerical stability
        sinp = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))
        pitch = sinp

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.stack([roll, pitch, yaw], axis=-1)

    def get_body_rates(self, state: np.ndarray) -> np.ndarray:
        """Returns [p, q, r] from state."""
        return state[..., 10:13]

    def get_euler_rates(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the time derivative of Euler Angles [dphi, dtheta, dpsi]
        from the current Body Rates [p, q, r] and Attitude.
        """
        # 1. Get current orientation
        euler = self.get_euler_angles(state)
        phi   = euler[..., 0:1] # Roll
        theta = euler[..., 1:2] # Pitch
        
        # 2. Get current body rates
        rates = self.get_body_rates(state)
        p, q, r = rates[..., 0:1], rates[..., 1:2], rates[..., 2:3]
        
        # 3. Apply Kinematic Transformation Matrix
        tan_theta = np.tan(theta)
        cos_theta = np.cos(theta)
        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        
        # Avoid division by zero for safety (return 0 if exactly 90 deg)
        safe_cos_theta = np.where(np.abs(cos_theta) < 1e-6, 1e-6, cos_theta)
        
        dphi   = p + (q * sin_phi + r * cos_phi) * tan_theta
        dtheta = q * cos_phi - r * sin_phi
        dpsi   = (q * sin_phi + r * cos_phi) / safe_cos_theta
        
        return np.concatenate([dphi, dtheta, dpsi], axis=-1)

    def get_motor_speeds(self, state: np.ndarray) -> np.ndarray:
        """Returns [w1, w2, w3, w4] (rad/s) from state."""
        return state[..., 13:17]

    def get_rotation_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the 3x3 Rotation Matrix (Body -> World) for a batch of states.
        Shape: (Batch, 3, 3)
        """
        q = self.get_quaternion(state)
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        r11 = 1 - 2*(y**2 + z**2); r12 = 2*(x*y - w*z);     r13 = 2*(x*z + w*y)
        r21 = 2*(x*y + w*z);     r22 = 1 - 2*(x**2 + z**2); r23 = 2*(y*z - w*x)
        r31 = 2*(x*z - w*y);     r32 = 2*(y*z + w*x);     r33 = 1 - 2*(x**2 + y**2)
        
        R = np.stack([
            np.stack([r11, r12, r13], axis=-1),
            np.stack([r21, r22, r23], axis=-1),
            np.stack([r31, r32, r33], axis=-1)
        ], axis=-2)
        
        return R

    def body_to_world(self, state: np.ndarray, vector_body: np.ndarray) -> np.ndarray:
        """
        Rotates a vector from Body Frame to World Frame.
        vector_body shape: (Batch, 3)
        """
        R = self.get_rotation_matrix(state) # (Batch, 3, 3)
        # Batch Matmul: (B, 3, 3) @ (B, 3, 1) -> (B, 3, 1)
        return (R @ vector_body[..., None])[..., 0]

    def world_to_body(self, state: np.ndarray, vector_world: np.ndarray) -> np.ndarray:
        """
        Rotates a vector from World Frame to Body Frame.
        """
        R = self.get_rotation_matrix(state)
        # Transpose R for inverse rotation
        R_T = np.transpose(R, (0, 2, 1))
        return (R_T @ vector_world[..., None])[..., 0]

    def get_thrust_and_torque(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the instantaneous [Total Thrust (N)] and [Torque_x, Torque_y, Torque_z (Nm)]
        generated by the motors in the current state (ignoring drag/flapping).
        Useful for checking saturation or actuator limits.
        """
        w = self.get_motor_speeds(state)
        
        # Thrust
        T_i = self.kf * (w ** 2)
        
        # Ground Effect (re-calculated here for accuracy)
        z = self.get_position(state)[..., 2:3]
        ge_factor = 1.0 + self.ge_k * (self.ge_h / np.maximum(z, 0.1))**2
        thrust_total = np.sum(T_i, axis=1, keepdims=True) * ge_factor
        
        # Torques using Effective Moment Arms
        Mx = np.sum(T_i * self.eff_ry[None, :], axis=1, keepdims=True)
        My = np.sum(T_i * (-self.eff_rx[None, :]), axis=1, keepdims=True)
        Mz = np.sum(self.yaw_signs[None, :] * self.km * (w ** 2), axis=1, keepdims=True)
        
        return thrust_total, np.concatenate([Mx, My, Mz], axis=1)

    def check_safety_bounds(self, state: np.ndarray, 
                            x_lim=(-100, 100), 
                            y_lim=(-100, 100), 
                            z_lim=(0, 100), 
                            tilt_limit_deg=45) -> bool:
        """
        Simple safety check for simulation. Returns False if out of bounds.
        """
        pos = self.get_position(state)
        x, y, z = pos[0, 0], pos[0, 1], pos[0, 2]
        
        if not (x_lim[0] < x < x_lim[1]): return False
        if not (y_lim[0] < y < y_lim[1]): return False
        if not (z_lim[0] < z < z_lim[1]): return False
        
        # Tilt check
        euler = self.get_euler_angles(state)
        roll, pitch = euler[0, 0], euler[0, 1]
        limit = np.deg2rad(tilt_limit_deg)
        
        if abs(roll) > limit or abs(pitch) > limit: return False
        
        return True
    
    def get_rpm(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the motor speeds in Revolutions Per Minute (RPM).
        Useful for checking against real-world motor limits.
        """
        w_rads = self.get_motor_speeds(state)
        return w_rads * (60.0 / (2.0 * np.pi))
    
    def get_current(self, state: np.ndarray) -> np.ndarray:
        """Returns instantaneous current draw (Amps)"""
        w_curr = self.get_motor_speeds(state)
        total_w2 = np.sum(w_curr**2, axis=1, keepdims=True)
        return self.k_i * total_w2

    def get_voltage(self, state: np.ndarray) -> np.ndarray:
        """Returns instantaneous battery voltage (Volts) under load"""
        I = self.get_current(state)
        return self.v_max - (I * self.r_int)

    @staticmethod
    def euler_to_quat(phi, theta, psi):
        """Converts Euler Angles to Unit Quaternion [w, x, y, z]"""
        cy = np.cos(psi * 0.5); sy = np.sin(psi * 0.5)
        cp = np.cos(phi * 0.5); sp = np.sin(phi * 0.5)
        ct = np.cos(theta * 0.5); st = np.sin(theta * 0.5)
        
        w = cp * ct * cy + sp * st * sy
        x = sp * ct * cy - cp * st * sy
        y = cp * st * cy + sp * ct * sy
        z = cp * ct * sy - sp * st * cy
        return np.array([w, x, y, z])

class SmallFixedWingDynamics(Dynamics):
    r"""
    6-DOF small fixed-wing (UAV) dynamics with standard aerodynamic model.

    **State vector (12-dim)**:
        [x, vx, y, vy, z, vz, phi, theta, psi, p, q, r]

        - x, y    : horizontal position (World frame, right-forward convention)
        - z       : altitude (positive up)
        - vx,vy,vz: linear velocity (World frame)
        - phi     : roll  (rotation about body X)
        - theta   : pitch (rotation about body Y)
        - psi     : yaw   (heading, rotation about body Z)
        - p, q, r : angular rates (body frame)

    **Control vector (4-dim)**:
        [delta_t, delta_a, delta_e, delta_r]

        - delta_t : throttle ∈ [0, 1]
        - delta_a : aileron  deflection [rad]
        - delta_e : elevator deflection [rad]
        - delta_r : rudder   deflection [rad]

    **Frame convention**: same as QuadcopterDynamics — X right, Y forward, Z up.
    ``psi=0`` → nose points +Y; ``heading_convention_offset = -π/2``.

    Aerodynamic model
    -----------------
    Forces and moments follow the standard linearised stability-axis formulation
    from Beard & McLain (2012), *Small Unmanned Aircraft: Theory and Practice*,
    Princeton University Press, Chapters 2-4.

    Lift:   ``C_L = C_L0 + C_Lα·α + C_Lq·(c̄/(2V))·q + C_Lδe·δe``
    Drag:   ``C_D = C_D0 + C_Dα·α``  (parabolic polar)
    Moment: ``C_m = C_m0 + C_mα·α + C_mq·(c̄/(2V))·q + C_mδe·δe``
    Roll:   ``C_l = C_lp·(b/(2V))·p + C_lδa·δa + C_lδr·δr``
    Yaw:    ``C_n = C_nr·(b/(2V))·r + C_nδa·δa + C_nδr·δr``

    Thrust is modelled as ``T = k_t · delta_t`` (actuator-linear model).

    .. note::
        The previous model ``T = k_t · delta_t · Va²`` was **physically wrong**.
        It created a positive-velocity feedback loop (∂T/∂Va > 0) that
        destabilised the linearised Jacobian, causing the SQP/OSQP solver to
        receive a non-convex KKT matrix (OSQP error 4) on the first call.
        The correct propeller model has thrust *decreasing* with airspeed.

        **Migration**: if you were using the old model with a small ``k_thrust``
        (e.g. ``0.013``) tuned so that ``k_t · delta_t · Va² ≈ trim_thrust``,
        rescale: ``k_thrust_new = k_thrust_old × Va_cruise²``.
        Example: ``k_thrust=0.013`` at Va=15 m/s → ``k_thrust_new ≈ 3.0``.

    Default parameters approximate a 1.5-kg, 1.0-m wingspan electric trainer
    (similar to the Aerosonde UAV used in Beard & McLain).

    Reference
    ---------
    Beard, R. W., & McLain, T. W. (2012).
    *Small Unmanned Aircraft: Theory and Practice*.
    Princeton University Press. ISBN 978-0-691-14921-9.
    """

    def __init__(
        self,
        mass:         float = 1.56,      # kg
        gravity:      float = 9.807,     # m/s²
        wingspan:     float = 1.0,       # b  [m]
        chord:        float = 0.18,      # c̄  [m]
        wing_area:    float = 0.18,      # S  [m²]
        inertia:      list  = [0.082, 0.113, 0.132],  # [Ix, Iy, Iz]  kg·m²
        inertia_xz:   float = 0.0123,    # Ixz  kg·m²
        # Aerodynamic coefficients
        CL0:   float =  0.28,   CDp:   float = 0.03,   # profile drag
        CLa:   float =  3.45,   CDa:   float = 0.30,   # AoA slope
        CLq:   float =  0.0,    Cm0:   float = -0.02,
        CLde:  float =  0.36,   Cma:   float = -0.38,
        CD0:   float =  0.03,   Cmq:   float = -3.6,
        Cmde:  float = -0.5,
        # Roll damping / coupling  (B&M Table 2.2 / Appendix E notation)
        Clp:   float = -0.50,   # roll  damping    (roll   rate → roll  moment)
        Clda:  float =  0.14,   # aileron effectiveness
        Cldr:  float =  0.026,  # rudder→roll coupling
        Clb:   float = -0.13,   # dihedral effect  (sideslip β → roll moment)
        Clr:   float =  0.145,  # roll-due-to-yaw  (yaw rate r → roll moment)
        # Yaw damping / coupling
        Cnp:   float =  0.022,  # yaw-due-to-roll  (roll rate p → yaw moment)
        Cnr:   float = -0.35,   # yaw damping      (yaw  rate r → yaw moment)
        Cnda:  float =  0.006,  # adverse yaw (aileron→yaw)
        Cndr:  float = -0.032,  # rudder effectiveness
        Cnb:   float =  0.073,  # weathercock stability (β → yaw moment)
        # Side-force coefficients  (B&M Eq. 4.9)
        CYb:   float = -0.26,   # side force due to sideslip
        CYp:   float =  0.0,    # side force due to roll rate
        CYr:   float =  0.0,    # side force due to yaw rate
        CYda:  float =  0.0,    # side force due to aileron
        CYdr:  float =  0.14,   # side force due to rudder
        # ── Propulsion (B&M Eq. 4.11) ────────────────────────────────────────
        # T = 0.5 · ρ · S_prop · C_prop · [(k_motor · δt)² − Va²]
        # Stable because ∂T/∂Va = −ρ · S_prop · C_prop · Va ≤ 0
        #   (thrust naturally decreases with airspeed — correct propeller physics)
        # Defaults tuned for a 1.56 kg, 1.0 m wingspan electric trainer at Va=15 m/s:
        #   trim at ~35% throttle, T/W ≈ 1.35 at full throttle.
        S_prop:   float = 0.0120,  # propeller disc area [m²]  (≈12.4 cm diameter)
        C_prop:   float = 1.0,     # prop efficiency coefficient (dimensionless)
        k_motor:  float = 55.0,    # motor speed constant [m/s]: T→0 when Va = k_motor·δt
        # ── Stall model ───────────────────────────────────────────────────────
        M_sigmoid: float = 50.0,   # sigmoid blending half-width [rad]
        alpha0:    float = 0.4712, # stall transition centre [rad]
    ) -> None:

        constants = {k: v for k, v in locals().items() if k != "self"}
        super().__init__(
            constants=constants,
            state_derivative_orders=[1, 1, 1] + [0, 0, 0] + [0, 0, 0],
            control_derivative_orders=[0] * 4,
            wrapable_primitive_state=[3, 4, 5],
            jax_backend=True,
        )

        self.m     = float(mass)
        self.g     = float(gravity)
        self.b     = float(wingspan)
        self.c     = float(chord)
        self.S     = float(wing_area)
        self.Ix, self.Iy, self.Iz = [float(v) for v in inertia]
        self.Ixz   = float(inertia_xz)

        # ── Γ terms (B&M Eqs 4.4–4.7, pp. 36–37) ─────────────────────────────
        # B&M uses: Jx = roll  (fwd-axis) inertia,
        #           Jy = pitch (rt-axis)  inertia,
        #           Jz = yaw   (vert)     inertia,
        #           Jxz = fwd–vert product of inertia.
        # In Larp frame (X right, Y forward, Z up):
        #   Jx_BM  = Iy_LARP  (roll  = rotation about Y_fwd)
        #   Jy_BM  = Ix_LARP  (pitch = rotation about X_right)
        #   Jz_BM  = Iz_LARP  (yaw   = rotation about Z_up)
        #   Jxz_BM = Ixz_LARP (fwd–up coupling; small for symmetric airframes)
        Jx  = self.Iy   # B&M roll  inertia → Larp Iy
        Jy  = self.Ix   # B&M pitch inertia → Larp Ix
        Jz  = self.Iz   # B&M yaw   inertia → Larp Iz
        Jxz = self.Ixz

        G  = Jx * Jz - Jxz**2          # Γ denominator  (B&M p.36)
        self._G  = G
        self._g1 =  Jxz * (Jx - Jy + Jz) / G   # Γ₁
        self._g2 =  (Jz * (Jz - Jy) + Jxz**2) / G  # Γ₂
        self._g3 =  Jz / G                       # Γ₃
        self._g4 =  Jxz / G                      # Γ₄
        self._g5 =  (Jz - Jx) / Jy              # Γ₅
        self._g6 =  Jxz / Jy                    # Γ₆
        self._g7 =  (Jx * (Jx - Jy) + Jxz**2) / G  # Γ₇
        self._g8 =  Jx / G                       # Γ₈

        # Aero coefficients
        self.CL0   = CL0;   self.CLa  = CLa;  self.CLq  = CLq;  self.CLde = CLde
        self.CD0   = CD0;   self.CDp  = CDp;  self.CDa  = CDa
        self.Cm0   = Cm0;   self.Cma  = Cma;  self.Cmq  = Cmq;  self.Cmde = Cmde
        self.Clp   = Clp;   self.Clda = Clda; self.Cldr = Cldr
        self.Clb   = Clb;   self.Clr  = Clr
        self.Cnp   = Cnp;   self.Cnr  = Cnr;  self.Cnda = Cnda; self.Cndr = Cndr
        self.Cnb   = Cnb
        self.CYb   = CYb;   self.CYp  = CYp;  self.CYr  = CYr
        self.CYda  = CYda;  self.CYdr = CYdr

        # Propulsion (B&M)
        self.S_prop  = float(S_prop)
        self.C_prop  = float(C_prop)
        self.k_motor = float(k_motor)

        self.M_s   = float(M_sigmoid)
        self.a0    = float(alpha0)

    # ── Aero helpers ──────────────────────────────────────────────────────────

    def _sigmoid(self, alpha, np):
        """Sigmoid blend factor for post-stall CL (Beard & McLain Eq. 4.10)."""
        ea = np.exp(-self.M_s * (alpha - self.a0))   # large for pre-stall (α < α₀)
        eb = np.exp( self.M_s * (alpha + self.a0))   # large for all α > 0
        return (1.0 + ea + eb) / ((1.0 + ea) * (1.0 + eb))

    def _rotation_matrix(self, phi, theta, psi, np):
        """ZYX rotation matrix R_bw (body→world columns)."""
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi),   np.sin(psi)
        # Each entry shape: (B, 1)
        R = [[ct*cs,                 ct*ss,                -st    ],
             [sp*st*cs - cp*ss,  sp*st*ss + cp*cs,  sp*ct  ],
             [cp*st*cs + sp*ss,  cp*st*ss - sp*cs,  cp*ct  ]]
        return R  # list of lists of (B,1) arrays

    def f(self, first_order_state: np.ndarray,
          first_order_control: np.ndarray,
          np=None) -> np.ndarray:
        r"""
        Continuous-time state derivative :math:`\dot{x} = f(x, u)`.

        Follows Beard & McLain (2012) *Small Unmanned Aircraft*, Chapters 3–4,
        with sign corrections for the Larp body-frame convention
        (X right, Y forward, Z up) versus B&M NED (X forward, Y right, Z down).

        Larp ↔ B&M variable mapping
        ----------------------------
        ============  =============================  ==============
        Larp symbol   Physical meaning               B&M symbol
        ============  =============================  ==============
        phi  (idx 6)  Pitch angle  (about X_right)   θ  (theta)
        theta(idx 7)  Roll  angle  (about Y_fwd)     ϕ  (phi)
        psi  (idx 8)  Yaw   angle  (about Z_up)      ψ  (same sign)
        p    (idx 9)  Pitch rate   (about X_right)   q  (pitch rate)
        q    (idx 10) Roll  rate   (about Y_fwd)     p  (roll  rate)
        r    (idx 11) Yaw   rate   (about Z_up)      −r (sign flip)
        ============  =============================  ==============

        Thrust follows B&M Eq. 4.11:
        ``T = 0.5·ρ·Sp·Cp·[(k_motor·δt)² − Va²]``, clamped ≥ 0
        (∂T/∂Va ≤ 0 → stable; thrust naturally decreases with airspeed).
        """
        if np is None: np = importlib.import_module("numpy")

        # ── 1. Unpack state ───────────────────────────────────────────────────
        # [x, vx,  y, vy,  z, vz,  phi, theta, psi,  p,  q,  r]
        #  0   1   2   3   4   5    6     7      8    9   10  11
        vx, vy, vz      = first_order_state[:, 1:2], first_order_state[:, 3:4], first_order_state[:, 5:6]
        phi, theta, psi = first_order_state[:, 6:7], first_order_state[:, 7:8], first_order_state[:, 8:9]
        p, q, r_        = first_order_state[:, 9:10], first_order_state[:, 10:11], first_order_state[:, 11:12]

        # ── 2. Unpack control ─────────────────────────────────────────────────
        delta_t = np.clip(first_order_control[:, 0:1], 0.0, 1.0)
        delta_a = first_order_control[:, 1:2]
        delta_e = first_order_control[:, 2:3]
        delta_r = first_order_control[:, 3:4]

        # ── 3. Trig shortcuts ─────────────────────────────────────────────────
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi),   np.sin(psi)

        # ── 4. World → body velocity  (R_wb = R_bw^T, ZYX Euler) ─────────────
        u_b = ( cs*ct)*vx + ( ss*ct)*vy + (-st    )*vz                         # lateral  (body X)
        v_b = (cs*sp*st - ss*cp)*vx + (ss*sp*st + cs*cp)*vy + (sp*ct)*vz       # forward  (body Y)
        w_b = (cs*cp*st + ss*sp)*vx + (ss*cp*st - cs*sp)*vy + (cp*ct)*vz       # upward   (body Z)

        # ── 5. Airspeed & aerodynamic angles (B&M §2.3) ──────────────────────
        Va2 = np.maximum(vx**2 + vy**2 + vz**2, 0.01)
        Va  = np.sqrt(Va2)
        alpha = np.arctan2(-w_b, np.maximum(v_b, 1e-4))   # saturates at stall
        beta  = np.arcsin(np.clip(u_b / Va, -1.0, 1.0))

        # ── 6. Dynamic pressure scalars ───────────────────────────────────────
        rho   = 1.225
        q_dyn = 0.5 * rho * Va2 * self.S
        b_2V  = self.b / (2.0 * Va)
        c_2V  = self.c / (2.0 * Va)

        # ── 7. Force coefficients (B&M Eqs 4.7–4.9) ──────────────────────────

        # Lift  CL  (post-stall sigmoid blend, B&M Eq. 4.10)
        sig      = self._sigmoid(alpha, np)
        CL_lin   = self.CL0 + self.CLa * alpha
        CL_stall = 2.0 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha)
        CL = ((1.0 - sig) * CL_lin + sig * CL_stall
              + self.CLq * c_2V * p    # CLq uses q_BM = p_LARP
              + self.CLde * delta_e)

        # Drag  CD  (parabolic polar, B&M Eq. 4.8)
        CD = (self.CDp
              + (self.CL0 + self.CLa * alpha)**2 / (np.pi * 0.85 * (self.b**2 / self.S))
              + self.CDa * np.abs(alpha))

        # Side force  CY  (B&M Eq. 4.9)
        # CYp: p_BM = q_LARP;  CYr: r_BM = −r_LARP → sign flip
        CY = (self.CYb  * beta
              + self.CYp  * b_2V * q
              - self.CYr  * b_2V * r_
              + self.CYda * delta_a
              + self.CYdr * delta_r)

        # ── 8. Moment coefficients (B&M Eqs 4.3–4.6) ─────────────────────────
        # Sign convention for Larp moments vs B&M:
        #   Mx_b = m_BM  (pitch, about Y_right = X_LARP)     → no flip
        #   My_b = l_BM  (roll,  about X_fwd   = Y_LARP)     → no flip
        #   Mz_b = −n_BM (yaw,   Z_up  ≠ Z_down)             → overall sign flip

        # Pitching moment  Mx_b = m_BM  (B&M Eq. 4.5)
        # Cmq uses q_BM = p_LARP
        Mx_b = q_dyn * self.c * (self.Cm0
                                  + self.Cma  * alpha
                                  + self.Cmq  * c_2V * p
                                  + self.Cmde * delta_e)

        # Rolling moment  My_b = l_BM  (B&M Eq. 4.4)
        # Clp uses p_BM = q_LARP;  Clr uses r_BM = −r_LARP (→ sign flip on Clr term)
        # Clb = B&M Clβ (dihedral effect: β → roll)
        My_b = q_dyn * self.b * (self.Clb  * beta
                                  + self.Clp  * b_2V * q
                                  - self.Clr  * b_2V * r_   # r_LARP = −r_BM
                                  + self.Clda * delta_a
                                  + self.Cldr * delta_r)

        # Yawing moment  Mz_b = −n_BM  (B&M Eq. 4.6, negated)
        # Cnb = B&M Cnβ (weathercock: β → yaw);  Cnp uses p_BM = q_LARP
        # After negation: −Cnb·β, −Cnp·q, +Cnr·r_ (double flip), −Cnda, −Cndr
        Mz_b = q_dyn * self.b * (-self.Cnb  * beta
                                   - self.Cnp  * b_2V * q
                                   + self.Cnr  * b_2V * r_
                                   - self.Cnda * delta_a
                                   - self.Cndr * delta_r)

        # ── 9. Propulsive thrust (B&M Eq. 4.11) ──────────────────────────────
        # T = 0.5·ρ·Sp·Cp·[(k_motor·δt)² − Va²]  (clamped; no thrust reversal)
        T_prop = np.maximum(
            0.5 * rho * self.S_prop * self.C_prop * ((self.k_motor * delta_t)**2 - Va2),
            0.0,
        )

        # ── 10. Body forces ───────────────────────────────────────────────────
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        Fx_b = CY * q_dyn                                          # lateral  (body X)
        Fy_b = T_prop - cos_a * CD * q_dyn + sin_a * CL * q_dyn   # forward  (body Y)
        Fz_b =          sin_a * CD * q_dyn + cos_a * CL * q_dyn   # upward   (body Z, Z-up → both +)

        # ── 11. World accelerations  (R_bw : body → world) ───────────────────
        ax_ = ((cs*ct)*Fx_b + (cs*sp*st - ss*cp)*Fy_b + (cs*cp*st + ss*sp)*Fz_b) / self.m
        ay_ = ((ss*ct)*Fx_b + (ss*sp*st + cs*cp)*Fy_b + (ss*cp*st - cs*sp)*Fz_b) / self.m
        az_ = ((-st  )*Fx_b + (sp*ct          )*Fy_b + (cp*ct            )*Fz_b) / self.m - self.g

        # ── 12. Rotational dynamics — B&M Γ equations (Eqs 4.4–4.7) ──────────
        #
        # B&M uses (p_b=roll, q_b=pitch, r_b=yaw).
        # Larp substitution: p_b=q, q_b=p, r_b=−r_;  l_b=My_b, m_b=Mx_b, n_b=−Mz_b.
        # Jy_BM = Ix_LARP (pitch inertia about right axis).
        #
        # B&M Eq. 4.4  ṗ_b → dq/dt  (roll rate in Larp)
        dq = ( self._g1 * q * p            # Γ₁·p_b·q_b = Γ₁·q·p  (B&M p_b→q, q_b→p)
             + self._g2 * p * r_          # −Γ₂·q_b·r_b = +Γ₂·p·r_
             + self._g3 * My_b            #  Γ₃·l_b
             - self._g4 * Mz_b)           #  Γ₄·n_b = −Γ₄·Mz_b

        # B&M Eq. 4.5  q̇_b → dp/dt  (pitch rate in Larp)
        dp = (-self._g5 * q * r_          # Γ₅·p_b·r_b = −Γ₅·q·r_
             - self._g6 * (q**2 - r_**2)  # −Γ₆·(p_b²−r_b²) = −Γ₆·(q²−r_²)
             + Mx_b / self.Ix)            # m_b/Jy_BM = Mx_b/Ix_LARP

        # B&M Eq. 4.7  ṙ_b → dr/dt  (yaw rate in Larp, with negation ṙ_b=−dr/dt)
        dr = (-self._g7 * q * p           # −Γ₇·p_b·q_b
             - self._g1 * p * r_          # −(−Γ₁·q_b·r_b) = −Γ₁·p·r_
             - self._g4 * My_b            # −Γ₄·l_b
             + self._g8 * Mz_b)           # −Γ₈·n_b = +Γ₈·Mz_b

        # ── 13. Euler angle kinematics (ZYX: Rz·Ry·Rx) ───────────────────────
        theta_c = np.clip(theta, -1.5, 1.5)
        tt   = np.tan(theta_c)
        sect = 1.0 / np.cos(theta_c)

        dphi   = p + (q * sp + r_ * cp) * tt
        dtheta = q * cp - r_ * sp
        dpsi   = (q * sp + r_ * cp) * sect

        return np.concatenate([
            vx, ax_,
            vy, ay_,
            vz, az_,
            dphi, dtheta, dpsi,
            dp, dq, dr,
        ], axis=1)

    @property
    def angle_indices(self):
        return [6, 7, 8]

    @property
    def heading_convention_offset(self) -> float:
        """psi=0 → nose +Y (same convention as QuadcopterDynamics)."""
        return -np.pi / 2

class eVTOL41Dynamics(Dynamics):
    r"""
    6-DOF hybrid eVTOL: 4 lift rotors + 1 pusher propeller + fixed-wing surfaces.

    Designed for urban air mobility and cargo delivery. Transitions naturally between
    hover (lift-rotors dominant) and cruise (wing + pusher dominant) without explicit
    mode switching.

    State vector (12-dim): ``[x, vx, y, vy, z, vz, phi, theta, psi, p, q, r]``

    Same layout as ``QuadcopterDynamics`` and ``SmallFixedWingDynamics``.

    Control vector (8-dim): ``[w1, w2, w3, w4, delta_t, delta_a, delta_e, delta_r]``

    ============  =============================================
    ``w1..w4``    Lift rotor angular speeds [rad/s]
    ``delta_t``   Pusher throttle ∈ [0, 1]
    ``delta_a``   Aileron deflection [rad]
    ``delta_e``   Elevator deflection [rad]
    ``delta_r``   Rudder deflection [rad]
    ============  =============================================

    Lift-rotor layout (Larp frame: X right, Y forward, Z up)::

        Motor 1: Front-Right  (+bx, +fy)  CW
        Motor 2: Rear-Right   (+bx, −ry)  CCW
        Motor 3: Rear-Left    (−bx, −ry)  CW
        Motor 4: Front-Left   (−bx, +fy)  CCW

    **Pusher thrust model** (B&M Eq. 4.11)::

        T_p = 0.5 * rho * S_prop * C_prop * ((k_motor * delta_t)^2 − Va^2)

    where ``k_motor * delta_t`` is the effective motor velocity parameter [m/s].
    See :meth:`pusher_speed` for the forward mapping from throttle to this parameter.

    **Aerodynamic model**: Simplified stability-axis lift/drag (B&M §4).  Wing forces
    are blended to zero below ``_Va_blend`` (default 8 m/s) so the model is well-posed
    during hover and near-vertical flight.

    References
    ----------
    Beard, R. W., & McLain, T. W. (2012). *Small Unmanned Aircraft: Theory and
    Practice*. Princeton University Press.
    """

    def __init__(
        self,
        mass:           float = 22.0,    # kg  (compact urban cargo drone)
        gravity:        float = 9.807,
        wingspan:       float = 3.2,     # b [m]  high-AR wing for cruise efficiency
        chord:          float = 0.28,    # c̄ [m]
        wing_area:      float = 0.90,    # S [m²]
        inertia:        list  = [0.85, 1.20, 1.80],   # [Ix, Iy, Iz] kg·m²
        # Rotor geometry
        rotor_boom_x:   float = 1.10,    # bx: lateral boom offset from centreline [m]
        rotor_front_y:  float = 0.55,    # fy: front-motor fore-aft offset [m]
        rotor_rear_y:   float = 0.45,    # ry: rear-motor fore-aft offset [m]
        thrust_constant: float = 2.5e-4, # kf  [N/(rad/s)²]
        torque_constant: float = 4.5e-6, # km  [Nm/(rad/s)²]
        motor_inertia:  float = 8e-4,    # Ir  [kg·m²]
        translational_drag: list = [0.6, 0.6, 1.0],  # kt
        rotational_drag:    list = [0.3, 0.3, 0.4],  # kr
        # Aerodynamic coefficients (simplified B&M)
        CL0:  float =  0.20,   CLa:  float =  4.2,   CLde: float =  0.4,
        CD0:  float =  0.025,  CDa:  float =  0.22,  CDp:  float =  0.028,
        Cm0:  float = -0.01,   Cma:  float = -0.45,  Cmde: float = -0.55,  Cmq: float = -3.8,
        Clp:  float = -0.48,   Clda: float =  0.16,  Clb:  float = -0.12,
        Cnr:  float = -0.32,   Cndr: float = -0.028, Cnb:  float =  0.065,
        CYb:  float = -0.24,   CYdr: float =  0.12,
        # Pusher propulsion (B&M Eq. 4.11 form)
        S_prop:  float = 0.018,
        C_prop:  float = 1.0,
        k_motor: float = 48.0,
    ) -> None:

        constants = {k: v for k, v in locals().items() if k != 'self'}
        super().__init__(
            constants=constants,
            state_derivative_orders=[1, 1, 1] + [0, 0, 0] + [0, 0, 0],
            control_derivative_orders=[0] * 8,
            wrapable_primitive_state=[3, 4, 5],
            jax_backend=True,
        )

        self.m  = float(mass);   self.g  = float(gravity)
        self.b  = float(wingspan); self.c = float(chord); self.S = float(wing_area)
        self.Ix, self.Iy, self.Iz = [float(v) for v in inertia]
        self.bx = float(rotor_boom_x)
        self.fy = float(rotor_front_y)
        self.ry = float(rotor_rear_y)
        # Convenience aliases kept for hover_speed() and general use
        self.lx = self.bx; self.ly = (self.fy + self.ry) / 2.0
        self.kf = float(thrust_constant); self.km = float(torque_constant); self.Ir = float(motor_inertia)
        self.kt = np.array(translational_drag); self.kr = np.array(rotational_drag)

        # Aero
        self.CL0=CL0; self.CLa=CLa; self.CLde=CLde
        self.CD0=CD0; self.CDa=CDa; self.CDp=CDp
        self.Cm0=Cm0; self.Cma=Cma; self.Cmde=Cmde; self.Cmq=Cmq
        self.Clp=Clp; self.Clda=Clda; self.Clb=Clb
        self.Cnr=Cnr; self.Cndr=Cndr; self.Cnb=Cnb
        self.CYb=CYb; self.CYdr=CYdr
        self.S_prop=float(S_prop); self.C_prop=float(C_prop); self.k_motor=float(k_motor)

        # Γ terms (B&M Eqs 4.4-4.7) — same frame mapping as SmallFixedWingDynamics
        Jx  = self.Iy; Jy  = self.Ix; Jz  = self.Iz; Jxz = 0.0
        G   = Jx * Jz
        self._g3 = Jz / G;   self._g4 = 0.0
        self._g5 = (Jz - Jx) / Jy
        self._g8 = Jx / G
        self._Jy = Jy

        # Twin-boom rotor layout:
        #   1: Front-Right (+bx,+fy) CW    2: Rear-Right (+bx,−ry) CCW
        #   3: Rear-Left   (−bx,−ry) CW    4: Front-Left (−bx,+fy) CCW
        self.motor_pos = np.array([
            [+self.bx, +self.fy],
            [+self.bx, -self.ry],
            [-self.bx, -self.ry],
            [-self.bx, +self.fy],
        ])
        self.yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])

    # ── helpers ───────────────────────────────────────────────────────────────

    def hover_speed(self) -> float:
        """
        Average lift-rotor speed for level hover (equal-speed assumption).

        Returns
        -------
        float
            Rotor angular speed [rad/s] such that all four rotors together
            produce thrust equal to vehicle weight.  When ``fy != ry`` the
            front and rear motors must spin at different speeds for a trimmed
            hover; use :meth:`hover_speeds` in that case.
        """
        return float(np.sqrt((self.m * self.g) / (4.0 * self.kf)))

    def hover_speeds(self) -> Tuple[float, float]:
        """
        Trimmed front and rear lift-rotor speeds for pitch-moment-free hover.

        When ``fy != ry`` the front and rear motor pairs must spin at different
        speeds to zero the net pitch moment.  The balance condition is::

            2·kf·w_front²·fy = 2·kf·w_rear²·ry  →  w_front/w_rear = √(ry/fy)
            2·kf·(w_front² + w_rear²) = m·g

        Motor command order: ``[w_front, w_rear, w_rear, w_front]`` (motors 1, 2, 3, 4).

        Returns
        -------
        w_front : float
            Angular speed [rad/s] for motors 1 and 4 (front pair).
        w_rear : float
            Angular speed [rad/s] for motors 2 and 3 (rear pair).
        """
        mg = self.m * self.g; fy = self.fy; ry = self.ry
        w_front = float(np.sqrt(mg * ry / (2.0 * self.kf * (fy + ry))))
        w_rear  = float(np.sqrt(mg * fy / (2.0 * self.kf * (fy + ry))))
        return w_front, w_rear

    def pusher_speed(self, delta_t: float, Va: float = 0.0) -> Tuple[float, float]:
        """
        Map pusher throttle to the B&M motor velocity parameter and net thrust.

        The pusher propulsion model follows B&M Eq. 4.11::

            T_p = 0.5 · rho · S_prop · C_prop · (V_motor² − Va²)

        where the **motor velocity parameter** ``V_motor = k_motor · delta_t`` [m/s]
        acts as an effective induced-velocity ceiling at a given throttle setting.
        This is *not* a physical shaft angular speed; it is the velocity equivalent
        used to match static-thrust data.  To obtain an approximate shaft speed,
        divide by the propeller tip radius.

        Parameters
        ----------
        delta_t : float
            Pusher throttle command, clipped to [0, 1].
        Va : float, optional
            Airspeed magnitude [m/s].  Affects thrust but not ``V_motor``.

        Returns
        -------
        V_motor : float
            Effective motor velocity parameter ``k_motor · delta_t`` [m/s].
        T_push : float
            Net pusher thrust [N] at the given airspeed.  Zero in the
            windmill regime (``Va >= V_motor``).

        Examples
        --------
        >>> dyn = eVTOL41Dynamics()
        >>> V_motor, T = dyn.pusher_speed(delta_t=1.0, Va=0.0)
        >>> print(f"V_motor={V_motor:.1f} m/s, T={T:.1f} N")
        V_motor=48.0 m/s, T=25.4 N
        """
        delta_t = float(np.clip(delta_t, 0.0, 1.0))
        V_motor = self.k_motor * delta_t
        rho = 1.225
        T_push = float(max(0.0, 0.5 * rho * self.S_prop * self.C_prop * (V_motor**2 - Va**2)))
        return V_motor, T_push

    @property
    def angle_indices(self) -> List[int]:
        """Indices of angle states (phi, theta, psi) in the flattened first-order state."""
        return [6, 7, 8]

    @property
    def heading_convention_offset(self) -> float:
        """Offset so that psi=0 aligns with the +Y (forward) world axis."""
        return -np.pi / 2

    # ── dynamics ──────────────────────────────────────────────────────────────

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray,
          np: Optional[ModuleType] = None) -> np.ndarray:
        """
        Continuous-time dynamics: state derivative ``x_dot = f(x, u)``.

        Parameters
        ----------
        first_order_state : ndarray of shape (B, 12)
            Batched state ``[x, vx, y, vy, z, vz, phi, theta, psi, p, q, r]``.
        first_order_control : ndarray of shape (B, 8)
            Batched control ``[w1, w2, w3, w4, delta_t, delta_a, delta_e, delta_r]``.
        np : module, optional
            Numerical backend.  Pass ``jnp`` when tracing under JAX; defaults
            to NumPy.

        Returns
        -------
        ndarray of shape (B, 12)
            Time derivative of the state in the same order as the input.
        """
        if np is None: np = importlib.import_module("numpy")

        # ── 1. Unpack ─────────────────────────────────────────────────────────
        vel   = first_order_state[:, 1:6:2]       # [vx, vy, vz]  world frame
        phi   = first_order_state[:, 6:7]
        theta = first_order_state[:, 7:8]
        psi   = first_order_state[:, 8:9]
        p, q, r_ = (first_order_state[:, 9:10],
                    first_order_state[:, 10:11],
                    first_order_state[:, 11:12])
        vx, vy, vz = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]

        w   = first_order_control[:, 0:4]          # lift rotor speeds
        delta_t = np.clip(first_order_control[:, 4:5], 0.0, 1.0)
        delta_a = first_order_control[:, 5:6]
        delta_e = first_order_control[:, 6:7]
        delta_r = first_order_control[:, 7:8]

        # ── 2. Trig ───────────────────────────────────────────────────────────
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cs, ss = np.cos(psi), np.sin(psi)

        # ── 3. Rotation matrix columns (body→world) ───────────────────────────
        # R[:,0]=X_w, R[:,1]=Y_w, R[:,2]=Z_w  (each (B,1))
        Rx = np.concatenate([cs*ct, ss*ct, -st], axis=1)
        Ry = np.concatenate([cs*sp*st - ss*cp, ss*sp*st + cs*cp, sp*ct], axis=1)
        Rz = np.concatenate([cs*cp*st + ss*sp, ss*cp*st - cs*sp, cp*ct], axis=1)

        # ── 4. World→body velocity ────────────────────────────────────────────
        v_world = np.concatenate([vx, vy, vz], axis=1)   # (B,3)
        u_b = (Rx * v_world).sum(axis=1, keepdims=True)   # lateral  (body X)
        v_b = (Ry * v_world).sum(axis=1, keepdims=True)   # forward  (body Y)
        w_b = (Rz * v_world).sum(axis=1, keepdims=True)   # upward   (body Z)

        # ── 5. Aerodynamics (B&M, Z-up frame) ────────────────────────────────
        Va2   = np.maximum(vx**2 + vy**2 + vz**2, 0.01); Va = np.sqrt(Va2)
        alpha = np.arctan2(-w_b, np.maximum(v_b, 1e-4))   # Z-up sign correction
        beta  = np.arcsin(np.clip(u_b / Va, -1.0, 1.0))

        rho   = 1.225
        q_dyn = 0.5 * rho * Va2 * self.S
        b_2V  = self.b / (2.0 * Va); c_2V = self.c / (2.0 * Va)

        # Aerodynamic blend: suppress wing aero at low airspeed.
        # At Va < Va_blend (~8 m/s) wings produce negligible force; alpha is also
        # ill-defined during near-vertical flight.  Fade in quadratically so the
        # cruise aero is unaffected above Va_blend.
        _vab  = getattr(self, '_Va_blend', 8.0)
        aero_blend = np.clip(Va2 / (_vab ** 2), 0.0, 1.0)

        CL = self.CL0 + self.CLa * alpha + self.CLde * delta_e
        CD = self.CDp + (self.CL0 + self.CLa * alpha)**2 / (np.pi * 0.85 * (self.b**2 / self.S)) + self.CDa * np.abs(alpha)
        CY = self.CYb * beta + self.CYdr * delta_r

        # Body forces from aerodynamics (blended to zero at low Va)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        Fx_aero = aero_blend * CY * q_dyn
        Fy_aero = aero_blend * (-cos_a * CD * q_dyn + sin_a * CL * q_dyn)
        Fz_aero = aero_blend * ( sin_a * CD * q_dyn + cos_a * CL * q_dyn)

        # Aerodynamic moments (blended to zero at low Va)
        Mx_aero = aero_blend * q_dyn * self.c * (self.Cm0 + self.Cma * alpha + self.Cmq * c_2V * p + self.Cmde * delta_e)
        My_aero = aero_blend * q_dyn * self.b * (self.Clb * beta + self.Clp * b_2V * q + self.Clda * delta_a)
        Mz_aero = aero_blend * q_dyn * self.b * (-self.Cnb * beta + self.Cnr * b_2V * r_ + self.Cndr * delta_r)

        # ── 6. Lift rotors ────────────────────────────────────────────────────
        w2 = w ** 2
        F_lift_body = self.kf * np.sum(w2, axis=1, keepdims=True)        # body Z

        # Pitch moment: sum(kf * ry_i * wi^2)  (ry = ly)
        ry = np.array([[self.motor_pos[i, 1] for i in range(4)]])
        # Roll moment: sum(kf * (-rx_i) * wi^2)
        rx = np.array([[self.motor_pos[i, 0] for i in range(4)]])
        Mx_rot = self.kf * np.sum(ry * w2, axis=1, keepdims=True)
        My_rot = self.kf * np.sum(-rx * w2, axis=1, keepdims=True)
        Mz_rot = self.km * np.sum(self.yaw_signs * w2, axis=1, keepdims=True)
        omega_r = np.sum(self.yaw_signs * w, axis=1, keepdims=True)

        # ── 7. Pusher thrust (B&M Eq. 4.11, body Y direction) ────────────────
        T_push = np.maximum(0.5 * rho * self.S_prop * self.C_prop *
                            ((self.k_motor * delta_t)**2 - Va2), 0.0)

        # ── 8. Total body forces ──────────────────────────────────────────────
        Fx_b_total = Fx_aero
        Fy_b_total = Fy_aero + T_push
        Fz_b_total = Fz_aero + F_lift_body

        # ── 9. World accelerations ────────────────────────────────────────────
        drag_w = self.kt * vel
        ax_ = ((Rx[:, 0:1]*Fx_b_total + Ry[:, 0:1]*Fy_b_total + Rz[:, 0:1]*Fz_b_total)
               / self.m - drag_w[:, 0:1])
        ay_ = ((Rx[:, 1:2]*Fx_b_total + Ry[:, 1:2]*Fy_b_total + Rz[:, 1:2]*Fz_b_total)
               / self.m - drag_w[:, 1:2])
        az_ = ((Rx[:, 2:3]*Fx_b_total + Ry[:, 2:3]*Fy_b_total + Rz[:, 2:3]*Fz_b_total)
               / self.m - drag_w[:, 2:3] - self.g)

        # ── 10. Rotational dynamics ───────────────────────────────────────────
        tau_x = Mx_rot + Mx_aero   # pitch  (Larp Ix = B&M Jy)
        tau_y = My_rot + My_aero   # roll   (Larp Iy = B&M Jx)
        tau_z = Mz_rot + Mz_aero   # yaw

        drag_rot = self.kr * np.concatenate([p, q, r_], axis=1)
        gyro_p   = self.Ir * q  * omega_r
        gyro_q   = -self.Ir * p * omega_r

        # Standard Euler equations: I*dΩ/dt = τ − Ω×(IΩ)
        # Ω×(IΩ) = [(Iz−Iy)qr, (Ix−Iz)pr, (Iy−Ix)pq] → subtract from τ
        cp_p = (self.Iy - self.Iz) * q  * r_   # (Iy−Iz)·q·r for dp equation
        cp_q = (self.Iz - self.Ix) * p  * r_   # (Iz−Ix)·p·r for dq equation
        cp_r = (self.Ix - self.Iy) * p  * q    # (Ix−Iy)·p·q for dr equation

        dp = (tau_x + cp_p - gyro_p - drag_rot[:, 0:1]) / self.Ix   # pitch rate
        dq = (tau_y + cp_q - gyro_q - drag_rot[:, 1:2]) / self.Iy   # roll rate
        dr = (tau_z + cp_r           - drag_rot[:, 2:3]) / self.Iz   # yaw rate

        # ── 11. Euler kinematics ──────────────────────────────────────────────
        theta_c = np.clip(theta, -1.5, 1.5)
        tt   = np.tan(theta_c); sect = 1.0 / np.cos(theta_c)
        dphi   = p + (q * sp + r_ * cp) * tt
        dtheta = q * cp - r_ * sp
        dpsi   = (q * sp + r_ * cp) * sect

        return np.concatenate([vx, ax_, vy, ay_, vz, az_, dphi, dtheta, dpsi, dp, dq, dr], axis=1)

class MJXDynamics(Dynamics):
    r"""
    Rigid-body dynamics engine backed by MuJoCo XLA (mjx).

    Loads arbitrary MJCF/XML models and exposes a fully differentiable, batched
    dynamics environment.  Jacobians and Hessians are computed via JAX
    auto-differentiation; rollouts use the high-fidelity ``mjx.step`` integrator.

    State layout: ``x = [q_pos (nq,), q_vel (nv,)]``.  Total dimension is
    ``nq + nv``.  For models without free joints ``nq == nv``; a single free
    joint adds one extra quaternion component (``nq == nv + 1``).

    Control layout: the raw control vector ``u`` is passed through
    :meth:`control_to_nv` before being routed to MuJoCo.  Override that method
    to implement underactuated or custom transmission mappings.

    .. note::
       Subclass and override :meth:`control_to_nv` (and optionally
       :meth:`nv_to_control`) to adapt this class to a specific vehicle.
       After overriding, call ``_setup_jax_functions()`` again so the JIT
       closures pick up the new mapping.

    Attributes
    ----------
    mj_model : mujoco.MjModel
        CPU-side MuJoCo model (authoritative for XML metadata).
    mjx_model : mjx.Model
        JAX-compiled model used for all numerical computations.
    nq : int
        Number of generalized position coordinates.
    nv : int
        Number of generalized velocity coordinates (degrees of freedom).
    nu : int
        Number of actuators defined in the MJCF.

    Examples
    --------
    >>> dyn = MJXDynamics("assets/quadrotor.xml")
    >>> x0 = np.zeros((16, dyn.nq + dyn.nv))   # batch of 16
    >>> us = np.zeros((16, 50, dyn.nu))          # 50-step control sequence
    >>> xs = dyn.rollout(x0, us, dt=0.01)        # shape (16, 50, nq+nv)

    See Also
    --------
    eVTOL41Dynamics : Analytic hybrid eVTOL model.
    QuadcopterDynamics : Lightweight quadrotor model.
    """

    def __init__(self, model_path: str) -> None:
        """
        Load an MJCF model from disk and initialise the dynamics engine.

        Parameters
        ----------
        model_path : str
            Path to a MuJoCo XML (MJCF) file.

        Raises
        ------
        RuntimeError
            If JAX or MuJoCo is not installed.
        """
        if not MJX_INSTALLED:
            raise RuntimeError(
                "MJX is not installed. "
                "Install JAX and MuJoCo to enable MJXDynamics."
            )

        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        super().__init__(
            constants={"model_path": model_path},
            state_derivative_orders=[0] * (self.nq + self.nv),
            control_derivative_orders=[0] * self.nu,
            holonomic=False,
            jax_backend=True,
        )

    @classmethod
    def from_xml_string(cls, xml: str) -> "MJXDynamics":
        """
        Create an ``MJXDynamics`` instance directly from an MJCF XML string.

        Parameters
        ----------
        xml : str
            Complete MuJoCo XML model string.

        Returns
        -------
        MJXDynamics
            Initialised dynamics instance.

        Examples
        --------
        >>> xml = open("assets/quadrotor.xml").read()
        >>> dyn = MJXDynamics.from_xml_string(xml)
        """
        if not MJX_INSTALLED:
            raise RuntimeError(
                "MJX is not installed. "
                "Install JAX and MuJoCo to enable MJXDynamics."
            )
        obj = object.__new__(cls)
        obj.mj_model = mujoco.MjModel.from_xml_string(xml)
        obj.mjx_model = mjx.put_model(obj.mj_model)
        obj.nq = obj.mjx_model.nq
        obj.nv = obj.mjx_model.nv
        obj.nu = obj.mjx_model.nu
        Dynamics.__init__(
            obj,
            constants={"model_path": "<xml_string>"},
            state_derivative_orders=[0] * (obj.nq + obj.nv),
            control_derivative_orders=[0] * obj.nu,
            holonomic=False,
            jax_backend=True,
        )
        return obj

    def control_to_nv(self, u: np.ndarray, np: Optional[ModuleType] = None) -> np.ndarray:
        """
        Map raw control input to MuJoCo actuator commands or generalised forces.

        The default is the identity mapping.  Override this in a subclass to
        implement non-trivial transmissions, e.g. converting rotor speeds to
        body-frame forces/torques for an underactuated vehicle.

        .. note::
           After overriding this method, call ``_setup_jax_functions()`` to
           recompile the JIT closures with the updated mapping.

        Parameters
        ----------
        u : ndarray of shape (B, nu)
            Batched raw control input.
        np : module, optional
            Numerical backend.  Pass ``jnp`` when tracing under JAX.

        Returns
        -------
        ndarray of shape (B, nu) or (B, nv)
            Actuator commands (shape ``(B, nu)``) or generalised forces
            (shape ``(B, nv)``).  The routing to ``ctrl`` vs ``qfrc_applied``
            is determined automatically by the output shape.
        """
        if np is None:
            np = jnp
        return u

    def nv_to_control(self, tau: np.ndarray, np: Optional[ModuleType] = None) -> np.ndarray:
        """
        Inverse mapping from generalised forces back to raw control input.

        Parameters
        ----------
        tau : ndarray of shape (B, nv)
            Generalised forces or actuator commands.
        np : module, optional
            Numerical backend.

        Returns
        -------
        ndarray of shape (B, nu)
            Raw control vector.

        Raises
        ------
        NotImplementedError
            Always — must be implemented by the user in a subclass.
        """
        raise NotImplementedError(
            "nv_to_control must be implemented by the subclass "
            "to match the control_to_nv mapping."
        )

    @property
    def angle_indices(self) -> List[int]:
        """Empty — MuJoCo manages its own quaternion normalisation."""
        return []

    @property
    def heading_convention_offset(self) -> float:
        """Default 0.0; override in a subclass if the model uses a different yaw convention."""
        return 0.0

    def _setup_jax_functions(self):
        """
        Build and JIT-compile the core MJX functions.

        Overrides the base-class method to install optimised ``mjx``-backed
        implementations of ``_f_jit``, ``_mjx_step_jit``, and
        ``_mjx_rollout_jit`` *before* the base class constructs its
        auto-differentiation Jacobians on top of ``self.f``.

        Free-joint handling
        -------------------
        When ``nq == nv + 1`` the model contains exactly one root free joint
        whose quaternion occupies ``qpos[3:7]``.  The position derivative for
        that quaternion is computed analytically as::

            dq = 0.5 * q ⊗ [0, ω]

        Models with multiple free joints or non-root free joints are not
        handled and will fall back to the generic zero-pad branch.
        """
        if not MJX_INSTALLED:
            return

        nq, nv, nu = self.nq, self.nv, self.nu

        def _apply_control(data, tau):
            if tau.shape[-1] == nu:
                return data.replace(ctrl=tau)
            return data.replace(qfrc_applied=tau)

        def _dqpos(qpos, qvel):
            if nq == nv:
                return qvel
            if nq == nv + 1:
                dq_lin = qvel[:3]
                wx, wy, wz = qvel[3], qvel[4], qvel[5]
                qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
                dqw = 0.5 * (-qx*wx - qy*wy - qz*wz)
                dqx = 0.5 * ( qw*wx + qy*wz - qz*wy)
                dqy = 0.5 * ( qw*wy - qx*wz + qz*wx)
                dqz = 0.5 * ( qw*wz + qx*wy - qy*wx)
                dq_quat = jnp.array([dqw, dqx, dqy, dqz])
                return jnp.concatenate([dq_lin, dq_quat, qvel[6:]])
            # fallback: zero-pad extra position slots
            return jnp.concatenate([qvel, jnp.zeros(nq - nv)])

        def _f_single(x, u):
            qpos, qvel = x[:nq], x[nq:]
            tau = self.control_to_nv(u[None, :], np=jnp)[0]
            data = mjx.make_data(self.mjx_model)
            data = _apply_control(data.replace(qpos=qpos, qvel=qvel), tau)
            data = mjx.forward(self.mjx_model, data)
            return jnp.concatenate([_dqpos(qpos, qvel), data.qacc])

        def _mjx_step_single(x, u, dt):
            model = self.mjx_model.replace(opt=self.mjx_model.opt.replace(timestep=dt))
            qpos, qvel = x[:nq], x[nq:]
            tau = self.control_to_nv(u[None, :], np=jnp)[0]
            data = mjx.make_data(model)
            data = _apply_control(data.replace(qpos=qpos, qvel=qvel), tau)
            data = mjx.step(model, data)
            return jnp.concatenate([data.qpos, data.qvel])

        def _mjx_rollout_single(x0_single, us_single, dt):
            """Scan over time axis: us_single has shape (T, nu)."""
            def scan_op(x_prev, u_curr):
                x_next = _mjx_step_single(x_prev, u_curr, dt)
                return x_next, x_next
            _, xs_traj = lax.scan(scan_op, x0_single, us_single)
            return xs_traj

        self._f_jit = jit(vmap(_f_single, in_axes=(0, 0)))
        self._mjx_step_jit = jit(vmap(_mjx_step_single, in_axes=(0, 0, None)))
        # us has shape (B, T, nu): vmap over batch axis 0 for both x0 and us.
        self._mjx_rollout_jit = jit(vmap(_mjx_rollout_single, in_axes=(0, 0, None)))

        # Build base-class Jacobians (jacfwd over self.f).
        super()._setup_jax_functions()

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray,
          np: Optional[ModuleType] = None) -> np.ndarray:
        """
        Continuous-time state derivative ``[dq_pos, dq_vel]`` via ``mjx.forward``.

        Parameters
        ----------
        first_order_state : ndarray of shape (B, nq + nv)
            Batched generalised positions and velocities.
        first_order_control : ndarray of shape (B, nu)
            Batched raw control input (passed through :meth:`control_to_nv`).
        np : module, optional
            Accepted for base-class compatibility; ``jnp`` is always used
            internally.

        Returns
        -------
        ndarray of shape (B, nq + nv)
            Time derivative of the state.
        """
        return self._f_jit(jnp.asarray(first_order_state), jnp.asarray(first_order_control))

    def step(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1,
             estimate: bool = False) -> np.ndarray:
        """
        Advance the MuJoCo system by one time step using ``mjx.step``.

        Parameters
        ----------
        x0 : ndarray of shape (B, nq + nv)
            Current state batch.
        u0 : ndarray of shape (B, nu)
            Current control batch.
        dt : float, optional
            Integration time step [s].
        estimate : bool, optional
            Ignored.  ``mjx.step`` is always used for high-fidelity integration.

        Returns
        -------
        ndarray of shape (B, nq + nv)
            Next state batch.
        """
        return np.asarray(self._mjx_step_jit(jnp.asarray(x0), jnp.asarray(u0), dt))

    def rollout(self, x0: np.ndarray, us: np.ndarray, dt: float = 0.1,
                estimate: bool = False) -> np.ndarray:
        """
        Batched forward rollout over a control sequence using ``mjx.step``.

        Parameters
        ----------
        x0 : ndarray of shape (B, nq + nv)
            Initial state batch.
        us : ndarray of shape (B, T, nu)
            Control sequence for each batch element over T time steps.
        dt : float, optional
            Integration time step [s].
        estimate : bool, optional
            Ignored.  ``mjx.step`` is always used.

        Returns
        -------
        ndarray of shape (B, T, nq + nv)
            State trajectory for each batch element.
        """
        return np.asarray(self._mjx_rollout_jit(jnp.asarray(x0), jnp.asarray(us), dt))

