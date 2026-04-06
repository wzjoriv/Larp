from abc import ABC, abstractmethod
import importlib
import inspect
from types import ModuleType
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from itertools import chain
import warnings

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

_LAMBDA_QUAT = 10.0

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
        dqw -= _LAMBDA_QUAT * q_err * qw
        dqx -= _LAMBDA_QUAT * q_err * qx
        dqy -= _LAMBDA_QUAT * q_err * qy
        dqz -= _LAMBDA_QUAT * q_err * qz

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

    # --- Static Helpers ---
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

    Thrust is modelled as ``T = k_t · delta_t · V_a²`` (propeller quadratic).

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
        # Roll / yaw
        Clp:   float = -0.50,   Cnp:   float =  0.022,
        Clda:  float =  0.14,   Cnr:   float = -0.35,
        Cldr:  float =  0.026,  Cnda:  float =  0.006,
        Cndr:  float = -0.032,
        # Thrust model:  T = k_t · delta_t · V_a²
        k_thrust: float = 1.50,
        # Stall sigmoid blending half-width (rad)
        M_sigmoid: float = 50.0,
        alpha0:    float = 0.4712,       # stall transition centre (rad)
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

        # Pre-compute inertia inverse (for body rates)
        G  = self.Ix * self.Iz - self.Ixz ** 2
        self._G   = G
        self._g1  =  self.Ixz * (self.Ix - self.Iy + self.Iz) / G
        self._g2  =  (self.Iz * (self.Iz - self.Iy) + self.Ixz ** 2) / G
        self._g3  =  self.Iz / G
        self._g4  =  self.Ixz / G
        self._g5  =  (self.Iz - self.Ix) / self.Iy
        self._g6  =  self.Ixz / self.Iy
        self._g7  =  ((self.Ix - self.Iy) * self.Ix + self.Ixz ** 2) / G
        self._g8  =  self.Ix / G

        # Aero coefficients
        self.CL0   = CL0;   self.CLa  = CLa;  self.CLq  = CLq;  self.CLde = CLde
        self.CD0   = CD0;   self.CDp  = CDp;  self.CDa  = CDa
        self.Cm0   = Cm0;   self.Cma  = Cma;  self.Cmq  = Cmq;  self.Cmde = Cmde
        self.Clp   = Clp;   self.Clda = Clda; self.Cldr = Cldr
        self.Cnp   = Cnp;   self.Cnr  = Cnr;  self.Cnda = Cnda; self.Cndr = Cndr
        self.k_t   = float(k_thrust)
        self.M_s   = float(M_sigmoid)
        self.a0    = float(alpha0)

        # Lateral coefficients (Cy_beta approximation)
        self.CYb   = -0.26
        self.CYp   = 0.0
        self.CYr   = 0.0
        self.CYda  = 0.0
        self.CYdr  = 0.14

    # ── Aero helpers ──────────────────────────────────────────────────────────

    def _sigmoid(self, alpha, np):
        """Sigmoid blend factor for post-stall CL (Beard & McLain Eq. 4.10)."""
        ea = np.exp(self.M_s * (alpha - self.a0))
        eb = np.exp(-self.M_s * (alpha + self.a0))
        return (1 + ea + eb) / ((1 + ea) * (1 + eb))

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
        Compute the continuous-time state derivative :math:`\dot{x} = f(x, u)`.

        Larp body-frame convention
        --------------------------
        Larp uses X=right, Y=forward, Z=up (ENU), which differs from the
        standard Beard & McLain (B&M) NED convention.  The mapping is:

        ===============  ===============  ========================
        Larp state       Physical meaning  B&M equivalent
        ===============  ===============  ========================
        phi   (index 6)  Pitch angle       theta_BM (about Y_right)
        theta (index 7)  Roll  angle       phi_BM   (about X_fwd)
        psi   (index 8)  Yaw   angle       psi_BM   (about Z_up, same sign)
        p     (index 9)  Pitch rate        q_BM
        q     (index 10) Roll  rate        p_BM
        r     (index 11) Yaw   rate        -r_BM  (Z_up vs Z_down)
        ===============  ===============  ========================

        Sign corrections applied versus a naïve B&M transcription:

        * **Lift (Fz_b)**: positive = upward in Z-up frame.
          ``Fz_b = +sin(alpha)*CD + cos(alpha)*CL`` (both terms positive).
        * **Side-force yaw term**: ``-CYr`` not ``+CYr`` (r_LARP = -r_BM).
        * **Yaw moment (Mz_b)**: overall sign flip relative to B&M N
          (moment about Z_up vs Z_down); also ``-Cnp`` and ``-Cnda/-Cndr``.
        """
        if np is None: np = importlib.import_module("numpy")

        # ── 1. Unpack state ───────────────────────────────────────────────
        # State: [x, vx,  y, vy,  z, vz,  phi, theta, psi,  p, q, r]
        # Idx:    0   1   2   3   4   5    6     7      8    9  10 11
        vx, vy, vz   = first_order_state[:, 1:2], first_order_state[:, 3:4], first_order_state[:, 5:6]
        phi, theta, psi = first_order_state[:, 6:7], first_order_state[:, 7:8], first_order_state[:, 8:9]
        p, q, r_ = first_order_state[:, 9:10], first_order_state[:, 10:11], first_order_state[:, 11:12]
        # phi   = pitch angle (rotation about Larp X = right axis)
        # theta = roll  angle (rotation about Larp Y = forward axis)
        # p     = pitch rate,  q = roll rate,  r_ = yaw rate (about Z_up)

        # ── 2. Unpack control ─────────────────────────────────────────────
        delta_t = np.clip(first_order_control[:, 0:1], 0.0, 1.0)  # throttle ∈ [0,1]
        delta_a = first_order_control[:, 1:2]   # aileron  [rad]
        delta_e = first_order_control[:, 2:3]   # elevator [rad]
        delta_r = first_order_control[:, 3:4]   # rudder   [rad]

        # ── 3. Trig shortcuts ─────────────────────────────────────────────
        cp, sp = np.cos(phi),   np.sin(phi)    # phi   = pitch
        ct, st = np.cos(theta), np.sin(theta)  # theta = roll
        cs, ss = np.cos(psi),   np.sin(psi)    # psi   = yaw

        # ── 4. World → body velocity  (R_wb = R_bw^T, ZYX: Rz·Ry·Rx) ───
        # body-x = right (lateral), body-y = forward, body-z = up
        u_b = ( cs*ct)*vx + ( ss*ct)*vy + (-st    )*vz   # lateral   (body X)
        v_b = (cs*sp*st - ss*cp)*vx + (ss*sp*st + cs*cp)*vy + (sp*ct)*vz  # forward   (body Y)
        w_b = (cs*cp*st + ss*sp)*vx + (ss*cp*st - cs*sp)*vy + (cp*ct)*vz  # upward    (body Z)

        # ── 5. Airspeed & aerodynamic angles ─────────────────────────────
        Va2 = np.maximum(vx**2 + vy**2 + vz**2, 0.01)  # floor before sqrt
        Va  = np.sqrt(Va2)

        # Angle of attack: angle in the forward-up (body Y-Z) plane.
        # Guard: if forward speed v_b → 0 (e.g. stall), alpha saturates to ±90°.
        alpha = np.arctan2(w_b, np.maximum(v_b, 1e-4))

        # Sideslip angle: lateral deviation.
        beta = np.arcsin(np.clip(u_b / Va, -1.0, 1.0))

        # ── 6. Aerodynamic force coefficients ────────────────────────────
        rho   = 1.225                        # air density [kg/m³]
        q_dyn = 0.5 * rho * Va2 * self.S    # dynamic pressure × wing area [N]
        b_2V  = self.b / (2.0 * Va)         # b/(2V)
        c_2V  = self.c / (2.0 * Va)         # c̄/(2V)

        # Lift coefficient (with post-stall sigmoid blend)
        sig    = self._sigmoid(alpha, np)
        CL_lin = self.CL0 + self.CLa * alpha
        CL_stall = 2.0 * np.sign(alpha) * np.sin(alpha)**2 * np.cos(alpha)
        CL = (1.0 - sig) * CL_lin + sig * CL_stall
        # p = Larp pitch rate = B&M pitch rate q → CLq uses p here
        CL = CL + self.CLq * c_2V * p + self.CLde * delta_e

        # Drag coefficient (parabolic polar)
        CD = (self.CDp
              + (self.CL0 + self.CLa * alpha)**2 / (np.pi * 0.85 * (self.b**2 / self.S))
              + self.CDa * np.abs(alpha))

        # Side-force coefficient.
        # CYp: B&M uses p_BM (roll rate) = Larp q.
        # CYr: B&M uses r_BM (yaw rate)  = -Larp r_  → sign flip on CYr term.
        CY = (self.CYb * beta
              + self.CYp * b_2V * q
              - self.CYr * b_2V * r_       # sign flip: r_LARP = -r_BM
              + self.CYdr * delta_r)

        # ── 7. Aerodynamic moments (Larp body frame) ──────────────────────
        # Mx_b (about Larp X=right) = B&M pitching moment M (about Y_NED=right).
        # Uses Larp pitch rate p (= B&M q).
        Mx_b = q_dyn * self.c * (self.Cm0
                                  + self.Cma  * alpha
                                  + self.Cmq  * c_2V * p   # p = Larp pitch rate
                                  + self.Cmde * delta_e)

        # My_b (about Larp Y=forward) = B&M rolling moment L (about X_NED=forward).
        # Uses Larp roll rate q (= B&M p).
        My_b = q_dyn * self.b * (self.Clp  * b_2V * q   # q = Larp roll rate
                                  + self.Clda * delta_a
                                  + self.Cldr * delta_r)

        # Mz_b (about Larp Z=up) = −B&M yawing moment N (about Z_NED=down).
        # Overall sign flip because Z_LARP = −Z_NED.
        # Cnp uses p_BM = q_LARP; Cnr uses r_BM = -r_LARP (Cnr term sign cancels
        # with the overall flip → stays positive). Cnda/Cndr also flip.
        Mz_b = q_dyn * self.b * (-self.Cnp  * b_2V * q   # −Cnp (sign flip)
                                   + self.Cnr  * b_2V * r_  # +Cnr (double flip)
                                   - self.Cnda * delta_a     # sign flip
                                   - self.Cndr * delta_r)    # sign flip

        # ── 8. Body forces ────────────────────────────────────────────────
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)

        # Lateral (body X, right): side force only
        Fx_b = CY * q_dyn

        # Longitudinal (body Y, forward): thrust − drag·cos α + lift·sin α
        Fy_b = -cos_a * CD * q_dyn + sin_a * CL * q_dyn + self.k_t * delta_t * Va2

        # Vertical (body Z, up): lift·cos α + drag·sin α  ← both POSITIVE in Z-up frame
        # BUG FIX: was (−sin_a·CD − cos_a·CL) which pointed lift downward; corrected to:
        Fz_b = sin_a * CD * q_dyn + cos_a * CL * q_dyn

        # ── 9. World accelerations  (F_body → world via R_bw) ─────────────
        ax_ = ((cs*ct)*Fx_b + (cs*sp*st - ss*cp)*Fy_b + (cs*cp*st + ss*sp)*Fz_b) / self.m
        ay_ = ((ss*ct)*Fx_b + (ss*sp*st + cs*cp)*Fy_b + (ss*cp*st - cs*sp)*Fz_b) / self.m
        az_ = ((-st  )*Fx_b + (sp*ct          )*Fy_b + (cp*ct            )*Fz_b) / self.m - self.g

        # ── 10. Rotational dynamics (simplified — no inertia cross-coupling) ──
        dp = Mx_b / self.Ix   # d(pitch rate)/dt
        dq = My_b / self.Iy   # d(roll  rate)/dt
        dr = Mz_b / self.Iz   # d(yaw   rate)/dt

        # ── 11. Euler angle kinematics (ZYX: Rz·Ry·Rx) ──────────────────
        # Clip roll to avoid tan / sec singularity at ±90° bank.
        theta_c = np.clip(theta, -1.5, 1.5)
        tt   = np.tan(theta_c)
        sect = 1.0 / np.cos(theta_c)

        dphi   = p + (q * sp + r_ * cp) * tt    # d(pitch)/dt
        dtheta = q * cp - r_ * sp                # d(roll)/dt
        dpsi   = (q * sp + r_ * cp) * sect       # d(yaw)/dt

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

import importlib
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import numpy as np

from larp.const import JAX_INSTALLED, MJX_INSTALLED
from larp.dynamics import Dynamics

if JAX_INSTALLED:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, lax
    
if MJX_INSTALLED:
    import mujoco
    from mujoco import mjx


class MJXDynamics(Dynamics):
    r"""
    A unified rigid-body dynamics engine powered by MuJoCo XLA (mjx).

    This class parses arbitrary URDF, MJCF, or OpenUSD models into a fully 
    differentiable, batched dynamics environment. It natively computes continuous 
    Jacobians, discrete Hessians, and rollouts using JAX compilation.

    **State Representation:**
    The flattened first-order state is treated as a concatenation of generalized 
    positions and velocities: :math:`x = [q_{pos}, q_{vel}]`. 
    This means the state dimension is :math:`n_q + n_v`.

    **Control Representation:**
    The raw control input :math:`u` can be mapped to generalized forces :math:`\tau` 
    (or actuator controls) via the :meth:`control_to_nv` method. This allows 
    solvers to optimize over logical inputs (like PWM or thrust) while MuJoCo 
    handles the underlying rigid body mechanics.

    Attributes
    ----------
    mj_model : mujoco.MjModel
        The standard CPU-based MuJoCo model.
    mjx_model : mjx.Model
        The JAX-compiled MuJoCo model used for all computations.
    nq : int
        Number of generalized coordinates (position dimension).
    nv : int
        Number of degrees of freedom (velocity dimension).
    nu : int
        Number of actuators/controls natively defined in the MJCF model.

    Example
    -------
    .. code-block:: python

        # Load a complex eVTOL or Octocopter model
        dyn = MJXDynamics(model_path="assets/evtol.xml")
        
        x0 = np.zeros((10, dyn.nq + dyn.nv))
        u0 = np.zeros((10, dyn.nu))
        
        # Rollout 50 steps using exact MuJoCo physics (estimate is ignored)
        xs = dyn.rollout(x0, us_sequence, dt=0.01)
    """

    def __init__(self, model_path: str) -> None:
        if not MJX_INSTALLED:
            raise RuntimeError(
                "MJX is not installed or available. "
                "Please install JAX and MuJoCo to enable hardware-accelerated MJXDynamics."
            )

        # 1. Load Models
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # 2. Dimensions
        self.nq = self.mjx_model.nq
        self.nv = self.mjx_model.nv
        self.nu = self.mjx_model.nu

        # 3. Bypass Kinematic Shifting
        # By setting the derivative order to 0 for all elements, the base class 
        # treats the entire [qpos, qvel] vector as independent primitives.
        state_orders = [0] * (self.nq + self.nv)
        control_orders = [0] * self.nu

        super().__init__(
            constants={"model_path": model_path},
            state_derivative_orders=state_orders,
            control_derivative_orders=control_orders,
            holonomic=False,
            jax_backend=True # Always enforce True for MJXDynamics
        )

    def control_to_nv(self, u: np.ndarray, np: Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Maps the raw control input :math:`u` to generalized forces or native actuators.
        
        Defaults to the identity mapping. Override this method if your model is 
        underactuated or requires a complex transmission mapping (e.g., mapping 
        8 rotor speeds of an octocopter to 6 rigid-body forces).

        :param u: The control input batch. Shape: ``(batch_size, control_dim)``.
        :param np: Numerical backend (e.g., ``jnp``) passed during tracing.
        :return: Generalized forces or actuator inputs.
        """
        if np is None: np = jnp
        return u

    def nv_to_control(self, tau: np.ndarray, np: Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Inverse mapping from generalized forces back to control input :math:`u`.
        """
        raise NotImplementedError("nv_to_control must be implemented by the user.")

    @property
    def angle_indices(self) -> List[int]:
        return []

    def _setup_jax_functions(self):
        """
        Overrides the base class JAX setup to inject highly optimized, pre-compiled 
        `mjx` functions before the base class builds the Jacobians.
        """
        
        # --- 1. Define Pure Unbatched Functions ---
        def _f_single(x, u):
            qpos = x[:self.nq]
            qvel = x[self.nq:]
            
            # Map controls (add dummy batch dim, map, then strip)
            tau = self.control_to_nv(u[None, :], np=jnp)[0]
            
            data = mjx.make_data(self.mjx_model)
            data = data.replace(qpos=qpos, qvel=qvel)
            
            # Route the mapped control vector dynamically based on its shape
            if tau.shape[-1] == self.nu:
                data = data.replace(ctrl=tau)
            elif tau.shape[-1] == self.nv:
                data = data.replace(qfrc_applied=tau)
                
            data = mjx.forward(self.mjx_model, data)
            qacc = data.qacc
            
            # Compute dqpos (Handle Quaternions if Free Joint is present)
            if self.nq == self.nv:
                dqpos = qvel
            elif self.nq == self.nv + 1:
                dq_lin = qvel[:3]
                w = qvel[3:6]
                qw, qx, qy, qz = qpos[3:7]
                
                # Quaternion derivative: 0.5 * q \otimes [0, w]
                dqw = 0.5 * (-qx*w[0] - qy*w[1] - qz*w[2])
                dqx = 0.5 * ( qw*w[0] + qy*w[2] - qz*w[1])
                dqy = 0.5 * ( qw*w[1] - qx*w[2] + qz*w[0])
                dqz = 0.5 * ( qw*w[2] + qx*w[1] - qy*w[0])
                
                dq_quat = jnp.array([dqw, dqx, dqy, dqz])
                dqpos = jnp.concatenate([dq_lin, dq_quat, qvel[6:]])
            else:
                pad = jnp.zeros(self.nq - self.nv)
                dqpos = jnp.concatenate([qvel, pad])
                
            return jnp.concatenate([dqpos, qacc])

        def _mjx_step_single(x, u, dt):
            model = self.mjx_model.replace(opt=self.mjx_model.opt.replace(timestep=dt))
            qpos = x[:self.nq]
            qvel = x[self.nq:]
            
            tau = self.control_to_nv(u[None, :], np=jnp)[0]
            
            data = mjx.make_data(model)
            data = data.replace(qpos=qpos, qvel=qvel)
            
            if tau.shape[-1] == self.nu:
                data = data.replace(ctrl=tau)
            elif tau.shape[-1] == self.nv:
                data = data.replace(qfrc_applied=tau)
                
            data = mjx.step(model, data)
            return jnp.concatenate([data.qpos, data.qvel])

        def _mjx_rollout_single(x0_single, us_time_single, dt):
            def scan_op(x_prev, u_curr):
                x_next = _mjx_step_single(x_prev, u_curr, dt)
                return x_next, x_next
            _, xs_traj = lax.scan(scan_op, x0_single, us_time_single)
            return xs_traj

        # --- 2. Compile, Vmap, and Store ---
        self._f_jit = jit(vmap(_f_single, in_axes=(0, 0)))
        self._mjx_step_jit = jit(vmap(_mjx_step_single, in_axes=(0, 0, None)))
        self._mjx_rollout_jit = jit(vmap(_mjx_rollout_single, in_axes=(0, 1, None)))

        # --- 3. Let Base Class Build Jacobians ---
        # The base class will now use our optimized self.f to construct dfdx, dfdu, etc.
        super()._setup_jax_functions()

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np: Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Computes the continuous-time derivative :math:`[\dot{q}_{pos}, \dot{q}_{vel}]`.

        This leverages ``mjx.forward`` to compute accelerations directly.
        If the model contains free joints (quaternions), it automatically calculates 
        the correct quaternion derivatives.
        """
        if np is None: np = jnp
        # Route directly to the compiled, batched function
        return self._f_jit(jnp.asarray(first_order_state), jnp.asarray(first_order_control))

    def step(self, x0: np.ndarray, u0: np.ndarray, dt: float = 0.1, estimate: bool = False) -> np.ndarray:
        r"""
        Advances the MuJoCo system dynamics forward by one time step :math:`dt`.

        :param x0: Current state batch :math:`x_k`.
        :param u0: Current control batch :math:`u_k`.
        :param dt: Time step duration in seconds.
        :param estimate: Retained for signature compatibility, but ignored. 
            `mjx.step` is strictly used for high-fidelity physics resolution.
        """
        x_next = self._mjx_step_jit(jnp.asarray(x0), jnp.asarray(u0), dt)
        return np.asarray(x_next)

    def rollout(self, x0: np.ndarray, us: np.ndarray, dt: float = 0.1, estimate: bool = False) -> np.ndarray:
        r"""
        Simulates a batched forward rollout of the MuJoCo dynamics over a sequence of controls.
        
        :param estimate: Retained for signature compatibility, but ignored.
        """
        xs = self._mjx_rollout_jit(jnp.asarray(x0), jnp.asarray(us), dt)
        return np.asarray(xs)