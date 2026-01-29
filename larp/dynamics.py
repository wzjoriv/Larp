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
        :param jax_backend: If ``True`` and JAX is installed, enables JIT compilation for linearization 
            and discretization methods. This requires the subclass to implement ``f`` 
            in a way that accepts the ``np`` module argument for backend injection.
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

        self._setup_jax_functions()

    @property
    @abstractmethod
    def angle_indices(self) -> List[int]:
        """Indices in the flattened state vector that represent angles (need wrapping)."""
        return []

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

            # --- IMPLEMENTATION A: Exact ZOH (Matrix Exponential) ---
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
            
            # --- IMPLEMENTATION B: Forward Euler Approximation ---
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
        if self.jax_backend:
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
                         jax_backend=True)    

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
        #   - 3 Angles (Order 0: phi, theta, psi -> manually integrated via kinematics)
        #   - 3 Rates (Order 0: p, q, r -> manually integrated via dynamics)
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
    
class QuadcopterV2Dynamics(Dynamics):
    """
    Quadcopter dynamics using individual rotor speeds as control inputs.
    
    Frame Convention:
        x: Right (Lateral)
        y: Forward (Longitudinal)
        z: Up (Vertical)
    
    Rotational Mapping:
        phi (Rotation about X): Pitch Angle
        theta (Rotation about Y): Roll Angle
        psi (Rotation about Z): Yaw Angle

    Motor Order:
        1: Back-Right (X)  or Right (+)  | CW
        2: Back-Left (X)   or Back  (+)  | CCW
        3: Front-Left (X)  or Left  (+)  | CW
        4: Front-Right (X) or Front (+)  | CCW
    """

    def __init__(self, 
                 frame: str = 'x', # 'x' or '+'
                 inertia = [3.8e-3, 3.8e-3, 7.1e-3], 
                 mass = 1.0, 
                 gravity = 9.807,
                 arm_length = 0.32, 
                 thrust_constant = 3.13e-5, 
                 translational_drag = [0.1, 0.1, 0.15],
                 torque_constant = 7.5e-7, 
                 rotational_drag = [0.1, 0.1, 0.15], 
                 motor_inertia = 6e-5) -> None:
        
        constants = {k: v for k,v in locals().items() if k != 'self'}
        super().__init__(constants=constants, state_derivative_orders=[1]*6, control_derivative_orders=[0]*4, jax_backend=True)
        
        self.frame = frame
        self.m, self.g, self.l = float(mass), float(gravity), float(arm_length)
        self.b, self.d, self.Jr = float(thrust_constant), float(torque_constant), float(motor_inertia)
        self.I = np.array(inertia, dtype=float)
        self.Ct, self.Cr = np.array(translational_drag, dtype=float), np.array(rotational_drag, dtype=float)

        # Mixing Matrix
        if frame == 'x':
            # 1: Back-Right (X>0, Y<0) -> -45 deg
            # 2: Back-Left  (X<0, Y<0) -> -135 deg
            # 3: Front-Left (X<0, Y>0) -> +135 deg
            # 4: Front-Right(X>0, Y>0) -> +45 deg
            angles = np.array([-np.pi/4, -3*np.pi/4, 3*np.pi/4, np.pi/4])
        elif frame == '+':
            # 1: Right (X>0) -> 0 deg
            # 2: Back  (Y<0) -> -90 deg
            # 3: Left  (X<0) -> 180 deg
            # 4: Front (Y>0) -> 90 deg
            angles = np.array([0.0, -np.pi/2, np.pi, np.pi/2])
        else:
            raise ValueError(f"Unknown frame type: {frame}")

        self.motor_pos = np.stack([self.l * np.cos(angles), self.l * np.sin(angles)], axis=1)
        
        # --- Axis Logic ---
        # Pitch: Rotation about X-axis. Controlled by Y positions (Front/Back).
        # Roll: Rotation about Y-axis. Controlled by X positions (Right/Left).
        
        col_thrust = np.ones(4) * self.b
        
        # Torque X (Pitch): Positive y (Front) -> Positive Torque (Nose Up)
        col_pitch_torque = self.b * self.motor_pos[:, 1]
        
        # Torque Y (Roll): Positive x (Right) -> Negative Torque (Roll Left/Right Wing Up)
        col_roll_torque = self.b * (-self.motor_pos[:, 0])
        
        col_yaw = np.array([self.d, -self.d, self.d, -self.d])

        # Stack order: [Thrust, Torque_X (Pitch), Torque_Y (Roll), Torque_Z (Yaw)]
        self.M = np.stack([col_thrust, col_pitch_torque, col_roll_torque, col_yaw], axis=0)
        self.inv_M = np.linalg.inv(self.M)

    @property
    def angle_indices(self) -> List[int]:
        # State: [x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi]
        return [6, 8, 10]

    def to_rotor_speed(self, x0, thrust=None, roll=None, pitch=None, yaw=None):
        """
        Maps desired Forces/Torques to Rotor Speeds (Control Inputs).
        Solves: w^2 = M^{-1} * u
        """
        batch_size = x0.shape[0]
        
        def _format_input(val):
            if val is None:
                return np.zeros((batch_size, 1))
            val = np.array(val)
            if val.ndim == 0:
                return np.full((batch_size, 1), val)
            return val.reshape(batch_size, 1)

        u_thrust = _format_input(thrust)
        u_roll = _format_input(roll)
        u_pitch = _format_input(pitch)
        u_yaw = _format_input(yaw)
        
        # Order must match M: [Thrust, Pitch (Torque X), Roll (Torque Y), Yaw]
        u = np.concatenate([u_thrust, u_pitch, u_roll, u_yaw], axis=1)
        
        w_sq = u @ self.inv_M.T
        w = np.sqrt(np.maximum(w_sq, 0.0))
        
        return w

    def to_force(self, first_order_control: np.ndarray) -> np.ndarray:
        """ 
        Converts vector of rotor speeds (w) to body frame forces/torques (u).
        Returns: [Thrust, Pitch_Torque, Roll_Torque, Yaw_Torque]
        """
        w_2 = first_order_control**2
        return w_2 @ self.M.T

    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        if np is None: np = importlib.import_module("numpy")

        # --- Unpack State ---
        # State: [x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi]
        vel = first_order_state[:, 1:6:2] 
        ang = first_order_state[:, 6:11:2] 
        rate = first_order_state[:, 7:12:2] 
        
        phi, theta, psi = ang[:, 0:1], ang[:, 1:2], ang[:, 2:3]
        
        # Control -> Forces
        w_2 = first_order_control**2
        forces = w_2 @ self.M.T
        thrust, torque = forces[:, 0:1], forces[:, 1:]

        # Rotation Matrix (Body -> World) R = Rz(psi) * Ry(theta) * Rx(phi)
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)
        c_s, s_s = np.cos(psi), np.sin(psi)
        
        # Row 1
        r11 = c_t * c_s
        r12 = s_p * s_t * c_s - c_p * s_s
        r13 = c_p * s_t * c_s + s_p * s_s
        
        # Row 2
        r21 = c_t * s_s
        r22 = s_p * s_t * s_s + c_p * c_s
        r23 = c_p * s_t * s_s - s_p * c_s
        
        # Row 3
        r31 = -s_t
        r32 = s_p * c_t
        r33 = c_p * c_t

        # R constructed directly: Shape (Batch, 3, 3)
        R = np.stack([
            np.concatenate([r11, r12, r13], axis=1),
            np.concatenate([r21, r22, r23], axis=1),
            np.concatenate([r31, r32, r33], axis=1)
        ], axis=1)

        # Translational Dynamics
        # vel_body = R.T @ vel_world
        vel_body = np.transpose(R, (0, 2, 1)) @ vel[..., None]

        # Drag Force (Body Frame)
        drag_force = self.Ct.reshape(1,3,1) * vel_body

        # Thrust Force (Body Frame): [0, 0, T]
        zeros = np.zeros_like(thrust)
        thrust_vec = np.stack([zeros, zeros, thrust], axis=1)

        F_body = thrust_vec - drag_force

        # Convert to World Frame acceleration: a = (R @ F_body) / m - g
        accel_world = (R @ F_body).reshape(-1, 3) / self.m
        accel_z = accel_world[:, 2:3] - self.g
        
        # Rotational Dynamics
        omega_r = (first_order_control[:, 0:1] - first_order_control[:, 1:2] + 
                   first_order_control[:, 2:3] - first_order_control[:, 3:4])
        
        dphi, dth, dpsi = rate[:, 0:1], rate[:, 1:2], rate[:, 2:3]
        Ix, Iy, Iz = self.I
        
        # Euler Equations
        drag_phi = self.Cr[0] * dphi * np.abs(dphi)
        drag_th  = self.Cr[1] * dth  * np.abs(dth)
        drag_psi = self.Cr[2] * dpsi * np.abs(dpsi)

        # Euler Equations
        ddphi = (torque[:,0:1] - drag_phi - self.Jr*omega_r*dth - (Iz-Iy)*dth*dpsi)/Ix
        ddth  = (torque[:,1:2] - drag_th  + self.Jr*omega_r*dphi - (Ix-Iz)*dphi*dpsi)/Iy
        ddpsi = (torque[:,2:3] - drag_psi - (Iy-Ix)*dphi*dth)/Iz
        
        return np.concatenate([vel[:, 0:1], accel_world[:, 0:1], 
                               vel[:, 1:2], accel_world[:, 1:2], 
                               vel[:, 2:3], accel_z,
                               dphi, ddphi, dth, ddth, dpsi, ddpsi], axis=1)
    
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
        max_w_dynamic = self.max_w_nominal * (v_load / self.v_max)
        
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