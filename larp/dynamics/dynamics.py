from abc import ABC, abstractmethod
import inspect
from types import ModuleType
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm
from itertools import chain
import warnings

from larp.const import JAX_INSTALLED

if JAX_INSTALLED:
    import jax.numpy as jnp
    from jax import jacfwd, jit, vmap, lax
    from jax.scipy.linalg import expm as jexpm

"""
Author: Josue N Rivera

Dynamics classes
"""

class Dynamics(ABC):

    r"""
    Base class for dynamical systems with batched ops and optional JAX autodiff.

    Models systems in first-order form :math:`\dot{x} = f(x, u)`, where
    :math:`x \in \mathbb{R}^{n}` is the flattened state vector and
    :math:`u \in \mathbb{R}^{m}` is the flattened control vector.

    Maps "primitive" configuration variables (e.g., position :math:`p`) to a
    flattened first-order vector containing their derivatives
    (e.g., :math:`[p, \dot{p}, \ddot{p}]`).  If JAX is installed and analytical
    gradients are not provided, Jacobians and Hessians are computed automatically
    via forward-mode autodiff.

    Attributes
    ----------
    constants : dict
        Physical parameters used in the model (e.g., mass, length).
    holonomic : bool or None
        Whether the system is holonomic.
    state_derivative_orders : ndarray of int
        Highest derivative order for each primitive state variable.
    control_derivative_orders : ndarray of int
        Highest derivative order for each primitive control variable.
    primitive_state_n : int
        Number of primitive state variables.
    primitive_control_n : int
        Number of primitive control variables.
    first_order_state_n : int
        Total dimension :math:`n` of the flattened first-order state vector.
    first_order_control_n : int
        Total dimension :math:`m` of the flattened first-order control vector.
    first_state_orders : ndarray of int
        Derivative order of each element in the flattened state vector.
    first_control_orders : ndarray of int
        Derivative order of each element in the flattened control vector.
    state_primitive_mask : ndarray of bool
        True at indices corresponding to 0th-order (primitive) state entries.
    control_primitive_mask : ndarray of bool
        True at indices corresponding to 0th-order (primitive) control entries.
    first_wrapped_state_mask : ndarray of bool, shape (first_order_state_n,)
        True at state indices that require angular wrapping to :math:`[-\pi, \pi]`.
    jax_backend : bool
        Whether JAX JIT compilation is enabled for this instance.

    Examples
    --------
    Define a simple 1-primitive system and linearize at the origin:

    .. code-block:: python

        # 1 primitive state (order 1: pos + vel), 1 primitive control (order 0)
        dyn = Dynamics(state_derivative_orders=[1], control_derivative_orders=[0])

        x0 = np.zeros((1, 2))  # [pos, vel]
        u0 = np.zeros((1, 1))  # [force]

        A, B, f0 = dyn.linearize(x0, u0)
        print(A.shape)  # (1, 2, 2)
    """

    def __init__(self,
                 constants: Optional[Dict] = None,
                 state_derivative_orders: List[int] = [1],
                 control_derivative_orders: List[int] = [0],
                 wrapable_primitive_state: Optional[List[int]] = None,
                 holonomic: Optional[bool] = None,
                 jax_backend = False) -> None:

        r"""
        Parameters
        ----------
        constants : dict, optional
            Physical parameters or constants used in the model (e.g., mass, length).
            Defaults to an empty dict.
        state_derivative_orders : list of int
            Highest derivative order for each primitive state variable.
            For example, ``[2, 1]`` means:

            - Primitive 0 (position): entries :math:`[p, \dot{p}, \ddot{p}]`.
            - Primitive 1 (heading): entries :math:`[\theta, \dot{\theta}]`.
        control_derivative_orders : list of int
            Same as ``state_derivative_orders`` but for control variables.
            ``[0]`` means the control is a 0th-order quantity (e.g., force directly).
        wrapable_primitive_state : list of int, optional
            Indices into the *primitive* list (not the flattened vector) of states
            that represent angles and should be wrapped to :math:`[-\pi, \pi]`.
        holonomic : bool, optional
            Whether the system is holonomic.
        jax_backend : bool
            If ``True`` and JAX is installed, enables JIT compilation for ``step``,
            ``rollout``, ``linearize``, and ``discretize``.  Requires ``f`` to
            accept an ``np`` keyword argument for backend injection.  Regardless of
            this flag, JAX autodiff is used for Jacobians/Hessians when JAX is
            installed and the subclass does not override the derivative methods.
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
        self.first_state_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.state_derivative_orders])
        self.first_control_orders = np.concatenate([np.arange(i+1, dtype=int) for i in self.control_derivative_orders])

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
        Offset from the standard path-direction angle to this model's yaw state.

        Planners derive headings from ``arctan2(dy, dx)`` (measured from +X world
        axis).  Add this offset before writing the value into the reference state
        vector so that the heading convention of the dynamics model is respected.

        Returns
        -------
        float
            Angle offset in radians.  Default ``0.0`` (WMR, Car).
            Override to ``-pi/2`` for ``QuadcopterDynamics`` whose ``psi=0``
            points nose toward +Y rather than +X.

        Notes
        -----
        This property lives on ``Dynamics`` because the dynamics model is the
        authoritative source of its own state-variable conventions, consistent
        with how ROS2/tf2, Drake, and ALTRO handle frame semantics.
        """
        return 0.0

    def _setup_jax_functions(self):
        """Compile JAX JIT functions for derivatives, linearization, and discretization."""

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
        Split a batched first-order vector into a list of per-element arrays.

        Unzips the columns of ``first`` so each element of the state or control
        vector becomes a separate ``(batch_size, 1)`` array.  Commonly used
        inside ``f()`` to unpack the state vector into named variables.

        Parameters
        ----------
        first : ndarray, shape (B, k)
            Batched first-order state or control vector.

        Returns
        -------
        list of ndarray, each shape (B, 1)
            One array per element of the input vector.

        Examples
        --------
        >>> state = np.array([[1.0, 2.0, 3.14]])
        >>> x, y, theta = dyn.split_first(state)
        >>> x
        array([[1.]])
        """
        return [first[:, i:i+1] for i in range(first.shape[1])]

    @abstractmethod
    def f(self, first_order_state: np.ndarray, first_order_control: np.ndarray, np:Optional[ModuleType] = None) -> np.ndarray:
        r"""
        Compute the continuous-time state derivative :math:`\dot{x} = f(x, u)`.

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.
        np : module, optional
            Numerical backend (``numpy`` or ``jax.numpy``).  Implementations
            must route all math through this argument to support JAX autodiff.

        Returns
        -------
        ndarray, shape (B, n)
            Time derivative :math:`\dot{x}`.
        """
        raise NotImplementedError

    def dfdx(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Compute the Jacobian of the dynamics w.r.t. the state.

        :math:`A = \frac{\partial f(x, u)}{\partial x}`

        Falls back to JAX forward-mode autodiff when JAX is installed and this
        method is not overridden by the subclass.

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, n)
            State Jacobian :math:`A`.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
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
        Compute the Jacobian of the dynamics w.r.t. the control.

        :math:`B = \frac{\partial f(x, u)}{\partial u}`

        Falls back to JAX forward-mode autodiff when JAX is installed and this
        method is not overridden by the subclass.

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, m)
            Control Jacobian :math:`B`.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
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
        Compute the state-state Hessian of the dynamics.

        :math:`f_{xx} = \frac{\partial^2 f}{\partial x^2}`

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, n, n)
            State-state Hessian tensor.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
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
        Compute the control-control Hessian of the dynamics.

        :math:`f_{uu} = \frac{\partial^2 f}{\partial u^2}`

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, m, m)
            Control-control Hessian tensor.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
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
        Compute the state-control cross-Hessian of the dynamics.

        :math:`f_{xu} = \frac{\partial^2 f}{\partial x \partial u}`

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, n, m)
            State-control cross-Hessian tensor.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
        """
        if JAX_INSTALLED and self._dfdxu_jit:
            x = jnp.asarray(first_order_state)
            u = jnp.asarray(first_order_control)
            return np.asarray(self._dfdxu_jit(x, u))

        raise NotImplementedError(
            "dfdxu is not implemented and JAX is not installed. "+
            "Install JAX to enable autodiff."
        )

    def dfdux(self, first_order_state: np.ndarray, first_order_control: np.ndarray) -> np.ndarray:
        r"""
        Compute the control-state cross-Hessian of the dynamics.

        :math:`f_{ux} = \frac{\partial^2 f}{\partial u \partial x}`

        Parameters
        ----------
        first_order_state : ndarray, shape (B, n)
            Batched state vector :math:`x`.
        first_order_control : ndarray, shape (B, m)
            Batched control vector :math:`u`.

        Returns
        -------
        ndarray, shape (B, n, m, n)
            Control-state cross-Hessian tensor.

        Raises
        ------
        NotImplementedError
            If JAX is not installed and the subclass provides no implementation.
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
        Linearize the dynamics about a reference point :math:`(x_0, u_0)`.

        Computes a first-order Taylor expansion:

        :math:`\dot{x} \approx f(x_0, u_0) + A(x - x_0) + B(u - u_0)`

        where :math:`A = \partial f/\partial x|_{x_0,u_0}` and
        :math:`B = \partial f/\partial u|_{x_0,u_0}`.

        Parameters
        ----------
        x0 : ndarray, shape (B, n)
            Reference state batch.
        u0 : ndarray, shape (B, m)
            Reference control batch.

        Returns
        -------
        A : ndarray, shape (B, n, n)
            State Jacobian.
        B : ndarray, shape (B, n, m)
            Control Jacobian.
        f0 : ndarray, shape (B, n)
            Nominal dynamics :math:`f(x_0, u_0)`.

        Examples
        --------
        .. code-block:: python

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
        Discretize the linearized dynamics over a time step ``dt``.

        Converts the affine continuous system :math:`\dot{x} = Ax + Bu + K`
        into the discrete form :math:`x_{k+1} = A_d x_k + B_d u_k + f_d`.

        Parameters
        ----------
        x0 : ndarray, shape (B, n)
            Reference state for linearization.
        u0 : ndarray, shape (B, m)
            Reference control for linearization.
        dt : float
            Time step in seconds.
        estimate : bool
            If ``True`` (default), uses Forward Euler (:math:`A_d \approx I + A\,dt`).
            If ``False``, uses exact Zero-Order Hold via matrix exponential.

        Returns
        -------
        Ad : ndarray, shape (B, n, n)
            Discrete state transition matrix.
        Bd : ndarray, shape (B, n, m)
            Discrete control input matrix.
        fd : ndarray, shape (B, n)
            Discrete affine offset term.

        Examples
        --------
        .. code-block:: python

            Ad, Bd, fd = dyn.discretize(x_curr, u_curr, dt=0.05, estimate=False)
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
        Discretize the second-order dynamics Hessians over a time step ``dt``.

        Parameters
        ----------
        x0 : ndarray, shape (B, n)
            Reference state batch.
        u0 : ndarray, shape (B, m)
            Reference control batch.
        dt : float
            Time step in seconds.
        estimate : bool
            If ``True`` (default), uses Euler scaling: :math:`F_{xx} \approx f_{xx}\,dt`.
            If ``False``, differentiates the full ZOH step map twice via JAX autodiff
            to obtain the exact discrete Hessian (requires ``jax_backend=True``).

        Returns
        -------
        F_xx : ndarray, shape (B, n, n, n)
            Discrete state-state Hessian.
        F_uu : ndarray, shape (B, n, m, m)
            Discrete control-control Hessian.
        F_ux : ndarray, shape (B, n, m, n)
            Discrete control-state cross-Hessian.

        Raises
        ------
        NotImplementedError
            If ``estimate=False`` and neither ``jax_backend`` nor an analytical
            override of ``dfdxx`` / ``dfduu`` / ``dfdux`` is available.

        Notes
        -----
        The ZOH-exact path (``estimate=False``) yields the correct quantity for
        second-order DDP and avoids the :math:`O(dt)` truncation error of the
        Euler approximation.
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
        Advance the system one time step :math:`dt`.

        Supports batched inputs and optional JAX JIT compilation.

        Parameters
        ----------
        x0 : ndarray, shape (B, n)
            Current state batch :math:`x_k`.
        u0 : ndarray, shape (B, m)
            Current control batch :math:`u_k`.
        dt : float
            Time step in seconds.
        estimate : bool
            If ``True`` (default), uses Forward Euler integration.
            If ``False``, uses 4th-order Runge-Kutta (RK4).

        Returns
        -------
        ndarray, shape (B, n)
            Next state batch :math:`x_{k+1}`.

        Examples
        --------
        .. code-block:: python

            x_curr = np.zeros((10, 4))
            u_curr = np.ones((10, 2))
            x_next = dyn.step(x_curr, u_curr, dt=0.05, estimate=False)
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
        Simulate a forward rollout over a sequence of controls.

        Expects a **time-major** control sequence (time on axis 0).  The initial
        state ``x0`` is not included in the returned trajectory.

        Parameters
        ----------
        x0 : ndarray, shape (B, n)
            Initial state batch.
        us : ndarray, shape (T, B, m)
            Control sequence over ``T`` time steps.
        dt : float
            Time step in seconds.
        estimate : bool
            If ``True`` (default), uses Forward Euler integration.
            If ``False``, uses 4th-order Runge-Kutta (RK4).

        Returns
        -------
        ndarray, shape (T, B, n)
            State trajectory excluding the initial state.
            ``xs[0]`` is the state at :math:`t = dt`,
            ``xs[-1]`` is the state at :math:`t = T \cdot dt`.

        Examples
        --------
        .. code-block:: python

            x0 = np.zeros((5, 4))     # [Batch, State_Dim]
            us = np.ones((20, 5, 2))  # [Time, Batch, Control_Dim]
            xs = dyn.rollout(x0, us, dt=0.05)
            # xs.shape == (20, 5, 4)
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
        Symbolic names for each element of the flattened first-order state vector.

        Format: ``x_{primitive_idx}^{[derivative_order]}``.

        Returns
        -------
        list of str
            Labels for each state entry.

        Examples
        --------
        >>> dyn = Dynamics(state_derivative_orders=[1])
        >>> dyn.first_state_names
        ['x_0^{[0]}', 'x_0^{[1]}']
        """
        orders = self.state_derivative_orders
        return [f'x_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_control_names(self) -> List[str]:
        r"""
        Symbolic names for each element of the flattened first-order control vector.

        Format: ``u_{primitive_idx}^{[derivative_order]}``.

        Returns
        -------
        list of str
            Labels for each control entry.
        """
        orders = self.control_derivative_orders
        return [rf'u_{{{o_idx}}}^{{[{i}]}}' for o_idx in range(len(orders)) for i in range(orders[o_idx]+1)]

    @property
    def first_names(self) -> Tuple[List[str], List[str]]:
        """
        State and control symbolic names for the flattened first-order vectors.

        Returns
        -------
        state_names : list of str
            Output of :attr:`first_state_names`.
        control_names : list of str
            Output of :attr:`first_control_names`.
        """
        return self.first_state_names, self.first_control_names

    def __repr__(self, verbose: bool = False) -> str:
        """
        Return a string representation of the Dynamics object.

        Parameters
        ----------
        verbose : bool
            If ``True``, returns a detailed multi-line description including
            derivative orders, symbolic state/control names, and constants.
            Default ``False`` returns a concise single-line summary.

        Returns
        -------
        str
            Formatted string representation of the instance.
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
