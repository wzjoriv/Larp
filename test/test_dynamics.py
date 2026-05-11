
import larp
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

"""
Author: Josue N Rivera

"""

def test_whl_dynamics():

    dynamics = larp.dynamics.WMRDynamics(1.0)

    def f(t:np.ndarray, x:np.ndarray):

        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.sin(t / 2).reshape(-1, 1).repeat(2, axis=1)

        return dynamics.f(x, u)[0]

    t_span = (0, 5)
    t = np.linspace(*t_span, 100)
    x0 = np.array([1, 2, 1])
    
    xs = integrate.solve_ivp(f, t_span, x0, t_eval=t).y.T

def test_whl_stability():

    dynamics = larp.dynamics.WMRDynamics(1.0)
    dt = 0.1

    def f(t:np.ndarray, x:np.ndarray):

        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.zeros((1, 2))
        
        return dynamics.f(x, u)[0]
    
    t_span = (0, 5)
    t_ds = np.arange(t_span[0], t_span[1]+dt, dt)
    x0 = np.array([1, 2, 1])
    
    xs = integrate.solve_ivp(f, t_span, x0, t_eval=t_ds).y.T

    assert np.linalg.norm(xs - x0) < 1e-10, "The system deviates from a stable state."

    # Discrete system estimate (Estimate)
    xdc = x0.copy()
    xds = [xdc.copy()]

    for t in np.arange(t_span[0], t_span[1], dt):
        u = np.zeros((1, 2))

        A, B, g = dynamics.discretize(xdc[None], u[0][None], dt, estimate=True)
        xdc = g[0] + A[0] @ xdc + B[0] @ u[0]
        xds.append(xdc.copy())

    xds = np.array(xds)
    
    assert np.linalg.norm(xds - x0) < 1e-10, "The discritized system deviates from a stable state."

    # Discrete system estimate (Accurate)
    xdc = x0.copy()
    xds = [xdc.copy()]

    for t in np.arange(t_span[0], t_span[1], dt):
        u = np.zeros((1, 2))

        A, B, g = dynamics.discretize(xdc[None], u[0][None], dt, estimate=False)
        xdc = g[0] + A[0] @ xdc + B[0] @ u[0]
        xds.append(xdc.copy())

    xds = np.array(xds)
    
    assert np.linalg.norm(xds - x0) < 1e-10, "The accurate discritized system deviates from a stable state."

def test_whl_discretize():
    """Test WMRDynamics discretization against continuous integration."""

    dynamics = larp.dynamics.WMRDynamics(1.0)
    dt = 0.1

    # Continuous system dynamics for integration
    def f(t: float, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, dynamics.first_order_state_n)
        t_rounded = np.floor(t / dt) * dt
        u = np.array([[0.5 * (t_rounded - 1)**2, 0.2 * t_rounded * (t_rounded - 1)**2]])
        return dynamics.f(x, u)[0]

    t_span = (0.0, 2.0)
    t_ds = np.arange(t_span[0], t_span[1]+dt, dt)
    x0 = np.array([1.0, 2.0, 1.0])

    # Continuous integration
    xs = integrate.solve_ivp(f, t_span, x0, t_eval=t_ds).y.T

    # Discrete system estimate
    xdc = x0.copy()
    xds = [xdc.copy()]

    for t in np.arange(t_span[0], t_span[1], dt):
        t_rounded = round(t, 2)
        u = np.array([[0.5 * (t_rounded - 1)**2, 0.2 * t_rounded * (t_rounded - 1)**2]])

        A, B, g = dynamics.discretize(xdc[None], u[0][None], dt, estimate=True)
        xdc = g[0] + A[0] @ xdc + B[0] @ u[0]
        xds.append(xdc.copy())

    xds = np.array(xds)

    assert np.allclose(xds, xs, rtol=1e-2, atol=1e-2), "Error between continous and discritized system is significant."

def test_quadcopter_dynamics():

    dynamics = larp.dynamics.QuadcopterDynamics()

    def f(t:np.ndarray, x:np.ndarray):
        t = np.reshape(t, (-1, 1))
        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.zeros((1, 4))
        u[:, 0] = 51

        return dynamics.f(x, u)[0]

    t_span = (0, 5)
    t = np.linspace(*t_span, 100)
    x0 = np.array([0]*4+[1]+[0]*7)
    
    integrate.solve_ivp(f, t_span, x0, t_eval=t)

def test_discretize():

    dynamics = larp.dynamics.WMRDynamics()

    x0, u0 = np.array([0, 0, np.pi/4]).reshape(-1, 3), np.array([0, 0]).reshape(-1, 2)

    A, B, f = dynamics.discretize(x0, u0)