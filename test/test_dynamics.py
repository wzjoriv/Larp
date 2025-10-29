
import larp
import numpy as np
import scipy.integrate as integrate

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

    def f(t:np.ndarray, x:np.ndarray):

        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.zeros((1, 2))
        
        return dynamics.f(x, u)[0]
    
    t_span = (0, 5)
    t = np.linspace(*t_span, 100)
    x0 = np.array([1, 2, 1])
    
    xs = integrate.solve_ivp(f, t_span, x0, t_eval=t).y.T

    assert np.linalg.norm(xs - x0) < 1e-10, "The system deviates from a stable state."

def test_whl_discretize():

    dynamics = larp.dynamics.WMRDynamics(1.0)
    dt = 0.5

    # Continuous system
    def f(t:np.ndarray, x:np.ndarray):
        x = x.reshape(-1, dynamics.first_order_state_n)
        t = (t//dt)*dt
        u = np.ones((1, 2))*((t-1)**2)*np.array([0.5, 0.2*t])
        
        return dynamics.f(x, u)[0]
    
    t_span = (0, 4)
    t = np.linspace(*t_span, 100)
    x0 = np.array([1, 2, 1])
    
    xs = integrate.solve_ivp(f, t_span, x0, t_eval=t)

    # Discrite system: Estimate
    xdc = xd0 = x0
    xds = [xd0]

    for t in np.arange(t_span[0], t_span[1], dt):
        t = t//dt
        u = np.ones((1, 2))*((t-1)**2)*np.array([0.5, 0.2*t])

        A, B, f = dynamics.discretize(xdc.reshape(1, -1), u)[0]
        xc_d = f + A@xdc + B@u[0]

        xc_d.append(xdc)

    xds = np.array(xds)
    


def test_quadcopter_dynamics():

    dynamics = larp.dynamics.QuadcopterV2Dynamics()

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

    f, A, B = dynamics.discretize(x0, u0)