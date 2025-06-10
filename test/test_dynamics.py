
import larp.dynamics
import numpy as np
import scipy.integrate as integrate

"""
Author: Josue N Rivera

"""

def test_whl_dynamics():

    dynamics = larp.dynamics.WMRDynamics(1.0)

    def f(t:np.ndarray, x:np.ndarray):
        t = np.reshape(t, (-1, 1))
        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.sin(t / 2).reshape(-1, 1).repeat(2, axis=1)

        return dynamics.f(t, x, u)[0]

    t_span = (0, 10)
    t = np.linspace(*t_span, 10)
    x0 = np.array([1, 2, 1])
    
    integrate.solve_ivp(f, t_span, x0, t_eval=t)

def test_quadcopter_dynamics():

    dynamics = larp.dynamics.QuadcopterDynamics()

    def f(t:np.ndarray, x:np.ndarray):
        t = np.reshape(t, (-1, 1))
        x = x.reshape(-1, dynamics.first_order_state_n)
        u = np.zeros((1, 4))
        u[:, 0] = 51

        return dynamics.f(t, x, u)[0]

    t_span = (0, 5)
    t = np.linspace(*t_span, 100)
    x0 = np.array([0]*4+[1]+[0]*7)
    
    integrate.solve_ivp(f, t_span, x0, t_eval=t)

def test_discretize():

    dynamics = larp.dynamics.WMRDynamics()

    x0, u0 = np.array([0, 0, np.pi/4]).reshape(-1, 3), np.array([0, 0]).reshape(-1, 2)

    A, B = dynamics.discretize(x0, u0)
    