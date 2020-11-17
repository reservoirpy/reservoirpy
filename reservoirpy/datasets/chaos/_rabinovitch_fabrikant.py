from typing import Tuple

import numpy as np

from scipy.integrate import solve_ivp



def rabinovich_fabrikant(n_timesteps: int, state0: list=[-1, 0, 0.5], 
                         gamma: float=0.89, alpha: float=1.1, h: float=0.01)-> Tuple[np.ndarray, np.ndarray]:
    """Rabinovitch-Fabrikant equations timeseries.
    .. math::
        \dv{x}{t} = y(z - 1 + x^2) + \gamma x
    .. math::
        \dv{y}{t} = x(3z + 1 - x^2) + \gamma y
    .. math::
        \dv{z}{t} = -2z(\alpha + xy)
    Args:
        n_timesteps (int): Number of timesteps to generate.
        state0 (list, optional): Initial conditions of the system. Defaults to [-1, 0, 0.5].
        gamma (float, optional): Gamma parameter of the equations. Defaults to 0.89.
        alpha (float, optional): Alpha parameter of the equations. Defaults to 1.1.
        h (float, optional): Control the time delta between two timesteps. Defaults to 0.01.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated timeseries and timesteps.
    """
    
    def rabinovich_fabrikant_diff(t, state):
        x, y, z = state
        dx = y*(z - 1 + x**2) + gamma*x
        dy = x*(3*z + 1 - x**2) + gamma*y
        dz = -2*z*(alpha + x*y)
        return dx, dy, dz

    t = np.arange(0.0, n_timesteps*h, h)

    sol = solve_ivp(rabinovich_fabrikant_diff, 
                     y0=state0, 
                     t_span=(0.0, n_timesteps*h),
                     dense_output=True)
    
    return sol.sol(t).T, t