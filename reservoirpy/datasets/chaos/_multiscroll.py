from typing import Tuple 

import numpy as np

from scipy.integrate import solve_ivp


def multiscroll(n_timesteps: int, 
            state0: list=[-0.1, 0.5, -0.6], 
            a: float=40.0, 
            b: float=3.0, 
            c: float=28.0, 
            h: float=0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Double scroll attractor timeseries, a particular case of multiscroll attractor.
    .. math::
        \dv{x}{t} = a(y - x)
    .. math::
        \dv{y}{t} = (c - a)x - xz + cy
    .. math::
        \dv{z}{t} = xy - bz

    Args:
        n_timesteps (int): Number of timesteps to generate.
        state0 (list, optional): Initial conditions of the system. Defaults to [-0.1, 0.5, -0.6].
        a (float, optional): a parameter of the system. Defaults to 40.0.
        b (float, optional): b parameter of the system. Defaults to 3.0.
        c (float, optional): c parameter of the system. Defaults to 28.0.
        h (float, optional): Control the time delta between two timesteps. Defaults to 0.01.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated timeseries and timesteps.
    """

    def multiscroll_diff(t, state):
        x, y, z = state
        dx = a * (y - x)
        dy = (c - a)*x - x*z + c*y
        dz = x*y - b*z
        return dx, dy, dz

    t = np.arange(0.0, n_timesteps * h, h)
    
    sol = solve_ivp(multiscroll_diff,
                    y0=state0,
                    t_span=(0.0, n_timesteps*h),
                    dense_output=True)

    return sol.sol(t).T, t