from typing import Tuple

import numpy as np

from scipy.integrate import solve_ivp


def lorenz(n_timesteps: int, state0: list=[1.0, 1.0, 1.0], 
           rho: float=28.0, sigma: float=10.0, beta: float=8.0/3.0, h: float=0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Lorenz system timeseries.
    .. math::
        \dv{x}{t} = \sigma (y-x)
    .. math::
        \dv{y}{t} = x(\rho - z) - y
    .. math::
        \dv{z}{t} = xy - \beta z

    Args:
        n_timesteps (int): Number of timesteps to generate.
        state0 (list, optional): Initial condition of the system. Defaults to [1.0, 1.0, 1.0].
        rho (float, optional): Rho parameter of the system. Defaults to 28.0.
        sigma (float, optional): Sigma parameter of the system. Defaults to 10.0.
        beta (float, optional): Beta parameter of the system. Defaults to 8.0/3.0.
        h (float, optional): Control the time delta between two timesteps. Defaults to 0.01.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated timeseries and timesteps.
    """
    def lorenz_diff(t, state):
        x, y, z = state 
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    t = np.arange(0, n_timesteps * h, h)
    
    sol = solve_ivp(lorenz_diff, 
                    y0=state0, 
                    t_span=(0.0, n_timesteps*h),
                    dense_output=True)
    
    return sol.sol(t).T, t