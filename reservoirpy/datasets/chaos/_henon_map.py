import numpy as np 


def henon_map(n_timesteps: int, a: float=1.4, b: float=0.3, state0: list=[0.0, 0.0])->np.ndarray:
    """Hénon map timeseries:
    .. math::
        x(n+1) = 1 - ax(n)^2 + y(n)
        y(n+1) = bx(n)

    Args:
        n_timesteps (int): 
        a (float, optional): a parameter of the Hénon map. Defaults to 1.4 (classic chaotic Hénon map).
        b (float, optional): b parameter of the Hénon map. Defaults to 0.3 (classic chaotic Hénon map).
        state0 (list, optional): Initial condition of the timeseries. Defaults to [0.5, 0.5].
        
    Returns:
        np.ndarray: Logistic map timeseries.
    """
    states = np.zeros((n_timesteps, 2))
    states[0] = np.asarray(state0)

    for i in range(1, n_timesteps):
        states[i][0] = 1 - a*states[i-1][0]**2 + states[i-1][1]
        states[i][1] = b*states[i-1][0]

    return states