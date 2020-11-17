import numpy as np


def logistic_map(n_timesteps: int, r: float=3.9, x0: float=0.5)->np.ndarray:
    """Logistic map timeseries:
    .. math::
        x(n+1) = rx(n)(1-x(n))

    Args:
        n_timesteps (float): 
        r (float, optional): r parameter of the logistic map equation. Value should be between
                              3.6 and 3.9 to show intersting chaotic behaviour. Values between 3.45 and 3.56
                              expose a periodic behaviour beteen a increasing number of values. Defaults to 3.9.
        x0 (float, optional): Initial condition of the timeseries. Should be in ]0; 1[. Defaults to 0.5.

    Raises:
        ValueError: if x0 is outside ]0; 1[.
        ValueError: if r is not strictly positive.

    Returns:
        np.ndarray: Logistic map timeseries.
    """
    if r>0 and 0<x0<1:
        X = np.zeros(n_timesteps)
        X[0] = x0

        for i in range(1, n_timesteps):
            X[i] = r * X[i-1] * (1-X[i-1])

        return X
    elif r<=0:
        raise ValueError("r should be positive.")
    else:
        raise ValueError("Initial condition x0 should be in ]0;1[.")