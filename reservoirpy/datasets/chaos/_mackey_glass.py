from typing import Tuple

import collections

import numpy as np


DEFAULT_SEED = 5555


def mackey_glass(n_timesteps: int, 
                 tau: int=17, 
                 a: float=0.2, 
                 b: float=0.1, 
                 n: int=10, 
                 x0: float=1.2, 
                 h: float=1.0, 
                 seed: int=None) -> Tuple[np.ndarray, np.ndarray]:
    """Mackey-Glass timeseries, computed from the Mackey-Glass delay differential equation:
    .. math:: 
        \dv{x}{t} = \frac{ax(t-\tau)}{1+x(t-\tau)^n} - bx(t)
    
    Args:
        n_timesteps (int): Number of timesteps to compute.
        tau (int, optional): Time delay of Mackey-Glass equation. Defaults to 17.
        a (float, optional): Parameter a of equation.. Defaults to 0.2.
        b (float, optional): Parameter b of equation. Defaults to 0.1.
        n (int, optional): Parameter n of equation. Defaults to 10.
        x0 (float, optional): Initial value of the timeseries. Defaults to 1.2.
        h (float, optional): Time delta for the Runge-Kuta method. Can be assimilated
                             to the number of discrete point computed per timestep. Defaults to 1.0.
        seed (int, optional): Random state seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray, np.ndarray: Computed timeseries and corresponding timesteps.
    """
    
    if seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random.RandomState(DEFAULT_SEED)
    
    
    time = 0
    history_length = int(np.floor(tau/h))
    history = collections.deque(x0 * np.ones(history_length) + \
                                0.2 * (rs.rand(history_length) - 0.5))
    xt = x0

    X = np.zeros(n_timesteps)
    T = np.zeros(n_timesteps)

    for i in range(0, n_timesteps):
        X[i] = xt
        
        if tau == 0:
            xtau = 0.0
        else:
            xtau = history.popleft()
            history.append(xt)

        xth = _mc_rk4(xt, xtau, a=a, b=b, n=n)

        time = time + h
        T[i] = time
        xt = xth
    
    return X, T


def _mc_eq(xt, xtau, a=0.2, b=0.1, n=10):
    """
    Mackey-Glass time delay diffential equation, at values x(t) and x(t-tau):
    .. math:: \dv{x}{t} = \frac{ax(t-\tau)}{1+x(t-tau)^n} - bx(t)
    
    Arguments:
        xt {float} -- Value x(t) of the Mackey-Glass timeseries.
        xtau {float} -- Value x(t - tau) of the Mackey-Glass timeseries.
    Keywork arguments:
        a {float} -- Parameter a of the Mackey-Glass delay differential equation.
        b {float} -- Parameter b of the Mackey-Glass delay differential equation.
        n {float} -- Parameter n of the Mackey-Glass delay differential equation.
    Returns:
        {float} -- The value of the differential at a point t.
    """
    return -b*xt + a*xtau / (1+xtau**n)

    
def _mc_rk4(xt, xtau, a, b, n, h=1.0):
    """
    Runge-Kuta method (RK4) for Mackey-Glass timeseries discretization.
    
    Arguments:
        xt {float} -- Value x(t) of the Mackey-Glass timeseries.
        xtau {float} -- Value x(t - tau) of the Mackey-Glass timeseries.
    Keyword arguments:
        h {float} -- Time delta for the Runge-Kuta method. Can be assimilated
                     to the number of discrete point computed per timestep.
    Returns:
        {float} -- Runge-Kuta approximation of the timeseries at point t.
    """
    k1 = h * _mc_eq(xt, xtau, a, b, n)
    k2 = h * _mc_eq(xt + 0.5*k1, xtau, a, b, n)
    k3 = h * _mc_eq(xt + 0.5*k2, xtau, a, b, n)
    k4 = h * _mc_eq(xt + k3, xtau, a, b, n)
    
    return xt + k1/6 + k2/3 + k3/3 + k4/6