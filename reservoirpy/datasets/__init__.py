"""ReservoirPy toy and life-sized datasets.

Chaotic timeseries
------------------
    Timeseries expressing a chaotic behaviour, generated at will.

    All timeseries defined by differential equations on a
    continuous space are approximated using 4-5th order
    Runge-Kuta method [#]_, either homemade (for Mackey-Glass timeseries)
    or from Scipy `solve_ivp`_ tool.

    Available chaotic attractors:

    **Discrete timeseries**

    - :py:func:`logistic_map`: Logistic map timeseries.

    - :py:func:`henon_map`: Hénon map timeseries.

    **Approximations of continuous timeseries**

    - :py:func:`mackey_glass`: Mackey-Glass delayed differential equations timeseries.

    - py:func:`lorenz`: Lorenz system timeseries.

    - :py:func:`multiscroll`: Double scroll attractor timeseries.

    - :py:func:`rabinovich_fabrikant`: Rabinovitch-Fabrikant differential
      equations timeseries.

References
----------
    .. [#] `Runge–Kutta methods
           <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_
           on Wikipedia.


.. _solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""
from typing import Union

import numpy as np

from ._chaos import henon_map
from ._chaos import logistic_map
from ._chaos import lorenz
from ._chaos import multiscroll
from ._chaos import rabinovich_fabrikant
from ._chaos import mackey_glass

from ._seed import set_seed, get_seed


__all__ = [
    "henon_map",
    "logistic_map",
    "lorenz",
    "mackey_glass",
    "multiscroll",
    "rabinovich_fabrikant",
    "set_seed", "get_seed",
    "to_forecasting"
]


def to_forecasting(timeseries: np.ndarray,
                   forecast: int = 1,
                   axis: Union[int, float] = 0,
                   test_size: int = None):
    """Split a timeseries for forecasting tasks.

    Transform a timeseries :math:`X` into a series of
    input values :math:`X_t` and a series of output values
    :math:`X_{t+\\mathrm{forecast}}`.

    It is also possible to split the timeseries between training
    timesteps and testing timesteps.

    Parameters
    ----------
    timeseries : np.ndarray
        Timeseries to split.
    forecast : int, optional
        Number of time lag steps between
        the timeseries :math:`X_t` and the timeseries
        :math:`X_{t+\\mathrm{forecast}}`, by default 1,
        i.e. returns two timeseries with a time difference
        of 1 timesteps.
    axis : int, optional
        Time axis of the timeseries, by default 0
    test_size : int or float, optional
        If set, will also split the timeseries
        into a training phase and a testing phase of
        ``test_size`` timesteps. Can also be specified
        as a float ratio, by default None

    Returns
    -------
    tuple of numpy.ndarray
        :math:`X_t` and :math:`X_{t+\\mathrm{forecast}}`.

        If ``test_size`` is specified, will return:
        :math:`X_t`, :math:`X_t^{test}`,
        :math:`X_{t+\\mathrm{forecast}}`, :math:`X_{t+\\mathrm{forecast}}^{test}`.

        The size of the returned timeseries is therefore the size of
        :math:`X` minus the forecasting length ``forecast``.

    Raises
    ------
    ValueError
        If ``test_size`` is a float, it must be in [0, 1[.
    """

    series_ = np.moveaxis(timeseries.view(), axis, 0)
    time_len = series_.shape[0]

    if test_size is not None:
        if isinstance(test_size, float) and test_size < 1 and test_size >= 0:
            test_len = round(time_len * test_size)
        elif isinstance(test_size, int):
            test_len = test_size
        else:
            raise ValueError("invalid test_size argument: "
                             "test_size can be an integer or a float "
                             f"in [0, 1[, but is {test_size}.")
    else:
        test_len = 0

    X = series_[:-forecast]
    y = series_[forecast:]

    if test_len > 0:
        X_t = X[-test_len:]
        y_t = y[-test_len:]
        X = X[:-test_len]
        y = y[:-test_len]

        X = np.moveaxis(X, 0, axis)
        X_t = np.moveaxis(X_t, 0, axis)
        y = np.moveaxis(y, 0, axis)
        y_t = np.moveaxis(y_t, 0, axis)

        return X, X_t, y, y_t

    return np.moveaxis(X, 0, axis), np.moveaxis(y, 0, axis)
