"""
======================================
Datasets (:mod:`reservoirpy.datasets`)
======================================

.. currentmodule:: reservoirpy.datasets

Chaotic timeseries on continuous time
=====================================

Timeseries expressing a chaotic behaviour and defined on a continuous
time axis, generated at will.

All timeseries defined by differential equations on a
continuous space are by default approximated using 4-5th order
Runge-Kuta method [1]_, either homemade (for Mackey-Glass timeseries)
or from Scipy :py:func:`scipy.integrate.solve_ivp` tool.

.. autosummary::
   :toctree: generated/

    mackey_glass - Mackey-Glass delayed differential equations timeseries.
    lorenz - Lorenz system timeseries.
    multiscroll - Multi scroll attractor timeseries.
    doublescroll - Double scroll attractor timeseries.
    rabinovich_fabrikant - Rabinovitch-Fabrikant differential equations timeseries.

Chaotic timeseries on discrete time
===================================

Timeseries expressing a chaotic behaviour and defined on a discrete
time axis, generated at will.

Discrete timeseries are defined using recurrent time-delay relations.

.. autosummary::
    :toctree: generated/

    logistic_map - Logistic map timeseries
    henon_map - Hénon map timeseries
    narma - NARMA timeseries


Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    to_forecasting - Timeseries splitting utility
    set_seed - Change random seed for dataset generation
    get_seed - Return random seed used for dataset generation


References
==========

    .. [1] `Runge–Kutta methods
           <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_
           on Wikipedia.
    .. [2] M. Hénon, ‘A two-dimensional mapping with a strange
           attractor’, Comm. Math. Phys., vol. 50, no. 1, pp. 69–77, 1976.
    .. [3] `Hénon map <https://en.wikipedia.org/wiki/H%C3%A9non_map>`_
           on Wikipédia
    .. [4] R. M. May, ‘Simple mathematical models with very
           complicated dynamics’, Nature, vol. 261, no. 5560,
           Art. no. 5560, Jun. 1976, doi: 10.1038/261459a0.
    .. [5] `Logistic map <https://en.wikipedia.org/wiki/Logistic_map>`_
           on Wikipédia
    .. [6] E. N. Lorenz, ‘Deterministic Nonperiodic Flow’,
           Journal of the Atmospheric Sciences, vol. 20, no. 2,
           pp. 130–141, Mar. 1963,
           doi: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;
    .. [7] `Lorenz system <https://en.wikipedia.org/wiki/Lorenz_system>`_
           on Wikipedia.
    .. [8] M. C. Mackey and L. Glass, ‘Oscillation and chaos in
           physiological control systems’, Science, vol. 197, no. 4300,
           pp. 287–289, Jul. 1977, doi: 10.1126/science.267326.
    .. [9] `Mackey-Glass equations
            <https://en.wikipedia.org/wiki/Mackey-Glass_equations>`_
            on Wikipedia.
    .. [10] G. Chen and T. Ueta, ‘Yet another chaotic attractor’,
       Int. J. Bifurcation Chaos, vol. 09, no. 07, pp. 1465–1466,
       Jul. 1999, doi: 10.1142/S0218127499001024.
    .. [11] `Chen double scroll attractor
           <https://en.wikipedia.org/wiki/Multiscroll_attractor
           #Chen_attractor>`_
           on Wikipedia.
    .. [12] M. I. Rabinovich and A. L. Fabrikant,
       ‘Stochastic self-modulation of waves in
       nonequilibrium media’, p. 8, 1979.
    .. [13] `Rabinovich-Fabrikant equations
           <https://en.wikipedia.org/wiki/Rabinovich%E2%80
           %93Fabrikant_equations>`_
           on Wikipedia.
    .. [14] A. F. Atiya and A. G. Parlos, ‘New results on recurrent
       network training: unifying the algorithms and accelerating
       convergence,‘ in IEEE Transactions on Neural Networks,
       vol. 11, no. 3, pp. 697-709, May 2000,
       doi: 10.1109/72.846741.
    .. [15] B.Schrauwen, M. Wardermann, D. Verstraeten, J. Steil,
           D. Stroobandt, ‘Improving reservoirs using intrinsic
           plasticity‘,
           Neurocomputing, 71. 1159-1171, 2008,
           doi: 10.1016/j.neucom.2007.12.020.
"""
from typing import Union

import numpy as np

from ._chaos import henon_map
from ._chaos import logistic_map
from ._chaos import lorenz
from ._chaos import multiscroll
from ._chaos import rabinovich_fabrikant
from ._chaos import mackey_glass
from ._chaos import narma
from ._chaos import doublescroll

from ._seed import set_seed, get_seed


__all__ = [
    "henon_map",
    "logistic_map",
    "lorenz",
    "mackey_glass",
    "multiscroll",
    "rabinovich_fabrikant",
    "narma", "doublescroll",
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
