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
    lorenz96 - Lorenz 1996 attractor.
    rossler - Rossler attractor.
    kuramoto_sivashinsky - Kuramoto-Sivashinsky oscillators.
    mso - Multiple superimposed oscillators.
    mso2 - Multiple superimposed oscillators with 2 frequencies.
    mso8 - Multiple superimposed oscillators with 8 frequencies.

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


Classification/pattern recognition tasks
========================================

Classified datasets of timeseries.

.. autosummary::
    :toctree: generated/

    japanese_vowels - Japanese vowels task


Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    santafe_laser - Santa-Fe laser dataset
    to_forecasting - Timeseries splitting utility
    one_hot_encode - One-hot encoding utility
    from_aeon_classification - Conversion from Aeon's format
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
           on Wikipedia
    .. [4] R. M. May, ‘Simple mathematical models with very
           complicated dynamics’, Nature, vol. 261, no. 5560,
           Art. no. 5560, Jun. 1976, doi: 10.1038/261459a0.
    .. [5] `Logistic map <https://en.wikipedia.org/wiki/Logistic_map>`_
           on Wikipedia
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
            #Chen_attractor>`_ on Wikipedia.
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
    .. [16] M. Kudo, J. Toyama and M. Shimbo. (1999).
            "Multidimensional Curve Classification Using Passing-Through Regions".
            Pattern Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.
    .. [17] Lorenz, E. N. (1996, September).
            Predictability: A problem partly solved. In Proc.
            Seminar on predictability (Vol. 1, No. 1).
    .. [18] O.E. Rössler, "An equation for continuous chaos", Physics Letters A,
            vol 57, Issue 5, Pages 397-398, ISSN 0375-9601, 1976,
            https://doi.org/10.1016/0375-9601(76)90101-8.
    .. [19] Kuramoto, Y. (1978). Diffusion-Induced Chaos in Reaction Systems.
            Progress of Theoretical Physics Supplement, 64, 346–367.
            https://doi.org/10.1143/PTPS.64.346
    .. [20] Sivashinsky, G. I. (1977). Nonlinear analysis of hydrodynamic
            instability in laminar flames—I. Derivation of basic equations.
            Acta Astronautica, 4(11), 1177–1206.
            https://doi.org/10.1016/0094-5765(77)90096-0
    .. [21] Sivashinsky, G. I. (1980). On Flame Propagation Under Conditions
            of Stoichiometry. SIAM Journal on Applied Mathematics, 39(1), 67–82.
            https://doi.org/10.1137/0139007
    .. [22] Jaeger, H. (2004b). Seminar slides. (Online) Available
            http://www.faculty.iu-bremen.de/hjaeger/courses/SeminarSpring04/ESNStandardSlides.pdf.
    .. [24] Roeschies, B., & Igel, C. (2010). Structure optimization of
            reservoir networks. Logic Journal of IGPL, 18(5), 635-669.
    .. [25] Weigend, A. S. (2018). Time series prediction: forecasting the
            future and understanding the past. Routledge.
"""

from typing import Optional, Union

import numpy as np

from ._chaos import (
    doublescroll,
    henon_map,
    kuramoto_sivashinsky,
    logistic_map,
    lorenz,
    lorenz96,
    mackey_glass,
    multiscroll,
    narma,
    rabinovich_fabrikant,
    rossler,
)
from ._japanese_vowels import japanese_vowels
from ._santafe_laser import santafe_laser
from ._seed import get_seed, set_seed
from ._utils import from_aeon_classification, one_hot_encode

__all__ = [
    "henon_map",
    "logistic_map",
    "lorenz",
    "mackey_glass",
    "multiscroll",
    "rabinovich_fabrikant",
    "narma",
    "doublescroll",
    "japanese_vowels",
    "lorenz96",
    "rossler",
    "kuramoto_sivashinsky",
    "set_seed",
    "get_seed",
    "to_forecasting",
    "mso",
    "mso2",
    "mso8",
    "from_aeon_classification",
    "one_hot_encode",
    "santafe_laser",
]


def to_forecasting(
    timeseries: np.ndarray,
    forecast: int = 1,
    axis: int = 0,
    test_size: Optional[Union[int, float]] = None,
):
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

    Examples
    --------

    >>> from reservoirpy.datasets import to_forecasting
    >>> from numpy import linspace, sin
    >>> t = linspace(0, 500, 1000)
    >>> timeseries = sin(0.2*t) + sin(0.311*t)
    >>> X_train, X_test, y_train, y_test = to_forecasting(timeseries, forecast=10, test_size=200)

    .. plot::

        from reservoirpy.datasets import to_forecasting
        from numpy import linspace, sin
        t = linspace(0, 500, 1000); h=10
        timeseries = sin(0.2*t) + sin(0.311*t)
        X_train, X_test, y_train, y_test = to_forecasting(timeseries, forecast=h, test_size=200)
        plt.figure(figsize=(12, 3))
        plt.plot(t[:-200-h], X_train, label="Training data")
        plt.plot(t[:-200-h], y_train, label="Training ground truth")
        plt.plot(t[-200:], X_test, label="Testing data")
        plt.plot(t[-200:], y_test, label="Testing ground truth")
        plt.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
        plt.show()

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
            raise ValueError(
                "invalid test_size argument: "
                "test_size can be an integer or a float "
                f"in [0, 1[, but is {test_size}."
            )
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


def mso(n_timesteps: int, freqs: list, normalize: bool = True):
    """Multiple superimposed oscillator task [22]_

    This task is usually performed to evaluate a model resistance
    to perturbations and its asymptotic stability. See, for example: [23]_.

    .. math::

        MSO(t) = \\sum_{i} sin(f_i t)

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to generate
    freqs : list
        Frequencies of the sin waves
    normalize : bool, optional
        If `True`, scales the range of values in [-1, 1]

    Returns
    -------
    array of shape (n_timesteps, 1)
        Multiple superimposed oscillator timeseries.

    Examples
    --------

    >>> from reservoirpy.datasets import mso
    >>> timeseries = mso(200, freqs=[0.2, 0.44])
    >>> print(timeseries.shape)
    (200, 1)

    .. plot::

        from reservoirpy.datasets import mso
        timeseries = mso(200, freqs=[0.2, 0.44])
        plt.figure(figsize=(12, 4))
        plt.plot(timeseries)
        plt.show()

    References
    ----------
    .. [22] Jaeger, H. (2004b). Seminar slides. (Online) Available http://www.faculty.
            iu-bremen.de/hjaeger/courses/SeminarSpring04/ESNStandardSlides.pdf.
    .. [23] Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007).
            Optimization and applications of echo state networks with leaky-integrator
            neurons. Neural networks, 20(3), 335-352.
    """
    t = np.arange(n_timesteps).reshape(n_timesteps, 1)
    y = np.zeros((n_timesteps, 1))

    for f in freqs:
        y += np.sin(f * t)

    if normalize:
        return (2 * y - y.min() - y.max()) / (y.max() - y.min())
    else:
        return y


def mso2(n_timesteps: int, normalize: bool = True):
    """Multiple superimposed oscillator task with 2 frequencies [22]_

    This is the $MSO$ task with 2 frequencies: :math:`f_1=0.2` and
    :math:`f_1=0.311`, as first described in [22]_.

    .. math::

        MSO(t) = sin(0.2 t) + sin(0.311 t)

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to generate
    normalize : bool
        If `True`, scales the range of values in [-1, 1]

    Returns
    -------
    array of shape (n_timesteps, 1)
        Multiple superimposed oscillator timeseries.

    Examples
    --------

    >>> from reservoirpy.datasets import mso2
    >>> timeseries = mso(500)
    >>> print(timeseries.shape)
    (500, 1)

    .. plot::

        from reservoirpy.datasets import mso2
        timeseries = mso2(500)
        plt.figure(figsize=(12, 4))
        plt.plot(timeseries)
        plt.show()

    References
    ----------
    .. [22] Jaeger, H. (2004b). Seminar slides. (Online) Available http://www.faculty.
            iu-bremen.de/hjaeger/courses/SeminarSpring04/ESNStandardSlides.pdf.
    """
    return mso(n_timesteps=n_timesteps, freqs=[0.2, 0.311], normalize=normalize)


def mso8(n_timesteps: int, normalize: bool = True):
    """Multiple superimposed oscillator task with 8 frequencies [22]_ [24]_

    This is the $MSO$ task with 8 frequencies: :math:`0.2, 0.311, 0.42,
    0.51, 0.63, 0.74, 0.85, 0.97`, as first described in [24]_.

    .. math::

        MSO(t) = \\sum_{i=1}^{8} sin(f_i t)

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps to generate
    normalize : bool
        If `True`, scales the range of values in [-1, 1]

    Returns
    -------
    array of shape (n_timesteps, 1)
        Multiple superimposed oscillator timeseries.

    Examples
    --------

    >>> from reservoirpy.datasets import mso8
    >>> timeseries = mso(500)
    >>> print(timeseries.shape)
    (500, 1)

    .. plot::

        from reservoirpy.datasets import mso8
        timeseries = mso8(500)
        plt.figure(figsize=(12, 4))
        plt.plot(timeseries)
        plt.show()

    References
    ----------
    .. [22] Jaeger, H. (2004b). Seminar slides. (Online) Available
            http://www.faculty.iu-bremen.de/hjaeger/courses/SeminarSpring04/ESNStandardSlides.pdf.
    .. [24] Roeschies, B., & Igel, C. (2010). Structure optimization of reservoir networks.
            Logic Journal of IGPL, 18(5), 635-669.
    """
    return mso(
        n_timesteps=n_timesteps,
        freqs=[0.2, 0.311, 0.42, 0.51, 0.63, 0.74, 0.85, 0.97],
        normalize=normalize,
    )
