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

    :- py:func:`lorenz`: Lorenz system timeseries.

    - :py:func:`multiscroll`: Double scroll attractor timeseries.

    - :py:func:`rabinovitch_fabrikant`: Rabinovitch-Fabrikant differential
      equations timeseries.

References
----------
    .. [#] `Runge–Kutta methods
           <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_
           on Wikipedia.


.. _solve_ivp: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
"""
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
    "set_seed", "get_seed"
]
