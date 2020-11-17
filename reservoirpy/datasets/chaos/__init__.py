"""ReservoirPy chaotic timeseries generator.

All timeseries defined by differential equations on a continuous space are approximated
using 4-5th order Runge-Kuta method, either homemade (for Mackey-Glass timeseries) 
or from Scipy solve_ivp tool.

Available chaotic attractors:

    Discrete timeseries:
        logistic_map: Logistic map timeseries.
        henon_map: Hénon map timeseries.
        
    Approximations of continuous timeseries:
        mackey_glass: Mackey-Glass delayed differential equations timeseries.
        lorenz: Lorenz system timeseries.
        multiscroll: Double scroll attractor timeseries.
        rabinovitch_fabrikant: Rabinovitch-Fabrikant differential equations timeseries.
"""
from ._mackey_glass import mackey_glass
from ._logistic_map import logistic_map
from ._lorenz import lorenz
from ._multiscroll import multiscroll
from ._rabinovitch_fabrikant import rabinovich_fabrikant
from ._henon_map import henon_map