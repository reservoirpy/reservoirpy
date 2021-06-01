# Author: Nathan Trouvain at 26/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import timeit
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse, stats
from numpy.random import Generator
from numba import jit


def _next_state(lr,
                W,
                Win,
                activation,
                noise_in,
                noise_rc,
                x: np.ndarray,
                u: np.ndarray,
                ) -> np.ndarray:

    z = W @ x \
        + Win @ (u + noise_in) \

    z = activation(z) + noise_rc

    return _leaky_integration(lr, z, x).T


def _next_state_feedback(lr,
                         W,
                         Win,
                         Wfb,
                         activation,
                         noise_in,
                         noise_rc,
                         noise_fb,
                         x: np.ndarray,
                         u: np.ndarray,
                         fb: np.ndarray) -> np.ndarray:

    z = W @ x \
        + Win @ (u + noise_in) \
        + Wfb @ (fb + noise_fb)

    z = activation(z) + noise_rc

    return _leaky_integration(lr, z, x).T


def _leaky_integration(lr: float,
                       update: np.ndarray,
                       previous) -> np.ndarray:

    return (1 - lr) * previous + lr * update


def _rnn(cell):
    ...

