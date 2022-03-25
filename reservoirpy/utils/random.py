# Author: Nathan Trouvain at 06/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Union

import numpy as np
from numpy.random import MT19937, Generator, RandomState, default_rng

__SEED = None
__global_rg = default_rng()


def set_seed(seed):
    global __SEED
    global __global_rg
    if type(seed) is int:
        __SEED = seed
        __global_rg = default_rng(__SEED)
        np.random.seed(__SEED)
    else:
        raise TypeError(f"Random seed must be an integer, not {type(seed)}")


def rand_generator(seed: Union[int, Generator, RandomState] = None) -> Generator:
    if seed is None:
        return __global_rg
    # provided to support legacy RandomState generator
    # of Numpy. It is not the best thing to do however
    # and recommend the user to keep using integer seeds
    # and proper Numpy Generator API.
    if isinstance(seed, RandomState):
        mt19937 = MT19937()
        mt19937.state = seed.get_state()
        return Generator(mt19937)

    if isinstance(seed, Generator):
        return seed
    else:
        return default_rng(seed)


def noise(dist="normal", shape=1, gain=1.0, seed=None, **kwargs):
    if abs(gain) > 0.0:
        rng = rand_generator(seed)
        return gain * getattr(rng, dist)(**kwargs, size=shape)
    else:
        return np.zeros(shape)
