# Author: Nathan Trouvain at 06/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Union, Callable
from functools import partial

import numpy as np
from numpy.random import Generator, MT19937, RandomState, default_rng
from scipy import stats

__SEED = None
__global_rg = default_rng()


def set_seed(seed):
    global __SEED
    global __global_rg
    if type(seed) is int:
        __SEED = seed
        __global_rg = default_rng(__SEED)
    else:
        raise TypeError(f"Random seed must be an integer, not {type(seed)}")


def rand_generator(
        seed: Union[int, Generator, RandomState] = None) -> Generator:
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


def get_rvs(dist: str,
            seed=None,
            **kwargs) -> Callable:
    # override scipy.stats uniform rvs
    # to allow user to set the distribution with
    # common low/high values and not loc/scale
    rng = rand_generator(seed)

    if dist == "uniform":
        return _uniform_rvs(**kwargs,
                            random_state=rng)
    elif dist == "bimodal":
        return _bimodal_discrete_rvs(**kwargs,
                                     random_state=rng)
    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs,
                       random_state=rng)
    else:
        raise ValueError(f"'{dist}' is not a valid distribution name. "
                         "See 'scipy.stats' for all available distributions.")


def _bimodal_discrete_rvs(value: float = 1.,
                          random_state: Union[Generator, int] = None) -> Callable:

    def rvs(size: int = 1):
        return random_state.choice([value, -value], replace=True, size=size)

    return rvs


def _uniform_rvs(low: float = -1.0,
                 high: float = 1.0,
                 random_state: Union[Generator, int] = None) -> Callable:

    distribution = getattr(stats, "uniform")
    return partial(distribution(loc=low, scale=high-low).rvs,
                   random_state=random_state)


def noise(dist="normal",
          shape=1,
          gain=1.0,
          seed=None,
          **kwargs):
    if gain > 0.0 or gain < 0.0:
        rng = rand_generator(seed)
        return gain * getattr(rng, dist)(**kwargs, size=shape)
    else:
        return np.zeros(shape)


def normal_noise(shape=1, gain=1.0, loc=0.0, scale=1.0, seed=None):
    rng = rand_generator(seed)
    return gain * rng.normal(loc, scale, size=shape)


def uniform_noise(shape=1, gain=1.0, low=0.0, high=1.0, seed=None):
    rng = rand_generator(seed)
    return gain * rng.uniform(low, high, size=shape)
