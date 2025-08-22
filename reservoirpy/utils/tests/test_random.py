# Author: Nathan Trouvain at 25/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_equal

from reservoirpy.utils.random import noise, rand_generator, set_seed


def test_set_seed():
    set_seed(45)
    from reservoirpy.utils.random import __SEED

    assert __SEED == 45

    with pytest.raises(TypeError):
        set_seed("foo")


def test_random_generator_from_seed():

    gen1 = rand_generator(123)
    gen2 = np.random.default_rng(123)

    assert gen1.integers(1000) == gen2.integers(1000)


def test_noise():

    rng = np.random.default_rng(123)

    a = noise(rng, gain=0.0)
    assert_equal(a, np.zeros((1,)))

    rng = np.random.default_rng(123)

    a = noise(rng, dist="uniform", gain=2.0)
    b = 2.0 * np.random.default_rng(123).uniform()

    assert_equal(a, b)

    a = noise(rng, dist="uniform", gain=2.0)
    b = noise(rng, dist="uniform", gain=2.0)

    assert np.any(np.abs(a - b) > 1e-8)
