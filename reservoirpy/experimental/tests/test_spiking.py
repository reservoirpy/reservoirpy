import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ...datasets import mackey_glass
from .. import LIF


def test_lif():
    n_timesteps = 1_000
    neurons = 100

    lif = LIF(
        units=neurons,
        inhibitory=0.0,
        sr=1.0,
        lr=0.2,
        input_scaling=1.0,
        threshold=1.0,
        rc_connectivity=1.0,
    )

    x = mackey_glass(n_timesteps=n_timesteps)
    y = lif.run(x)

    assert y.shape == (n_timesteps, neurons)
    assert_array_equal(np.sort(np.unique(y)), np.array([0.0, 1.0]))
