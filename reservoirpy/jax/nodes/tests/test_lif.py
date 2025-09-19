import jax.numpy as jnp
from numpy.testing import assert_array_equal

from reservoirpy.datasets import mackey_glass
from reservoirpy.jax.nodes import LIF


def test_lif():
    n_timesteps = 140
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

    x = jnp.ones((n_timesteps, 1))
    y = lif.run(x)

    assert y.shape == (n_timesteps, neurons)
    assert_array_equal(jnp.sort(jnp.unique(y)), jnp.array([0.0, 1.0]))
