# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
from jax.experimental import sparse as jsparse
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


def test_lif_inhibitory_dense():
    # regression test: the inhibitory-column sign flip used to be discarded
    # (jax .at[].multiply() returns a new array instead of mutating in place),
    # so 'inhibitory' had no effect at all on a dense W.
    neurons = 20
    n_inhibitory = int(0.3 * neurons)

    lif = LIF(units=neurons, inhibitory=0.3, rc_connectivity=1.0, seed=0)
    x = jnp.ones((5, 1))
    lif.run(x)

    W = lif.W
    assert (W[:, :n_inhibitory] <= 0).all()
    assert (W[:, n_inhibitory:] >= 0).all()


def test_lif_inhibitory_sparse():
    # regression test: with the default (sparse/BCOO) W, the discarded-update
    # fix alone is not enough -- .at[] does not even exist on BCOO arrays, so
    # this used to raise AttributeError regardless of the discarded-value bug.
    neurons = 20
    n_inhibitory = int(0.3 * neurons)

    lif = LIF(units=neurons, inhibitory=0.3, seed=0)  # default rc_connectivity=0.1 -> sparse

    x = jnp.ones((5, 1))
    lif.run(x)  # must not raise

    assert isinstance(lif.W, jsparse.BCOO)

    W = lif.W.todense()
    assert (W[:, :n_inhibitory] <= 0).all()
    assert (W[:, n_inhibitory:] >= 0).all()
