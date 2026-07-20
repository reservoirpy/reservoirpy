# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import numpy
from numpy.testing import assert_allclose, assert_array_equal

from reservoirpy.jax.nodes import LIF
from reservoirpy.nodes import LIF as NumpyLIF


def _dense_W(node):
    node.initialize(jnp.ones((1,)))
    W = node.W
    if hasattr(W, "todense"):
        W = W.todense()
    return numpy.asarray(W)


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


def test_lif_inhibitory_matches_numpy_backend_sparse():
    # the default connectivity produces a sparse (BCOO) W; initialization must
    # not crash and must match the numpy backend
    jax_W = _dense_W(LIF(units=20, inhibitory=0.5, seed=1234))
    numpy_W = _dense_W(NumpyLIF(units=20, inhibitory=0.5, seed=1234))
    assert (numpy_W < 0).any()
    assert_allclose(jax_W, numpy_W, atol=1e-5)
