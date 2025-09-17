# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import numpy
import pytest
from numpy.testing import assert_array_almost_equal

from reservoirpy.jax.nodes import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh


@pytest.mark.parametrize(
    "node",
    (Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),
)
def test_activation_step(node):
    x = jnp.ones((10,))
    out = node(x)
    assert out.shape == x.shape
    assert node.input_dim == x.shape[-1]
    assert node.output_dim == x.shape[-1]


@pytest.mark.parametrize(
    "node",
    (Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),
)
def test_activation_run(node):
    x = jnp.ones((100, 10))
    out = node.run(x)
    assert out.shape == x.shape
    assert node.input_dim == x.shape[-1]
    assert node.output_dim == x.shape[-1]

    out2 = node.step(x[-1])
    assert_array_almost_equal(out[-1], out2)

    rng = numpy.random.default_rng(seed=1)
    X = list(rng.normal(size=(7, 100, 10)))
    out = node.run(X)
    assert isinstance(out, list)
    assert out[0].shape == X[0].shape
    assert node.input_dim == X[0].shape[-1]
    assert node.output_dim == X[0].shape[-1]

    out2 = node.step(X[0][-1])
    assert_array_almost_equal(out[0][-1], out2)
