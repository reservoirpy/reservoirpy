# Author: Nathan Trouvain at 15/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from reservoirpy.nodes import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh


@pytest.mark.parametrize(
    "node",
    (Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),
)
def test_activation_step(node):
    x = np.ones((10,))
    out = node(x)
    assert out.shape == x.shape
    assert node.input_dim == x.shape[-1]
    assert node.output_dim == x.shape[-1]


@pytest.mark.parametrize(
    "node",
    (Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),
)
def test_activation_run(node):
    x = np.ones((100, 10))
    out = node.run(x)
    assert out.shape == x.shape
    assert node.input_dim == x.shape[-1]
    assert node.output_dim == x.shape[-1]

    out2 = node.step(x[-1])
    np.testing.assert_array_almost_equal(out[-1], out2)

    X = list(np.ones((7, 100, 10)))
    out = node.run(X)
    assert isinstance(out, list)
    assert out[0].shape == X[0].shape
    assert node.input_dim == X[0].shape[-1]
    assert node.output_dim == X[0].shape[-1]

    out2 = node.step(X[0][-1])
    np.testing.assert_array_almost_equal(out[0][-1], out2)
