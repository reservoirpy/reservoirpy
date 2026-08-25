# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

import reservoirpy as rpy
from reservoirpy.nodes import (
    ES2N,
    LIF,
    LMS,
    NVAR,
    RLS,
    Identity,
    Input,
    IPReservoir,
    LocalPlasticityReservoir,
    Output,
    ReLU,
    Reservoir,
    Ridge,
    ScikitLearnNode,
    Sigmoid,
    Softmax,
    Softplus,
    Tanh,
)

NODES = [
    (ES2N, dict(units=10, seed=1)),
    (IPReservoir, dict(units=10, seed=2)),
    (Identity, dict()),
    (Input, dict()),
    (LIF, dict(units=10, seed=3)),
    (LMS, dict(Wout=np.ones((1, 1)), bias=np.ones((1,)))),
    (LocalPlasticityReservoir, dict(units=10, seed=4)),
    (NVAR, dict(delay=3, order=2)),
    (Output, dict()),
    (RLS, dict(Wout=np.ones((1, 1)), bias=np.ones((1,)))),
    (ReLU, dict()),
    (Reservoir, dict(units=10, seed=0)),
    (Ridge, dict(Wout=np.ones((1, 1)), bias=np.ones((1,)))),
    (ScikitLearnNode, dict(model=LinearRegression)),
    (Sigmoid, dict()),
    (Softmax, dict()),
    (Softplus, dict()),
    (Tanh, dict()),
]


@pytest.mark.parametrize("node_class, kwargs", NODES)
def test_multiseries(node_class, kwargs):
    rpy.set_seed(0)

    xs = np.arange(10).reshape(2, 5, 1)

    node = node_class(**kwargs)
    node2 = node_class(**kwargs)
    node3 = node_class(**kwargs)

    if node_class == ScikitLearnNode:
        node.fit(xs, xs)
        node2.fit(xs, xs)
        node3.fit(xs, xs)

    ys = node.run(xs)
    y0 = node2.run(xs[0])
    y1 = node3.run(xs[1])

    assert_allclose(ys[0], y0)
    assert_allclose(ys[1], y1)
