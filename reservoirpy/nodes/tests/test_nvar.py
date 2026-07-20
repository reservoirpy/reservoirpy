# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from math import comb

import numpy as np

from reservoirpy.nodes import NVAR


def _get_output_dim(input_dim, delay, order):
    linear_dim = delay * input_dim
    nonlinear_dim = comb(linear_dim + order - 1, order)
    return int(linear_dim + nonlinear_dim)


def test_nvar():
    node = NVAR(3, 2)

    data = np.ones((10,))
    res = node(data)

    assert node.store is not None
    assert node.strides == 1
    assert node.delay == 3
    assert node.order == 2
    assert node.input_dim == 10
    assert node.output_dim == _get_output_dim(10, 3, 2)
    assert res.shape == (_get_output_dim(10, 3, 2),)

    data = np.ones((1000, 10))
    res = node.run(data)

    assert res.shape == (1000, _get_output_dim(10, 3, 2))


def test_nvar_multiseries_resets_store():
    rng = np.random.default_rng(seed=0)
    xs = rng.normal(size=(3, 12, 2))

    multi = np.asarray(NVAR(2, 2).run(xs))
    # each series in a batched run must start from a fresh delay store, so it
    # must equal its own fresh single-series run
    for i in range(xs.shape[0]):
        single = np.asarray(NVAR(2, 2).run(xs[i]))
        assert np.abs(multi[i] - single).max() < 1e-6
