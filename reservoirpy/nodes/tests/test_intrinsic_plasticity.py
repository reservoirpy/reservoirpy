# Author: Nathan Trouvain at 24/02/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..intrinsic_plasticity import IPReservoir


def test_ip_init():
    res = IPReservoir(100, input_dim=5)
    x = np.ones((10, 5))

    res.initialize(x)

    assert res.W.shape == (100, 100)
    assert res.Win.shape == (100, 5)
    assert_allclose(res.a, np.ones((100,)))
    assert_allclose(res.b, np.zeros((100,)))

    res = IPReservoir(100)

    out = res.run(x)

    assert out.shape == (10, 100)
    assert res.W.shape == (100, 100)
    assert res.Win.shape == (100, 5)
    assert_allclose(res.a, np.ones((100,)))
    assert_allclose(res.b, np.zeros((100,)))

    with pytest.raises(ValueError):
        res = IPReservoir(100, activation="identity")


def test_intrinsic_plasticity():

    x = np.random.normal(size=(100, 5))
    X = [x[:10], x[:20]]

    res = IPReservoir(100, activation="tanh", epochs=2)

    res.fit(x)
    res.fit(X)

    assert res.a.shape == (100,)
    assert res.b.shape == (100,)

    res = IPReservoir(100, activation="sigmoid", epochs=1, mu=0.1)

    res.fit(x)
    res.fit(X)

    assert res.a.shape == (100,)
    assert res.b.shape == (100,)

    res.fit(x)
    res.fit(X)

    assert res.a.shape == (100,)
    assert res.b.shape == (100,)


def test_ip_model():
    x = np.random.normal(size=(100, 5))
    y = np.random.normal(size=(100, 2))
    X = [x[:10], x[:20]]
    Y = [y[:10], y[:20]]

    res = IPReservoir(100, activation="tanh", epochs=2, seed=1234)

    res.fit(X, Y)

    res2 = IPReservoir(100, activation="tanh", epochs=2, seed=1234)
    res2.fit(X)

    assert_allclose(res.a, res2.a)
    assert_allclose(res.b, res2.b)
