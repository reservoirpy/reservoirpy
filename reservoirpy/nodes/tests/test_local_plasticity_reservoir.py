import numpy as np
import pytest
from numpy.testing import assert_allclose

from ...mat_gen import ring
from .. import Ridge
from ..local_plasticity_reservoir import LocalPlasticityReservoir


def test_lsp_init():
    x = np.ones((10, 5))
    res = LocalPlasticityReservoir(100, input_dim=5)

    res.initialize(x)

    assert res.W.shape == (100, 100)
    assert res.Win.shape == (100, 5)

    res = LocalPlasticityReservoir(100)

    out = res.run(x)

    assert out.shape == (10, 100)
    assert res.W.shape == (100, 100)
    assert res.Win.shape == (100, 5)

    with pytest.raises(ValueError):
        _ = LocalPlasticityReservoir(local_rule="oja")


def test_lsp_rules():
    x = np.ones((10, 5))

    res = LocalPlasticityReservoir(100, local_rule="oja")
    _ = res.fit(x)
    res = LocalPlasticityReservoir(100, local_rule="anti-oja")
    _ = res.fit(x)
    res = LocalPlasticityReservoir(100, local_rule="hebbian")
    _ = res.fit(x)
    res = LocalPlasticityReservoir(100, local_rule="anti-hebbian")
    _ = res.fit(x)
    res = LocalPlasticityReservoir(100, local_rule="bcm")
    _ = res.fit(x)

    with pytest.raises(ValueError):
        res = LocalPlasticityReservoir(100, local_rule="anti-bcm")

    with pytest.raises(ValueError):
        res = LocalPlasticityReservoir(100, local_rule="anti_oja")


def test_local_synaptic_plasticity():
    rng = np.random.default_rng(seed=1)
    x = rng.normal(size=(100, 5))
    X = [x[:10], x[:20]]

    res = LocalPlasticityReservoir(100, local_rule="hebbian", epochs=2, seed=0)

    res.fit(x)
    res.fit(X)

    assert res.W.shape == (100, 100)

    res = LocalPlasticityReservoir(
        100, local_rule="oja", epochs=10, eta=1e-3, synapse_normalization=True, seed=10
    )
    res.initialize(x)

    initial_Wvals = res.W.data.copy()

    res.fit(x)
    res.fit(X)

    assert not np.allclose(initial_Wvals, res.W.data)


def test_lsp_matrices():
    rng = np.random.default_rng(seed=2504)
    x = rng.normal(size=(100, 5))

    W = ring(10, 10)

    lspres_ring = LocalPlasticityReservoir(W=W, seed=2504)
    lspres_rand = LocalPlasticityReservoir(units=10, seed=2504)

    lspres_ring.fit(x)
    lspres_rand.fit(x)

    assert not np.allclose(lspres_ring.W.toarray(), lspres_rand.W.toarray())

    # test dense matrix
    W = rng.normal(size=(10, 10))
    res = LocalPlasticityReservoir(W=W)
    res.fit(x)
