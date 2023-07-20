# Author: Nathan Trouvain at 05/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_equal

from reservoirpy import set_seed
from reservoirpy.nodes import ESN, Reservoir, Ridge


def test_esn_init():

    esn = ESN(units=100, output_dim=1, lr=0.8, sr=0.4, ridge=1e-5, Win_bias=False)

    data = np.ones((1, 10))
    res = esn(data)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.reservoir.lr == 0.8
    assert esn.reservoir.units == 100

    data = np.ones((10000, 10))
    res = esn.run(data)

    assert res.shape == (10000, 1)

    with pytest.raises(ValueError):
        esn = ESN(units=100, output_dim=1, learning_method="foo")

    with pytest.raises(ValueError):
        esn = ESN(units=100, output_dim=1, reservoir_method="foo")


def test_esn_init_from_obj():

    res = Reservoir(100, lr=0.8, sr=0.4, input_bias=False)
    read = Ridge(1, ridge=1e-5)

    esn = ESN(reservoir=res, readout=read)

    data = np.ones((1, 10))
    res = esn(data)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.reservoir.lr == 0.8
    assert esn.reservoir.units == 100

    data = np.ones((10000, 10))
    res = esn.run(data)

    assert res.shape == (10000, 1)


def test_esn_states():
    res = Reservoir(100, lr=0.8, sr=0.4, input_bias=False)
    read = Ridge(1, ridge=1e-5)

    esn = ESN(reservoir=res, readout=read)

    data = np.ones((2, 10, 10))
    out = esn.run(data, return_states="all")

    assert out["reservoir"][0].shape == (10, 100)
    assert out["readout"][0].shape == (10, 1)

    out = esn.run(data, return_states=["reservoir"])

    assert out["reservoir"][0].shape == (10, 100)

    s_reservoir = esn.state()
    assert_equal(s_reservoir, res.state())

    s_readout = esn.state(which="readout")
    assert_equal(s_readout, read.state())

    with pytest.raises(ValueError):
        esn.state(which="foo")


def test_esn_feedback():

    esn = ESN(units=100, output_dim=5, lr=0.8, sr=0.4, ridge=1e-5, feedback=True)

    data = np.ones((1, 10))
    res = esn(data)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.readout.Wout.shape == (100, 5)
    assert res.shape == (1, 5)
    assert esn.reservoir.Wfb is not None
    assert esn.reservoir.Wfb.shape == (100, 5)


@pytest.mark.parametrize("backend", ("loky", "multiprocessing", "threading"))
def test_esn_parallel_fit_reproducibility(backend):

    for i in range(10):
        set_seed(45)

        esn = ESN(
            units=100,
            lr=0.8,
            sr=0.4,
            ridge=1e-5,
            feedback=True,
            workers=-1,
            backend=backend,
        )

        X, Y = np.ones((10, 100, 10)), np.ones((10, 100, 5))
        esn.fit(X, Y)

        assert esn.reservoir.W.shape == (100, 100)
        assert esn.reservoir.Win.shape == (100, 10)
        assert esn.readout.Wout.shape == (100, 5)

        assert esn.reservoir.Wfb is not None
        assert esn.reservoir.Wfb.shape == (100, 5)

        assert np.mean(esn.readout.Wout) - 0.002418478571198347 < 1e-5


def test_hierarchical_esn_forbidden():

    esn1 = ESN(
        units=100,
        lr=0.8,
        sr=0.4,
        ridge=1e-5,
        feedback=True,
        workers=-1,
        backend="loky",
        name="E1",
    )

    esn2 = ESN(
        units=100,
        lr=0.8,
        sr=0.4,
        ridge=1e-5,
        feedback=True,
        workers=-1,
        backend="loky",
        name="E2",
    )

    # FrozenModel can't be linked (for now).
    with pytest.raises(TypeError):
        model = esn1 >> esn2
