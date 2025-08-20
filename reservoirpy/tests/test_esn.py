# Author: Nathan Trouvain at 05/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from reservoirpy import ESN
from reservoirpy.mat_gen import normal
from reservoirpy.nodes import Reservoir, Ridge


def test_esn_init():
    esn = ESN(units=100, lr=0.8, sr=0.4, ridge=1e-5)

    data = np.ones((1000, 10))
    res = esn.fit(data, data)
    data = np.ones((10,))
    res = esn(data)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.reservoir.lr == 0.8
    assert esn.reservoir.units == 100

    data = np.ones((1000, 10))
    res = esn.run(data)
    assert res.shape == (1000, 10)

    esn = ESN(units=100, lr=0.8, sr=0.4, ridge=1e-5, output_dim=7)
    with pytest.raises(ValueError):
        esn.initialize(np.ones((10, 3)), np.ones((10, 8)))


def test_esn_init_from_obj():
    res = Reservoir(100, lr=0.8, sr=0.4)
    read = Ridge(ridge=1e-5, output_dim=1)

    esn = ESN(reservoir=res, readout=read)

    x = np.ones((10, 2))
    y = np.arange(10).reshape(-1, 1)
    esn.fit(x, y)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 2)
    assert esn.reservoir.lr == 0.8
    assert esn.reservoir.units == 100

    data = np.ones((1000, 2))
    res = esn.run(data)

    assert res.shape == (1000, 1)


# def test_esn_states(): # TODO: implement a way to return reservoir activity
#     res = Reservoir(100, lr=0.8, sr=0.4)
#     read = Ridge(ridge=1e-5, output_dim=1)

#     esn = ESN(reservoir=res, readout=read)

#     data = np.ones((2, 10, 10))
#     out = esn.run(data, return_states="all")

#     assert out["reservoir"][0].shape == (10, 100)
#     assert out["readout"][0].shape == (10, 1)

#     out = esn.run(data, return_states=["reservoir"])

#     assert out["reservoir"][0].shape == (10, 100)

#     s_reservoir = esn.state()
#     assert_equal(s_reservoir, res.state())

#     s_readout = esn.state(which="readout")
#     assert_equal(s_readout, read.state())

#     with pytest.raises(ValueError):
#         esn.state(which="foo")


def test_esn_feedback():
    esn = ESN(units=100, output_dim=5, lr=0.8, sr=0.4, ridge=1e-5, feedback=True)

    x = np.ones((13, 10))
    y = np.ones((13, 5))
    res = esn.fit(x, y).run(x)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 15)
    assert esn.readout.Wout.shape == (100, 5)
    assert res.shape == (13, 5)

    res = esn(x[0])


def test_esn_argument_collision():
    esn = ESN(
        units=100,
        input_dim=15,  # should be the input_dim of the reservoir, not the readout
        output_dim=5,
        lr=0.8,
        sr=0.4,
        bias=1.0,  # should be the bias of the reservoir, not the readout
        readout_bias=normal,  # should be the bias of the readout
        fit_bias=False,
        seed=1,
        name="MyNode",
        ridge=1e-5,
        feedback=True,
        input_to_readout=True,
    )

    x = np.ones((13, 10))
    y = np.ones((13, 5))
    res = esn.fit(x, y).run(x)

    assert esn.reservoir.output_dim == 100
    assert esn.reservoir.input_dim == 15
    assert esn.readout.input_dim == 110
    assert esn.readout.output_dim == 5
    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 15)
    assert esn.readout.Wout.shape == (110, 5)
    assert esn.readout.bias.shape == (5,)
    assert esn.reservoir.bias == 1.0
    assert res.shape == (13, 5)
    assert len(esn.nodes) == 3
    assert len(esn.edges) == 4

    res = esn(x[0])


def test_esn_freerunning():
    rng = np.random.default_rng(seed=1)
    model = ESN(units=100, sr=1.0, bias=normal)
    x, y = rng.normal(size=(100, 0)), rng.normal(size=(100, 3))
    model.fit(x, y)
    res = model.predict(x=None, iters=213)

    assert model.reservoir.input_dim == 0
    assert model.readout.input_dim == 100
    assert model.readout.output_dim == 3
    assert res.shape == (213, 3)
