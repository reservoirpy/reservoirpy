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

    x = np.ones((1, 10))
    y = np.ones((1, 5))
    res = esn(x)

    assert esn.reservoir.W.shape == (100, 100)
    assert esn.reservoir.Win.shape == (100, 10)
    assert esn.readout.Wout.shape == (100, 5)
    assert res.shape == (1, 5)
    assert esn.reservoir.Wfb is not None
    assert esn.reservoir.Wfb.shape == (100, 5)

    esn.fit(x, y).run(x, forced_feedbacks=y)


def test_esn_parallel_fit_reproducibility():
    """Reproducibility of the ESN node across backends.
    Results may vary between OSes and NumPy versions.
    """
    seed = 1234
    rng = np.random.default_rng(seed=seed)
    X = list(rng.normal(0, 1, (10, 100, 10)))
    Y = [x @ rng.normal(0, 1, size=(10, 5)) for x in X]

    set_seed(seed)
    base_Wout = (
        ESN(
            units=100,
            ridge=1e-5,
            feedback=True,
            workers=-1,
            backend="sequential",
        )
        .fit(X, Y)
        .readout.Wout
    )

    for backend in ("loky", "multiprocessing", "threading", "sequential"):
        set_seed(seed)
        esn = ESN(
            units=100,
            ridge=1e-5,
            feedback=True,
            workers=-1,
            backend=backend,
        ).fit(X, Y)

        assert esn.reservoir.W.shape == (100, 100)
        assert esn.reservoir.Win.shape == (100, 10)
        assert esn.readout.Wout.shape == (100, 5)

        assert esn.reservoir.Wfb is not None
        assert esn.reservoir.Wfb.shape == (100, 5)
        assert np.abs(np.mean(esn.readout.Wout - base_Wout)) < 1e-14


def test_esn_parallel_run_reproducibility():
    """Reproducibility of the ESN node across backends. Results may
    vary between OSes and NumPy versions.
    """
    seed = 1000
    rng = np.random.default_rng(seed=seed)
    X = list(rng.normal(0, 1, (10, 100, 10)))
    Y = [x @ rng.normal(0, 1, size=(10, 5)) for x in X]

    set_seed(seed)
    # no feedback here. XXT and YXT sum orders are not deterministic
    # which results in small (float precision) differences across fits
    # and leads to error accumulation on run with feedback.
    set_seed(seed)
    esn = ESN(
        units=100,
        ridge=1e-5,
        workers=1,
        backend="sequential",
    ).fit(X, Y)

    base_y_out = esn.run(X[0])

    for backend in ("loky", "multiprocessing", "threading", "sequential"):
        set_seed(seed)
        esn = ESN(
            units=100,
            ridge=1e-5,
            workers=-1,
            backend=backend,
        ).fit(X, Y)

        y_out = esn.run(X[0])
        assert np.abs(np.mean(y_out - base_y_out)) < 1e-14


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
