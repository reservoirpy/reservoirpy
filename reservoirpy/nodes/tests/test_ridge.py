# Author: Nathan Trouvain at 24/09/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
# Author: Nathan Trouvain at 06/08/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from numpy.testing import assert_array_almost_equal

from reservoirpy.nodes import Reservoir, Ridge


def test_ridge_init():

    node = Ridge(10, ridge=1e-8)

    data = np.ones((1, 100))
    res = node(data)

    assert node.Wout.shape == (100, 10)
    assert node.bias.shape == (1, 10)
    assert node.ridge == 1e-8

    data = np.ones((10000, 100))
    res = node.run(data)

    assert res.shape == (10000, 10)


def test_ridge_partial_fit():

    node = Ridge(10, ridge=1e-8)

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
    res = node.fit(X, Y)

    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (1, 10)
    assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

    node = Ridge(10, ridge=1e-8)

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))

    for x, y in zip(X, Y):
        res = node.partial_fit(x, y)

    node.fit()

    assert node.Wout.shape == (100, 10)
    assert_array_almost_equal(node.Wout, np.ones((100, 10)) * 0.01, decimal=4)
    assert node.bias.shape == (1, 10)
    assert_array_almost_equal(node.bias, np.ones((1, 10)) * 0.01, decimal=4)

    data = np.ones((100, 100))
    res = node.run(data)

    assert res.shape == (100, 10)


def test_esn():

    readout = Ridge(10, ridge=1e-8)
    reservoir = Reservoir(100)

    esn = reservoir >> readout

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
    res = esn.fit(X, Y)

    assert readout.Wout.shape == (100, 10)
    assert readout.bias.shape == (1, 10)

    data = np.ones((100, 100))
    res = esn.run(data)

    assert res.shape == (100, 10)


def test_ridge_feedback():

    readout = Ridge(10, ridge=1e-8)
    reservoir = Reservoir(100)

    esn = reservoir >> readout

    reservoir <<= readout

    X, Y = np.ones((5, 200, 100)), np.ones((5, 200, 10))
    res = esn.fit(X, Y)

    assert readout.Wout.shape == (100, 10)
    assert readout.bias.shape == (1, 10)
    assert reservoir.Wfb.shape == (100, 10)

    data = np.ones((100, 100))
    res = esn.run(data)

    assert res.shape == (100, 10)


def test_hierarchical_esn():

    reservoir1 = Reservoir(100, input_dim=5, name="h1")
    readout1 = Ridge(ridge=1e-8, name="r1")

    reservoir2 = Reservoir(100, name="h2")
    readout2 = Ridge(ridge=1e-8, name="r2")

    esn = reservoir1 >> readout1 >> reservoir2 >> readout2

    X, Y = np.ones((1, 200, 5)), {
        "r1": np.ones((1, 200, 10)),
        "r2": np.ones((1, 200, 3)),
    }

    res = esn.fit(X, Y)

    assert readout1.Wout.shape == (100, 10)
    assert readout1.bias.shape == (1, 10)

    assert readout2.Wout.shape == (100, 3)
    assert readout2.bias.shape == (1, 3)

    assert reservoir1.Win.shape == (100, 5)
    assert reservoir2.Win.shape == (100, 10)

    data = np.ones((100, 5))
    res = esn.run(data)

    assert res.shape == (100, 3)


def test_parallel():
    if sys.platform in ["win32", "cygwin"] and sys.version_info < (3, 8):
        # joblib>=1.3.0 & Windows & Python<3.8 & loky combined are incompatible
        # see https://github.com/joblib/loky/issues/411
        return

    process_count = 4 * os.cpu_count()

    rng = np.random.default_rng(seed=42)
    x = rng.random((40000, 10))
    y = x[:, 2::-1] + rng.random((40000, 3)) / 10
    x_run = rng.random((20, 10))

    def run_ridge(i):
        readout = Ridge(ridge=1e-8)
        return readout.fit(x, y).run(x_run)

    parallel = Parallel(n_jobs=process_count, return_as="generator")
    results = list(parallel(delayed(run_ridge)(i) for i in range(process_count)))

    for result in results:
        assert np.all(result == results[0])
