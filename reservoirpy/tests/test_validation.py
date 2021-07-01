# Author: Nathan Trouvain at 23/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from ..utils.validation import (add_bias,
                                check_datatype,
                                check_input_lists,
                                check_reservoir_matrices)


@pytest.mark.parametrize("X,Y", (
        ([np.ones((10, 5)) for _ in range(10)], [np.ones((10, 2)) for _ in range(9)]),
        ([np.ones((10, 5)) for _ in range(9)] + [np.ones((9, 5))], [np.ones((10, 2)) for _ in range(10)]),
        ([np.ones((10, 5)) for _ in range(10)],
         [np.ones((10, 3))] + [np.ones((10, 2)) for _ in range(9)])
))
def test_bad_input_list(X, Y):
    with pytest.raises(ValueError):
        check_input_lists(X, dim_in=5, Y=Y, dim_out=2)


def test_good_input_list():
    X = [np.ones((10, 5)) for _ in range(10)]
    Y = [np.ones((10, 2)) for _ in range(10)]

    X1, Y1 = check_input_lists(X, dim_in=5, Y=Y, dim_out=2)

    assert all([np.all(x0 == x1) for x0, x1 in zip(X, X1)])
    assert all([np.all(x0 == x1) for x0, x1 in zip(Y, Y1)])


def test_bad_matrices_data():
    W = np.ones((10, 10))
    W[1, 1] = np.nan
    Win = np.ones((2, 10))

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win)

    W = np.ones((10, 10))
    Win = np.ones((2, 10))
    Win[0, 3] = np.inf

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win)

    W = np.ones((10, 10))
    Win = np.ones((2, 10))
    Wout = np.ones((2, 10))
    Wout[0, 0] = None

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win, Wout=Wout)


def test_bad_matrices_type():
    W = np.ones((10, 10))
    Win = np.ones((2, 10)).astype(str)

    with pytest.raises(TypeError):
        check_reservoir_matrices(W, Win)

    W = np.ones((10, 10)).tolist()
    W[0][0] = "a"
    Win = np.ones((2, 10))

    with pytest.raises(TypeError):
        check_reservoir_matrices(W, Win)


def test_bad_matrices_shapes():
    W = np.ones((5, 10))
    Win = np.ones((2, 10))

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win)

    W = np.ones((10, 10))
    Win = np.ones((9, 2))

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win)

    W = np.ones((10, 10))
    Win = np.ones((10, 2))
    Wout = np.zeros((10, 1))

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win, Wout=Wout)

    W = np.ones((10, 10))
    Win = np.ones((10, 2))
    Wout = np.zeros((11, 1))
    Wfb = np.ones((10, 2))

    with pytest.raises(ValueError):
        check_reservoir_matrices(W, Win, Wout=Wout, Wfb=Wfb)


def test_good_matrices():
    W = np.ones((10, 10))
    Win = np.ones((10, 2))
    Wout = np.zeros((1, 11))
    Wfb = np.ones((10, 1))

    W1, Win1, Wout1, Wfb1 = check_reservoir_matrices(W, Win, Wout=Wout, Wfb=Wfb)

    assert_array_equal(W1, W)
    assert_array_equal(Win1, Win)
    assert_array_equal(Wout1, Wout)
    assert_array_equal(Wfb1, Wfb)


def test_good_sparse_matrices():
    W = csr_matrix(np.ones((10, 10)))
    Win = csr_matrix(np.ones((10, 2)))
    Wout = csr_matrix(np.zeros((1, 11)))
    Wfb = csr_matrix(np.ones((10, 1)))

    W1, Win1, Wout1, Wfb1 = check_reservoir_matrices(W, Win, Wout=Wout, Wfb=Wfb)

    assert_array_equal(W1.toarray(), W.toarray())
    assert_array_equal(Win1.toarray(), Win.toarray())
    assert_array_equal(Wout1.toarray(), Wout.toarray())
    assert_array_equal(Wfb1.toarray(), Wfb.toarray())
