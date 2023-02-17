import numpy as np
from numpy.testing import assert_array_equal

from ..ops import Add
from ..reservoirs import Reservoir


def test_add():
    x = [np.ones((1, 5)) for _ in range(3)]

    a = Add()

    res = a(x)

    assert a.input_dim == 5
    assert_array_equal(res, 3 * np.ones((1, 5)))


def test_add_no_list():
    x = np.ones((1, 5))

    a = Add()

    res = a(x)

    assert_array_equal(res, np.ones((1, 5)))


def test_reservoir_union():

    reservoirs = [Reservoir(10, name=f"r{i}") for i in range(3)]

    model = reservoirs >> Add()

    x = {f"r{i}": np.ones((1, 5)) for i in range(3)}

    res = model(x)

    assert res.shape == (1, 10)