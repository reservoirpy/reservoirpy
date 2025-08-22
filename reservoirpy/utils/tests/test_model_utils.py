import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.utils.model_utils import data_from_buffer, join_data


def test_join_data():
    x1 = np.zeros((10, 2))
    out = join_data(x1)
    assert_array_equal(x1, out)

    x2 = np.ones((10, 12))
    out = join_data(x1, x2)
    assert out.shape == (10, 2 + 12)

    list1 = [np.ones((2 * i + 5, 7)) for i in range(3)]
    list2 = [np.ones((2 * i + 5, 3)) for i in range(3)]
    out = join_data(list1, list2, list1)
    assert isinstance(out, list)
    for el in out:
        assert el.shape[-1] == 7 + 3 + 7


def test_data_from_buffer():
    # simple timeseries
    x = np.arange(10).reshape(-1, 1)
    buffer = np.arange(-5, 0).reshape(-1, 1)
    new_buffer, new_x = data_from_buffer(buffer, x)
    assert new_buffer.shape == buffer.shape
    assert new_x.shape == x.shape
    np.testing.assert_array_equal(new_buffer, x[-5:])

    # multiseries 3D array
    x = np.arange(20).reshape(2, -1, 1)
    buffer = np.arange(-5, 0).reshape(-1, 1)
    new_buffer, new_x = data_from_buffer(buffer, x)
    assert new_buffer.shape == buffer.shape
    assert new_x.shape == x.shape
    np.testing.assert_array_equal(new_buffer, x[-1, -5:])

    # list of timeseries
    x = [np.arange(100 * i, 100 * i + 10).reshape(-1, 1) for i in range(3)]
    buffer = np.arange(-5, 0).reshape(-1, 1)
    new_buffer, new_x = data_from_buffer(buffer, x)
    assert len(new_x) == len(x)
    np.testing.assert_array_equal(new_buffer, x[-1][-5:])
