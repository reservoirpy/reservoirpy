# Author: Nathan Trouvain at 18/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from numpy.testing import assert_array_equal

from ..concat import Concat
from ..reservoir import Reservoir


def test_concat():
    x = [np.ones((1, 5)) for _ in range(3)]

    c = Concat()

    res = c(x)

    assert c.input_dim == (5, 5, 5)
    assert_array_equal(res, np.ones((1, 15)))

    res = c(x)


def test_concat_no_list():
    x = np.ones((1, 5))

    c = Concat()

    res = c(x)

    assert_array_equal(res, np.ones((1, 5)))


def test_reservoir_union():

    reservoirs = [Reservoir(10, name=f"r{i}") for i in range(3)]

    model = reservoirs >> Concat()

    x = {f"r{i}": np.ones((1, 5)) for i in range(3)}

    res = model(x)

    assert res.shape == (1, 30)

    res_final = Reservoir(20)

    model = reservoirs >> res_final

    res = model(x)

    assert res.shape == (1, 20)
    assert any([isinstance(n, Concat) for n in model.nodes])
