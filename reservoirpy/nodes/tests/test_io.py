# Author: Nathan Trouvain at 14/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_equal

from reservoirpy.nodes import Input, Output


def test_input():
    inp = Input()
    x = np.ones((10,))
    out = inp(x)
    assert_equal(out, x)
    x = np.ones((10, 10))
    out = inp.run(x)
    assert_equal(out, x)


def test_output():
    output = Output()
    x = np.ones((10,))
    out = output(x)
    assert_equal(out, x)
    x = np.ones((100, 10))
    out = output.run(x)
    assert_equal(out, x)
