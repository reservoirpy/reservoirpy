# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
from numpy.testing import assert_array_equal, assert_equal

from reservoirpy.jax.nodes import Input, Output


def test_input():
    inp = Input()
    x = jnp.ones((10,))
    out = inp(x)
    assert_array_equal(out, x)
    x = jnp.ones((10, 10))
    out = inp.run(x)
    assert_array_equal(out, x)


def test_output():
    output = Output()
    x = jnp.ones((10,))
    out = output(x)
    assert_array_equal(out, x)
    x = jnp.ones((100, 10))
    out = output.run(x)
    assert_array_equal(out, x)
