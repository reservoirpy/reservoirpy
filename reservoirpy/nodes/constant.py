from ..node import Node

import numpy as np


def constant_forward_fn(constant: Node, x=None):
    return constant.value


def constant_initialize(constant: Node, x=None, **kwargs):
    print(x)
    constant.set_input_dim(None)
    if isinstance(constant.value, np.ndarray):
        constant.set_output_dim(constant.value.size)
    else:
        constant.set_output_dim(1)


class Constant(Node):
    def __init__(self, value=0):
        super(Constant, self).__init__(
            forward=constant_forward_fn,
            initializer=constant_initialize,
            params={'value': np.array(value)})
        self._state = np.atleast_2d(np.array(value))
