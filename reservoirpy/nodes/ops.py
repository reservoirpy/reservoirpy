# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Sequence

import numpy as np

from ..node import Node


def concat_forward(concat: Node, data):
    axis = concat.axis

    if not isinstance(data, np.ndarray):
        if len(data) > 1:
            return np.concatenate(data, axis=axis)
        else:
            return np.asarray(data)
    else:
        return data


def concat_initialize(concat: Node, x=None, **kwargs):
    if x is not None:
        if isinstance(x, np.ndarray):
            concat.set_input_dim(x.shape[1])
            concat.set_output_dim(x.shape[1])
        elif isinstance(x, Sequence):
            result = concat_forward(concat, x)
            concat.set_input_dim(tuple([u.shape[1] for u in x]))
            if result.shape[0] > 1:
                concat.set_output_dim(result.shape)
            else:
                concat.set_output_dim(result.shape[1])


class Concat(Node):
    def __init__(self, axis=1, name=None):
        super(Concat, self).__init__(
            hypers={"axis": axis},
            forward=concat_forward,
            initializer=concat_initialize,
            name=name,
        )


def add_forward(add: Node, data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence):
        return sum(data)


def add_initialize(add: Node, x=None, **kwargs):
    if x is None:
        return
    if isinstance(x, np.ndarray):
        add.set_input_dim(x.shape[1])
        add.set_output_dim(x.shape[1])
    elif isinstance(x, Sequence):
        # Checking that all the shapes are equal or one (scalar inputs)
        dims = set(i.shape[1] for i in x)
        dims -= {1}
        if len(dims) > 1:
            raise AttributeError(
                "The inputs of the Add Node must all have the same dimension")
        dim = dims.pop()
        add.set_input_dim(dim)
        add.set_output_dim(dim)


class Add(Node):
    def __init__(self, name=None):
        super(Add, self).__init__(
            forward=add_forward,
            initializer=add_initialize,
            name=name)
