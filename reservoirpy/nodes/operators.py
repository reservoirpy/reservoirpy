# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Iterable

import numpy as np

from ..base import Node


def sum_forward(sum: Node, data):
    axis = sum.axis

    if len(data) > 1:
        x = np.concatenate(data, axis=0)
    else:
        x = data

    return np.sum(x, axis=axis)[np.newaxis, :]


def sum_initialize(sum: Node, x=None):
    if x is not None and len(x) > 1:
        if any([x[i].shape[1] != x[i+1].shape[1] for i in range(len(x)-1)]):
            raise ValueError(f"Dimension mismatch between input vectors"
                             f"of node {sum.name}: vectors have shapes "
                             f"{[v.shape for v in x]}.")

        sum.set_input_dim(x[0].shape[1])

        if sum.axis is None:
            output_dim = 1
        else:
            output_dim = len(x) if sum.axis == 1 else x.shape[1]

        sum.set_output_dim(output_dim)


def mul_forward(mul: Node, x):
    return x * mul.coef


def mul_initialize(mul: Node, x=None):
    if x is not None:
        mul.set_input_dim(x.shape[1])
        mul.set_output_dim(x.shape[1])


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
        elif isinstance(x, Iterable):
            result = concat_forward(concat, x)
            concat.set_input_dim(tuple([u.shape[1] for u in x]))
            if result.shape[0] > 1:
                concat.set_output_dim(result.shape)
            else:
                concat.set_output_dim(result.shape[1])


class Concat(Node):

    def __init__(self, axis=1, name=None):
        super(Concat, self).__init__(hypers={"axis": axis},
                                     forward=concat_forward,
                                     initializer=concat_initialize,
                                     name=name)


class Sum(Node):
    def __init__(self, axis=1, name=None):
        super(Sum, self).__init__(hypers={"axis": axis},
                                  forward=sum_forward,
                                  initializer=sum_initialize,
                                  name=name)


class Mul(Node):
    def __init__(self, coef, name=None):
        super(Mul, self).__init__(hypers={"coef": coef},
                                  forward=mul_forward,
                                  initializer=mul_initialize,
                                  name=name)
