# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..node import Node, Model


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
