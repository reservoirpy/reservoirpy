# Author: Nathan Trouvain at 08/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Sequence

import numpy as np

from reservoirpy.node import Node
from ..utils.validation import check_node_io


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
        super(Concat, self).__init__(hypers={"axis": axis},
                                     forward=concat_forward,
                                     initializer=concat_initialize,
                                     name=name)

    def _check_io(self, X, *args, io_type="input", **kwargs):
        if io_type == "input":
            if isinstance(X, np.ndarray):
                return check_node_io(self, X, *args, io_type=io_type, **kwargs)
            elif isinstance(X, Sequence):
                checked_X = []
                for i in range(len(X)):
                    input_dim = None
                    if self.is_initialized:
                        input_dim = self.input_dim[i]
                    checked_X.append(check_node_io(self, X[i],
                                                   input_dim,
                                                   **kwargs))
                return checked_X
            else:
                return check_node_io(self, X, *args, io_type=io_type, **kwargs)
