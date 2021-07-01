# Author: Nathan Trouvain at 09/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..utils.validation import is_object_sequence
from ..model import Node


def forward(concat, *data):
    axis = concat.axis
    if len(data) <= 1:  # nothing to concat, data is a single array
        return data[0]
    return np.concatenate(data, axis=axis)


def initialize(concat, x=None):
    if x is not None:
        # x is an array, no need to concatenate
        if isinstance(x, np.ndarray):
            concat.set_input_dim(x.shape)
            concat.set_output_dim(x.shape)

        elif is_object_sequence(x):
            shapes = [array.shape for array in x]

            if not set([len(s) for s in shapes]) == 1:
                raise ValueError(f"Impossible to concatenate "
                                 f"inputs with different number of "
                                 f"dimensions entering {concat.name}. "
                                 f"Received inputs of shape {shapes}.")

            for s in shapes.copy():
                del s[concat.axis]

            if not set(shapes) == 1:
                raise ValueError(f"Impossible to concatenate inputs "
                                 f"over axis {concat.axis} with mismatched "
                                 f"dimensions over other axes entering "
                                 f"{concat.name}. Received inputs of dimensions "
                                 f"{shapes} over other axes.")
            else:
                ndim = x[0].ndim
                # output dim is the sum of arrays dims over the selected axis
                output_dim = (sum([s[concat.axis] for s in shapes]), )
                # the rest is left untouched
                output_dim += tuple([shapes[ax] for ax in range(ndim) if ax != concat.axis])
                concat.set_output_dim(output_dim)


class Concat(Node):

    def __init__(self, axis=0, name=None):
        super(Concat, self).__init__(hypers={"axis": axis},
                                     forward=forward,
                                     initializer=initialize,
                                     name=name)
