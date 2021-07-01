# Author: Nathan Trouvain at 09/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np

from ..utils.validation import is_object_sequence
from ..model import Node


def forward(add, *data):
    if len(data) > 1:
        data = np.squeeze(data)
    else:  # nothing to add
        data = data[0]
    return np.sum(data, axis=0).reshape(1, -1)


def initialize(add, x=None):
    if x is not None:
        # x is an array, add over axis 0
        if isinstance(x, np.ndarray):
            add.set_input_dim(x.shape)
            add.set_output_dim((1, x.shape[1]))

        elif is_object_sequence(x):
            shapes = [array.shape for array in x]

            if not all([s[0] == 1 for s in shapes]):
                raise ValueError(f"Each timestep of data must be represented "
                                 f"by a vector of shape (1, dimension) when "
                                 f"entering node {add.name}. Received inputs "
                                 f"of shape {shapes}.")

            add.set_input_dim((len(x), x[0].shape[1]))

            if len(set([s[1] for s in shapes])) > 1:
                raise ValueError(f"Impossible to sum inputs: inputs have "
                                 f"different dimensions  entering node "
                                 f"{add.name}. Received inputs of shape "
                                 f"{shapes}.")
            else:
                add.set_output_dim((1, x[0].shape[1]))


class Add(Node):

    def __init__(self, name=None):
        super(Add, self).__init__(forward=forward,
                                  initializer=initialize,
                                  name=name)
