# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from reservoirpy.node import Node


def _io_initialize(io_node: "Node", x=None, **kwargs):
    if x is not None:
        if io_node.input_dim is None:
            io_node.set_input_dim(x.shape[1])
            io_node.set_output_dim(x.shape[1])
    else:
        if io_node.input_dim is None:
            raise RuntimeError(f"Impossible to infer shape of node {io_node}: "
                               f"no data was fed to the node. Try specify "
                               f"input dimension at node creation.")


def _input_forward(inp_node: "Input", x=None):
    if x is None and inp_node.has_feedback:
        x = inp_node.feedback()
    return x


class Input(Node):

    def __init__(self, input_dim=None, name=None):
        super(Input, self).__init__(forward=_input_forward,
                                    initializer=_io_initialize,
                                    input_dim=input_dim,
                                    output_dim=input_dim,
                                    name=name)


class Output(Node):

    def __init__(self, name=None):
        super(Output, self).__init__(forward=lambda inp, x: x,
                                     initializer=_io_initialize,
                                     name=name)
