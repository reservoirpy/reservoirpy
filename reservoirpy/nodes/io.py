# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from ..node import Node


def _io_initialize(io_node: "Node", x=None, **kwargs):
    if x is not None:
        if io_node.input_dim is None:
            io_node.set_input_dim(x.shape[1])
            io_node.set_output_dim(x.shape[1])


def _input_forward(inp_node: "Input", x):
    return x


class Input(Node):
    """Node feeding input data to other nodes in the models.

    Allow creating an input source and connecting it to several nodes at once.

    This node has no parameters and no hyperparameters.

    Parameters
    ----------
    input_dim : int
        Input dimension. Can be inferred at first call.
    name : str
        Node name.

    Example
    -------

    An input source feeding three different nodes in parallel.

    >>> from reservoirpy.nodes import Reservoir, Input
    >>> source = Input()
    >>> res1, res2, res3 = Reservoir(100), Reservoir(100), Reservoir(100)
    >>> model = source >> [res1, res2, res3]

    A model with different input sources. Use names to identify each source at runtime.

    >>> from reservoirpy.nodes import Reservoir, Input
    >>> source1, source2 = Input(name="s1"), Input(name="s2")
    >>> res1, res2 = Reservoir(100), Reservoir(100)
    >>> model = source1 >> [res1, res2] & source2 >> [res1, res2]
    >>> outputs = model.run({"s1": np.ones((10, 5)), "s2": np.ones((10, 3))})
    """

    def __init__(self, input_dim=None, name=None, **kwargs):
        super(Input, self).__init__(
            forward=_input_forward,
            initializer=_io_initialize,
            input_dim=input_dim,
            output_dim=input_dim,
            name=name,
            **kwargs,
        )


class Output(Node):
    """Convenience node which can be used to add an output to a model.

    For instance, this node can be connected to a reservoir within a model to inspect
    its states.

    Parameters
    ----------
    name : str
        Node name.

    Example
    -------

    We can use the :py:class:`Output` node to probe the hidden states of Reservoir
    in an Echo State Network:

    >>> from reservoirpy.nodes import Reservoir, Ridge, Output
    >>> reservoir = Reservoir(100)
    >>> readout = Ridge()
    >>> probe = Output(name="reservoir-states")
    >>> esn = reservoir >> readout & reservoir >> probe

    When running the model, states can then be retrieved as an output:

    >>> data = np.ones((10, 5))
    >>> outputs = esn.run(data)
    >>> states = outputs["reservoir-states"]
    """

    def __init__(self, name=None, **kwargs):
        super(Output, self).__init__(
            forward=_input_forward, initializer=_io_initialize, name=name, **kwargs
        )
