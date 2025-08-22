# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Optional, Sequence, Union

from ..node import Node
from ..type import NodeInput, State, Timeseries, Timestep


class Input(Node):
    """Node feeding input data to other nodes in the models.

    Allow creating an input source and connecting it to several nodes at once.

    This node has no parameters and no hyperparameters.

    Parameters
    ----------
    name : str, optional
        Node name.

    Example
    -------

    An input source feeding three different nodes in parallel.

    >>> from reservoirpy.nodes import Reservoir, Input
    >>> source = Input()
    >>> res1, res2, res3 = Reservoir(100), Reservoir(100), Reservoir(100)
    >>> model = source >> [res1, res2, res3]

    A model with different input sources. Use names to identify each source at runtime.

    >>> import numpy as np
    >>> from reservoirpy.nodes import Reservoir, Input
    >>> source1, source2 = Input(name="s1"), Input(name="s2")
    >>> res1, res2 = Reservoir(100, name="res1"), Reservoir(100, name="res2)
    >>> model = source1 >> [res1, res2] & source2 >> [res1, res2]
    >>> outputs = model.run({"s1": np.ones((10, 5)), "s2": np.ones((10, 3))})
    """

    initialized: bool
    state: State

    def __init__(self, name: Optional[str] = None):
        self.initialized = False
        self.name = name
        self.state = {}

    def initialize(self, x: Union[NodeInput, Timestep]):
        self._set_input_dim(x)
        self.output_dim = self.input_dim
        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        return {"out": x[-1]}, x


class Output(Node):
    """Convenience node which can be used to add an output to a model.

    For instance, this node can be connected to a reservoir within a model to inspect
    its states.

    Parameters
    ----------
    name : str, optional
        Node name.

    Example
    -------

    We can use the :py:class:`Output` node to probe the hidden states of Reservoir
    in an Echo State Network:

    >>> import numpy as np
    >>> from reservoirpy.nodes import Reservoir, Ridge, Output
    >>> reservoir = Reservoir(100)
    >>> readout = Ridge(name="readout")
    >>> probe = Output(name="reservoir-states")
    >>> esn = reservoir >> readout & reservoir >> probe
    >>> _ = esn.initialize(np.ones((1,1)), np.ones((1,1)))

    When running the model, states can then be retrieved as an output:

    >>> data = np.ones((10, 1))
    >>> outputs = esn.run(data)
    >>> states = outputs["reservoir-states"]
    """

    initialized: bool
    state: State

    def __init__(self, name: Optional[str] = None):
        self.initialized = False
        self.name = name
        self.state = {}

    def initialize(self, x: Union[NodeInput, Timestep]):
        dim = x.shape[-1] if not isinstance(x, Sequence) else x[0].shape[-1]
        self.input_dim = dim
        self.output_dim = dim
        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": x}

    def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        return {"out": x[-1]}, x
