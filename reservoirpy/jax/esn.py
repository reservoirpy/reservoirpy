# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Optional

from ..type import Edge
from ..utils.model_utils import obj_from_kwargs
from .model import Model
from .node import Node, TrainableNode
from .nodes import Input, Output, Reservoir, Ridge


class ESN(Model):
    """Simple Echo State Network.

    This class is provided as a wrapper for a simple reservoir connected to a
    readout.

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    reservoir : Node, optional
        A Node instance to use as a reservoir,
        such as a :py:class:`~reservoirpy.nodes.Reservoir` node.
    readout : Node, optional
        A Node instance to use as a readout,
        such as a :py:class:`~reservoirpy.nodes.Ridge` node
        (only this one is supported).
    feedback : bool, defaults to False
        If True, the readout is connected to the reservoir through
        a feedback connection.
    input_to_readout : bool, defaults to False
        If True, the input is directly fed to the readout. See
        :ref:`/user_guide/advanced_demo.ipynb#Input-to-readout-connections`.
    return_reservoir_activity : bool, defaults to False
        If True, the model outputs a dict with the reservoir activity in addition to the readout output.
    **kwargs
        Arguments passed to the reservoir and readout.

    See Also
    --------
    Reservoir
    Ridge

    Example
    -------
    >>> from reservoirpy import ESN
    >>>
    >>> model = ESN(units=100, sr=0.9, ridge=1e-6)  # reservoir and readout parameters at once
    >>> model.fit(x_train, y_train)
    >>>
    >>> model = ESN(reservoir=Reservoir(100, sr=0.9), readout=Ridge(1e-5))  # passing nodes as parameters
    >>>
    >>> model = ESN(units=100, input_to_readout=True, feedback=True)  # more complex model
    """

    #: A :py:class:`~reservoirpy.nodes.Reservoir` or a :py:class:`~reservoirpy.nodes.NVAR` instance.
    reservoir: Node
    #: A :py:class:`~reservoirpy.nodes.Ridge` instance.
    readout: TrainableNode
    #: Is readout connected to reservoir through feedback (False by default).
    feedback: bool
    #: Is the readout directly receiving the input (False by default).
    input_to_readout: bool
    #: Are the reservoir states returned by the model along the readout output (False by default).
    return_reservoir_activity: bool
    #: Output node for the reservoir, if ``return_reservoir_activity`` is set to True
    output_node: Optional[Output] = None
    #: Input node, if ``input_to_readout`` is set to True
    input_node: Optional[Input] = None

    def __init__(
        self,
        reservoir: Optional[Reservoir] = None,
        readout: Optional[Ridge] = None,
        feedback: bool = False,
        input_to_readout: bool = False,
        return_reservoir_activity: bool = False,
        **kwargs,
    ):
        if "name" in kwargs:
            kwargs.pop("name")

        if reservoir is None:
            reservoir = obj_from_kwargs(Reservoir, kwargs)

        if readout is None:
            # avoid argument name collision
            if "bias" in kwargs:
                kwargs.pop("bias")
            if "readout_bias" in kwargs:
                kwargs["bias"] = kwargs["readout_bias"]
                kwargs.pop("readout_bias")
            if "input_dim" in kwargs:
                kwargs.pop("input_dim")
            kwargs["name"] = "readout"
            readout = obj_from_kwargs(Ridge, kwargs)

        nodes: list[Node] = [reservoir, readout]
        edges: list[Edge] = [(reservoir, 0, readout)]

        if feedback:
            edges.append((readout, 1, reservoir))

        if input_to_readout:
            input_node = Input()
            nodes.append(input_node)
            edges.append((input_node, 0, reservoir))
            edges.append((input_node, 0, readout))
            self.input_node = input_node

        if return_reservoir_activity:
            output_node = Output(name="reservoir")
            nodes.append(output_node)
            edges.append((reservoir, 0, output_node))
            self.output_node = output_node

        self.reservoir = reservoir
        self.readout = readout
        self.feedback = feedback
        self.input_to_readout = input_to_readout
        self.return_reservoir_activity = return_reservoir_activity

        super(ESN, self).__init__(nodes=nodes, edges=edges)
