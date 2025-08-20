from typing import Optional

from reservoirpy.node import Node, TrainableNode
from reservoirpy.nodes.io import Input
from reservoirpy.nodes.reservoir import Reservoir
from reservoirpy.nodes.ridge import Ridge
from reservoirpy.type import Edge

from .model import Model
from .utils.model_utils import obj_from_kwargs


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
    >>> model = ESN(units=100, sr=0.9, ridge=1e-6)
    >>>
    """

    #: A :py:class:`~reservoirpy.nodes.Reservoir` or a :py:class:`~reservoirpy.nodes.NVAR` instance.
    reservoir: Node
    #: A :py:class:`~reservoirpy.nodes.Ridge` instance.
    readout: TrainableNode
    #: Is readout connected to reservoir through feedback (False by default).
    feedback: bool
    #: Does the readout directly receives the input (False by default).
    input_to_readout: bool
    #: Input node, if ``input_to_readout`` is set to True
    input_node: Optional[Input] = None

    def __init__(
        self,
        reservoir: Optional[Reservoir] = None,
        readout: Optional[Ridge] = None,
        feedback=False,
        input_to_readout=False,
        **kwargs,
    ):
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

        self.reservoir = reservoir
        self.readout = readout
        self.feedback = feedback
        self.input_to_readout = input_to_readout

        super(ESN, self).__init__(nodes=nodes, edges=edges)
