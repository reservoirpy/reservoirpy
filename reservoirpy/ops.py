"""
=========================================================
Node operations - link, merge (:py:mod:`reservoirpy.ops`)
=========================================================

Operations on :py:class:`~.Node` and :py:class:`~.Model`.

.. currentmodule:: reservoirpy.ops

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   link - Link Nodes or Model with a direct connection.
   merge - Merge Models.
   link_feedback - Link Nodes or Models with a feedback connection.
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from itertools import product
from typing import Sequence, Union

from .model import Model
from .node import Node
from .utils.graphflow import unique_ordered


class ModelBuilderUtil:
    """Utilitary model builder for the delay syntaxic sugar

    For example, when doing ``node1 >>3>> node2``, the first bit-shift operation considered is
    ``node1 >> 3`` which is evaluated as a :py:class:`ModelBuilderUtil` and then the second operation
    ``(node1>>3) >> node2`` which resolves as a Model, that links `node1` to `node2` with a delay of 3 timesteps.

    This should be invisible to the user, unless they mistakenly write things like ``node >> 1``.
    """

    def __init__(self, node: Union[Node, Model, Sequence[Union[Node, Model]]], delay: int, node_is_first: bool = True):
        self.node = node
        self.delay = delay
        self.node_is_first = node_is_first

    def __rshift__(self, other: Union[int, Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """self >> other"""
        if self.node_is_first:
            # (self.node >> self.delay) >> other
            return link(self.node, other, delay=self.delay)
        else:
            # (self.delay >> self.node) >> other
            return ModelBuilderUtil(node=link(self.node, other), delay=self.delay, node_is_first=False)

    def __rrshift__(self, other: Union[int, Node, "Model", Sequence[Union[Node, "Model"]]]) -> "Model":
        """other >> self"""
        if self.node_is_first:
            # other >> (self.node >> self.delay)
            return ModelBuilderUtil(node=link(other, self.node), delay=self.delay, node_is_first=True)
        else:
            # other >> (self.delay >> self.node)
            return link(other, self.node, delay=self.delay)


def _check_all_models(*operands):
    msg = "Impossible to link nodes: object {} is neither a Node nor a Model."
    for operand in operands:
        if isinstance(operand, Sequence):
            for model in operand:
                if not isinstance(model, (Node, Model)):
                    raise TypeError(msg.format(model))
        elif not isinstance(operand, (Node, Model)):
            raise TypeError(msg.format(operand))


def _link_1to1(
    left: Union[Node, Model], right: Union[Node, Model], delay=0
) -> tuple[list[Node], list[tuple[Node, int, Node]]]:
    """Connects two nodes or models. See `link` doc for more info."""
    # fetch all nodes in the two subgraphs, if they are models.

    nodes_left = [left] if isinstance(left, Node) else left.nodes
    nodes_right = [right] if isinstance(right, Node) else right.nodes
    all_nodes = unique_ordered(nodes_left + nodes_right)

    # fetch all edges in the two subgraphs, if they are models.
    edges_left = left.edges if isinstance(left, Model) else []
    edges_right = right.edges if isinstance(right, Model) else []
    all_edges = unique_ordered(edges_left + edges_right)

    # create edges between output nodes of the
    # subgraph 1 and input nodes of the subgraph 2.
    senders = [left] if isinstance(left, Node) else left.outputs
    receivers = [right] if isinstance(right, Node) else right.inputs

    new_edges = list(product(senders, [delay], receivers))

    # maybe nodes are already initialized?
    # check if connected dimensions are ok
    for sender, _, receiver in new_edges:
        if sender.initialized and receiver.initialized and sender.output_dim != receiver.input_dim:
            raise ValueError(
                f"Dimension mismatch between connected nodes: "
                f"sender node {sender} has output dimension "
                f"{sender.output_dim} but receiver node "
                f"{receiver} has input dimension "
                f"{receiver.input_dim}."
            )

    # all outputs from subgraph 1 are connected to
    # all inputs from subgraph 2.
    return all_nodes, all_edges + new_edges


def link(
    left: Union[Node, Model, Sequence[Union[Node, Model]]],
    right: Union[Node, Model, Sequence[Union[Node, Model]]],
    delay: int = 0,
) -> Model:
    """Link two :py:class:`~.Node` instances to form a :py:class:`~.Model`
    instance. `node1` output will be used as input for `node2` in the
    created model. This is similar to a function composition operation:

    .. math::

        model(x) = (node1 \\circ node2)(x) = node2(node1(x))

    You can also perform this operation using the ``>>`` operator::

        model = node1 >> node2

    Or using this function::

        model = link(node1, node2)

    -`node1` and `node2` can also be :py:class:`~.Model` instances. In this
    case, the new :py:class:`~.Model` created will contain all nodes previously
    contained in all the models, and link all `node1` outputs to all `node2`
    inputs. This allows to chain the  ``>>`` operator::

        step1 = node0 >> node1  # this is a model
        step2 = step1 >> node2  # this is another

    -`node1` and `node2` can finally be lists or tuples of nodes. In this
    case, all `node1` outputs will be linked to all `node2` inputs. You can
    still use the ``>>`` operator in this situation, except for many-to-many
    nodes connections::

        # many-to-one
        model = [node1, node2, ..., node] >> node_out
        # one-to-many
        model = node_in >> [node1, node2, ..., node]
        # ! many-to-many requires to use the `link` method explicitly!
        model = link([node1, node2, ..., node], [node1, node2, ..., node])

    Parameters
    ----------
        left : Node, Model or list of Node
            Nodes or lists of nodes to link.
        right : Node, Model or list of Node
            Nodes or lists of nodes to link.
        delay : int, defaults to 0
            Delay between the two parts

    Returns
    -------
        Model
            A :py:class:`~.Model` instance chaining the nodes.

    Raises
    ------
        TypeError
            Dimension mismatch between connected nodes: `left` output
            dimension if different from `right` input dimension.
            Reinitialize the nodes or create new ones.

    Notes
    -----

        Be careful to how you link the different nodes: `reservoirpy` does not
        allow to have circular dependencies between them::

            model = node1 >> node2  # fine
            model = node1 >> node2 >> node1  # raises! data would flow in
                                             # circles forever...
    """
    _check_all_models(left, right)

    left_seq: Sequence = left if isinstance(left, Sequence) else [left]
    right_seq: Sequence = right if isinstance(right, Sequence) else [right]

    nodes: list[Node] = []
    edges: list[tuple[Node, int, Node]] = []

    for left_node in left_seq:
        for right_node in right_seq:
            new_nodes, new_edges = _link_1to1(left_node, right_node, delay=delay)
            nodes += new_nodes
            edges += new_edges

    return Model(nodes=unique_ordered(nodes), edges=unique_ordered(edges))


def merge(*models: Union[Node, Model, Sequence[Union[Node, Model]]]) -> Model:
    """Merge different :py:class:`~.Model` or :py:class:`~.Node`
    instances into a single :py:class:`~.Model` instance.

    :py:class:`~.Node` instances contained in the models to merge will be
    gathered in a single model, along with all previously defined connections
    between them, if they exists.

    You can also perform this operation using the ``&`` operator::

        model = (node1 >> node2) & (node1 >> node3))

    This is equivalent to::

        model = merge((node1 >> node2), (node1 >> node3))

    The in-place operator can also be used::

        model &= other_model

    Which is equivalent to::

        model.update_graph(other_model.nodes, other_model.edges)

    Parameters
    ----------
    model: Model or Node
        First node or model to merge.
    *models : Model or Node
        All models to merge.

    Returns
    -------
    Model
        A new :py:class:`~.Model` instance.

    """
    _check_all_models(*models)

    nodes: list[Node] = []
    edges: list[tuple[Node, int, Node]] = []

    for model in models:
        if isinstance(model, Sequence):
            for element in model:
                if isinstance(element, Node):
                    nodes.append(element)
                elif isinstance(element, Model):
                    nodes += element.nodes
                    edges += element.edges
                else:
                    TypeError(f"Impossible to merge models: object {type(model)} is not a Node or a Model.")
        elif isinstance(model, Node):
            nodes.append(model)
        elif isinstance(model, Model):
            nodes += model.nodes
            edges += model.edges
        else:
            TypeError(f"Impossible to merge models: object {type(model)} is not a Node or a Model.")

    return Model(nodes=unique_ordered(nodes), edges=unique_ordered(edges))


def link_feedback(
    sender: Union[Node, Model, Sequence[Union[Node, Model]]],
    receiver: Union[Node, Model, Sequence[Union[Node, Model]]],
) -> Model:
    """Create a feedback connection from the ``sender`` to the ``receiver``.
    Feedback connections are regular node-to-node connections with a delay of
    one timestep.

    If ``sender`` or ``receiver`` is a Model or a list of Nodes or Models, all
    outputs of ``sender`` will be connected as feedback to ``receiver``.

     You can also perform this operation using the ``<<`` operator::

        node1 = node1 << node2
        # with feedback from a Model
        node1 = node1 << (fbnode1 >> fbnode2)
        # with feedback from a list of nodes or models
        node1 = node1 << [fbnode1, fbnode2, ...]

    You can also use this function to define feedback::

        node1 = link_feedback(node1, node2)

    Parameters
    ----------
    sender : Node, Model, or list of Nodes or Models
        Node(s) or Model(s) sending feedback.
    receiver : Node, Model, or list of Nodes or Models
        Node(s) or Model(s) receiving feedback.

    Returns
    -------
        Model
            A model instance with all node and connections from ``sender`` and
            ``receiver`` and with feedback connections from ``sender`` to
            ``receiver``.

    Raises
    ------
        TypeError
            If any of the senders or receivers are not Nodes or Models.
    """
    _check_all_models(sender, receiver)

    left_seq: Sequence = sender if isinstance(sender, Sequence) else [sender]
    right_seq: Sequence = receiver if isinstance(receiver, Sequence) else [receiver]

    nodes: list[Node] = []
    edges: list[tuple[Node, int, Node]] = []

    for sender_element in left_seq:
        for receiver_element in right_seq:
            new_nodes, new_edges = _link_1to1(sender_element, receiver_element, delay=1)
            nodes += new_nodes
            edges += new_edges

    return Model(nodes=unique_ordered(nodes), edges=unique_ordered(edges))
