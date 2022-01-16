"""
=========================================================
Node operations - link, merge (:py:mod:`reservoirpy.ops`)
=========================================================

Operations on :py:class:`~.Node` and :py:class:`~.Model`.

.. currentmodule:: reservoirpy.ops

.. autosummary::
   :toctree: generated/
   :template: autosummary/class.rst

   link - Link Nodes into a Model.
   link_feedback - Link Nodes through feedback connections.
   merge - Merge Models.
"""
# Author: Nathan Trouvain at 25/10/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from itertools import product
from typing import Union, Sequence
from uuid import uuid4

from .model import Model, FrozenModel
from .node import Node
from .types import GenericNode
from reservoirpy.utils.validation import check_all_nodes
from reservoirpy.nodes.concat import Concat


def _link_1to1(node1: GenericNode, node2: GenericNode, name=None) -> Model:
    """Connects two nodes or models. See `link` doc for more info."""
    # fetch all nodes in the two subgraphs, if they are models.
    all_nodes = []
    for node in (node1, node2):
        if isinstance(node, Model) and not isinstance(node, FrozenModel):
            all_nodes += node.nodes
        else:
            all_nodes += [node]

    # fetch all edges in the two subgraphs, if they are models.
    all_edges = []
    for node in (node1, node2):
        if isinstance(node, Model) and not isinstance(node, FrozenModel):
            all_edges += node.edges

    # create edges between output nodes of the
    # subgraph 1 and input nodes of the subgraph 2.
    senders = []
    if isinstance(node1, Model) and not isinstance(node, FrozenModel):
        senders += node1.output_nodes
    else:
        senders += [node1]

    receivers = []
    if isinstance(node2, Model) and not isinstance(node, FrozenModel):
        receivers += node2.input_nodes
    else:
        receivers += [node2]

    new_edges = list(product(senders, receivers))

    # maybe nodes are already initialized ?
    # check if connected dimensions are ok
    for sender, receiver in new_edges:
        if sender.is_initialized and \
                receiver.is_initialized and \
                sender.output_dim != receiver.input_dim:
            raise ValueError(f"Dimension mismatch between connected nodes: "
                             f"sender node {sender.name} has output dimension "
                             f"{sender.output_dim} but receiver node "
                             f"{receiver.name} has input dimension "
                             f"{receiver.input_dim}.")

    # all outputs from subgraph 1 are connected to
    # all inputs from subgraph 2.
    all_edges += new_edges

    # pack everything
    return Model(nodes=all_nodes, edges=all_edges, name=name)


def link(node1: Union[GenericNode, Sequence[GenericNode]],
         node2: Union[GenericNode, Sequence[GenericNode]],
         name: str = None) -> Model:
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
    case, all `node1` outputs will be linked to a :py:class:`~.Concat` node to
    concatenate them, and the :py:class:`~.Concat` node will be linked to all
    `node2` inputs. You can still use the ``>>`` operator in this situation,
    except for many-to-many nodes connections::

        # many-to-one
        model = [node1, node2, ..., node] >> node_out
        # one-to-many
        model = node_in >> [node1, node2, ..., node]
        # ! many-to-many requires to use the `link` method explicitely !
        model = link([node1, node2, ..., node], [node1, node2, ..., node])

    Parameters
    ----------
        node1, node2 : GenericNode or list of GenericNode
            Nodes or lists of nodes to link.
        name: str, optional
            Name for the chaining Model.

    Returns
    -------
        Model
            A :py:class:`~.Model` instance chaining the nodes.

    Raises
    ------
        TypeError
            Dimension mismatch between connected nodes: `node1` output
            dimension if different from `node2` input dimension.
            Reinitialize the nodes or create new ones.

    Notes
    -----

        Be carefull to how you link the different nodes: `reservoirpy` does not
        allow to have circular dependencies between them::

            model = node1 >> node2  # fine
            model = node1 >> node2 >> node1  # raises! data would flow in
                                             # circles forever...
    """

    check_all_nodes(node1, node2)

    frozens = []
    if isinstance(node1, Sequence):
        frozens += [n.name for n in node1 if isinstance(n, FrozenModel)]
    else:
        if isinstance(node1, FrozenModel):
            frozens.append(node1)
    if isinstance(node2, Sequence):
        frozens += [n.name for n in node2 if isinstance(n, FrozenModel)]
    else:
        if isinstance(node2, FrozenModel):
            frozens.append(2)

    if len(frozens) > 0:
        raise TypeError("Impossible to link FrozenModel to other Nodes or "
                        f"Model. FrozenModel found: {frozens}.")

    # get left side
    if isinstance(node1, Sequence):
        left_model = Model(name=str(uuid4()))

        if isinstance(node2, Concat):  # no need to add a Concat node then
            for n in node1:
                left_model &= _link_1to1(n, node2, name=str(uuid4()))
            return left_model
        else:  # concatenate everything on left side
            concat = Concat()
            for n in node1:
                left_model &= _link_1to1(n, concat, name=str(uuid4()))
    else:
        left_model = node1

    # connect left side with right side
    model = Model(name=name)
    if isinstance(node2, Sequence):
        for n in node2:
            model &= _link_1to1(left_model, n, name=str(uuid4()))
    else:
        model &= _link_1to1(left_model, node2, name=str(uuid4()))

    return model


def link_feedback(node: Node,
                  feedback: Union[GenericNode, Sequence[GenericNode]],
                  inplace: bool = False, name: str = None) -> GenericNode:
    """Create a feedback connection between the `feedback` node and `node`.
    Feedbacks nodes will be called at runtime using data from the previous
    call.

    This is not an inplace operation by default. This function will copy `node`
    and then sets the copy `_feedback` attribute as a reference to `feedback`
    node. If `inplace` is set to `True`, then `node` is not copied and the
    feedback is directly connected to `node`. If `feedback` is a list of nodes
    or models, then all nodes in the list are first connected to a
    :py:class:`~.Concat` node to create a model gathering all data from all nodes
    in a single feedback vector.

     You can also perform this operation using the ``<<`` operator::

        node1 = node1 << node2
        # with feedback from a Model
        node1 = node1 << (fbnode1 >> fbnode2)
        # with feedback from a list of nodes or models
        node1 = node1 << [fbnode1, fbnode2, ...]

    Which means that a feedback connection is now created between `node1` and
    `node2`. In other words, the forward function of `node1` depends on the
    previous output of `node2`:

    .. math::
        \\mathrm{node1}(x_t) = \\mathrm{node1}(x_t, \\mathrm{node2}(x_{t - 1}))

    You can also use this function to define feedback::

        node1 = link_feedback(node1, node2)
        # without copy (node1 is the same object throughout)
        node1 = link_feedback(node1, node2, inplace=True, name="n1_copy")

    Parameters
    ----------
    node : Node
        Node reciving feedback.
    feedback : GenericNode
        Node or Model sending feedback
    inplace : bool, defaults to False
        If `True`, then the function returns a copy of `node`.
    name : str, optional
        Name of the copy of `node` if `inplace` is `True`.

    Returns
    -------
        Node
            A node instance taking feedback from `feedback`.

    Raises
    ------
        TypeError
            - If `node` is a :py:class:`~.Model`.
            Models can not receive feedback.

            - If any of the feedback nodes are not :py:class:`~.GenericNode`
            instances.
    """

    if isinstance(node, Model):
        raise TypeError(f"{node} is not a Node. "
                        f"Models can't receive feedback.")

    msg = "Impossible to receive feedback from {}: " \
          "it is not a Node or a Model instance."

    if isinstance(feedback, Sequence):
        for fb in feedback:
            if not isinstance(fb, GenericNode):
                raise TypeError(msg.format(fb))

        all_fb = link(feedback, Concat())

    elif isinstance(feedback, GenericNode):
        all_fb = feedback

    else:
        raise TypeError(msg.format(feedback))

    if inplace:
        node._feedback = all_fb
        return node
    else:
        # first copy the node, then give it feedback
        # original node is not conencted to any feedback then
        new_node = node.copy(name=name)
        new_node._feedback = all_fb
        return new_node


def merge(model: GenericNode, *models: GenericNode, inplace: bool = False,
          name: str = None) -> GenericNode:
    """Merge different :py:class:`~.Model` or :py:class:`~.Node`
    instances into a single :py:class:`~.Model` instance.

    :py:class:`~.Node` instances contained in the models to merge will be
    gathered in a single model, along with all previously defined connections
    between them, if they exists.

    You can also perform this operation using the ``&`` operator::

        model = (node1 >> node2) & (node1 >> node3))

    This is equivalent to::

        model = merge((node1 >> node2), (node1 >> node3))

    The inplace operator can also be used::

        model &= other_model

    Which is equivalent to::

        model.update_graph(other_model.nodes, other_model.edges)

    Parameters
    ----------
    model: Model or Node
        First node or model to merge. The `inplace` parameter takes this
        instance as reference.
    *models : Model or Node
        All models to merge.
    inplace: bool, default to False
        If `True`, then will update Model `model` inplace. If `model` is not
        a Model instance, this parameter will causes the function to raise
        a `ValueError`.
    name: str, optional
        Name of the resulting Model.

    Returns
    -------
    Model
        A new :py:class:`~.Model` instance.

    Raises
    ------
    ValueError
        If `inplace` is `True` but `model` is not a Model instance, then the
        operation is impossible. Inplace merging can only take place on a
        Model instance.
    """
    msg = "Impossible to merge models: object {} is not a Model instance."

    if isinstance(model, GenericNode):
        all_nodes = set()
        all_edges = set()
        for m in models:
            # fuse models nodes and edges (right side argument)
            if isinstance(m, Model) and not isinstance(m, FrozenModel):
                all_nodes |= set(m.nodes)
                all_edges |= set(m.edges)
            elif isinstance(m, GenericNode):
                all_nodes |= {m}

        if inplace:
            if not isinstance(model, Model) or isinstance(model, FrozenModel):
                raise ValueError(f"Impossible to merge models inplace: "
                                 f"{model} is not a Model instance.")
            return model.update_graph(all_nodes, all_edges)

        else:
            # add left side model nodes
            if isinstance(model, Model) and not isinstance(model, FrozenModel):
                all_nodes |= set(model.nodes)
                all_edges |= set(model.edges)
            else:
                all_nodes |= {model}

            return Model(nodes=list(all_nodes), edges=list(all_edges),
                         name=name)

    else:
        raise TypeError(msg.format(type(model)))
