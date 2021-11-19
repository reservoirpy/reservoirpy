# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pytest

from .dummy_nodes import *


def test_node_link(plus_node, minus_node, offline_node,
                   offline_node2, inverter_node):

    model1 = plus_node >> minus_node
    model2 = minus_node >> plus_node

    assert model1.edges == [(plus_node, minus_node)]
    assert model2.edges == [(minus_node, plus_node)]
    assert set(model1.nodes) == set(model2.nodes)

    model3 = plus_node >> offline_node
    model4 = minus_node >> offline_node2

    model = model3 >> model4

    assert set(model.edges) == {(plus_node, offline_node),
                                (offline_node, minus_node),
                                (minus_node, offline_node2)}
    assert set(model.nodes) == set(model3.nodes) | set(model4.nodes)

    # cycles in the model !
    with pytest.raises(RuntimeError):
        model1 & model2

    with pytest.raises(RuntimeError):
        plus_node >> minus_node >> plus_node

    with pytest.raises(RuntimeError):
        plus_node >> plus_node

    x = np.ones((1, 5))
    x2 = np.ones((1, 6))
    plus_node(x)
    minus_node(x2)

    # bad dimensions
    with pytest.raises(ValueError):
        plus_node >> minus_node

    with pytest.raises(ValueError):
        model1(x)

    # merge inplace on a node
    with pytest.raises(TypeError):
        plus_node &= minus_node


def test_node_link_several(plus_node, minus_node, offline_node):

    model = [plus_node, minus_node] >> offline_node

    assert len(model.nodes) == 4
    assert len(model.edges) == 3

    model = plus_node >> [offline_node, minus_node]

    assert set(model.nodes) == {plus_node, minus_node, offline_node}
    assert set(model.edges) == {(plus_node, offline_node),
                                (plus_node, minus_node)}


def test_node_link_feedback(plus_node, minus_node):

    fb_plus_node = plus_node << minus_node

    assert id(fb_plus_node._feedback) == id(minus_node)
    assert plus_node._feedback is None

    plus_node <<= minus_node
    assert id(plus_node._feedback) == id(minus_node)
