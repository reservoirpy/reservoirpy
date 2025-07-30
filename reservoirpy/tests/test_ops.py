# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from .dummy_nodes import MinusNode, Offline, PlusNode


def test_node_link():
    plus_node = PlusNode()
    minus_node = MinusNode()
    offline1 = Offline()
    offline2 = Offline()

    model1 = plus_node >> minus_node
    model2 = minus_node >> plus_node

    assert model1.edges == [(plus_node, 0, minus_node)]
    assert model2.edges == [(minus_node, 0, plus_node)]
    assert set(model1.nodes) == set(model2.nodes)

    model3 = plus_node >> offline1
    model4 = minus_node >> offline2

    model = model3 >> model4

    assert model.edges == [
        (plus_node, 0, offline1),
        (minus_node, 0, offline2),
        (offline1, 0, minus_node),
    ]
    assert set(model.nodes) == set(model3.nodes) | set(model4.nodes)

    # cycles in the model!
    with pytest.raises(RuntimeError):
        _ = model1 & model2

    with pytest.raises(RuntimeError):
        _ = plus_node >> minus_node >> plus_node

    with pytest.raises(RuntimeError):
        _ = plus_node >> plus_node

    x = np.ones((5,))
    x2 = np.ones((6,))
    plus_node(x)
    minus_node(x2)

    # bad dimensions
    with pytest.raises(ValueError):
        _ = plus_node >> minus_node

    with pytest.raises(ValueError):
        _ = model1(x)

    # merge inplace on a node
    model = plus_node
    model &= minus_node
    assert model.nodes == [plus_node, minus_node]
    assert model.edges == []


def test_node_link_several():
    plus_node = PlusNode(name="Plus")
    minus_node = MinusNode(name="Minus")
    offline_node = Offline(name="Offline")
    model = [plus_node, minus_node] >> offline_node

    assert len(model.nodes) == 3
    assert len(model.edges) == 2

    model = plus_node >> [offline_node, minus_node]

    assert model.nodes == [plus_node, offline_node, minus_node]
    assert model.edges == [(plus_node, 0, offline_node), (plus_node, 0, minus_node)]


def test_model_merge():
    plus_node = PlusNode(name="Plus")
    minus_node = MinusNode(name="Minus")
    offline_node = Offline(name="Offline")

    branch1 = plus_node >> minus_node
    branch2 = plus_node >> offline_node

    model = branch1 & branch2

    assert set(model.nodes) == {plus_node, minus_node, offline_node}
    assert set(model.edges) == {
        (plus_node, 0, minus_node),
        (plus_node, 0, offline_node),
    }

    branch1 &= branch2

    assert set(branch1.nodes) == {plus_node, minus_node, offline_node}
    assert set(branch1.edges) == {
        (plus_node, 0, minus_node),
        (plus_node, 0, offline_node),
    }
