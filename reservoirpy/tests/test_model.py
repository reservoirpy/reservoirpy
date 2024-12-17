# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.nodes.io import Input, Output

from ..model import Model
from ..node import Node
from ..ops import merge
from .dummy_nodes import Inverter, MinusNode, Offline, PlusNode


def test_node_link(plus_node, minus_node):
    clean_registry(Model)

    model1 = plus_node >> minus_node
    model2 = minus_node >> plus_node

    model1._hypers["hyper1"] = "hyper1"
    model1._params["param1"] = "param1"
    assert model1.name == "Model-0"
    model1.name = "Model-1000"
    assert model1.name == "Model-1000"
    assert model1.params["PlusNode-0"]["c"] is None
    assert model1.hypers["PlusNode-0"]["h"] == 1
    assert model1.hyper1 == "hyper1"
    assert model1.param1 == "param1"
    assert model1["PlusNode-0"].input_dim is None

    assert model2.name == "Model-1"
    with pytest.raises(NameError):  # already taken
        model2.name = "Model-1000"
    assert model2.params["PlusNode-0"]["c"] is None
    assert model2.hypers["PlusNode-0"]["h"] == 1
    assert model2["PlusNode-0"].input_dim is None

    assert model1.edges == [(plus_node, minus_node)]
    assert model2.edges == [(minus_node, plus_node)]
    assert set(model1.nodes) == set(model2.nodes)

    with pytest.raises(RuntimeError):
        model1 & model2

    with pytest.raises(NameError):
        _ = model1.get_param("fake_parameter")

    with pytest.raises(RuntimeError):
        plus_node >> minus_node >> plus_node

    with pytest.raises(RuntimeError):
        plus_node >> plus_node


def test_complex_node_link():
    clean_registry(Model)

    A = Node(name="A")
    B = Node(name="B")
    C = Node(name="C")
    D = Node(name="D")
    E = Node(name="E")
    F = Node(name="F")
    In = Input(name="In")
    Out = Output(name="Out")

    path1, path2 = A >> F, B >> E
    path3 = In >> [A, B, C]
    path4 = A >> B >> C >> D >> E >> F >> Out
    model = path1 & path2 & path3 & path4

    assert len(model.nodes) == 12  # 8 user-defined + 4 concat nodes
    assert len(model.edges) == 15  # 11 user-defined + 4 created connections


def test_empty_model_init():
    model = Model()
    assert model.is_empty


def test_model_call(plus_node, minus_node):
    model = plus_node >> minus_node

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    res = model(data)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 2)
        else:
            assert_array_equal(arr, data - 2)

    res = model(data)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 4)
        else:
            assert_array_equal(arr, data)

    res = model(data, reset=True)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 2)
        else:
            assert_array_equal(arr, data - 2)

    res = model(data, stateful=False)

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, data + 4)
        else:
            assert_array_equal(arr, data)

    for node in model.output_nodes:
        if node.name == "PlusNode-0":
            assert_array_equal(node.state(), data + 2)
        else:
            assert_array_equal(node.state(), data - 2)


def test_model_with_state(plus_node, minus_node):
    model = plus_node >> minus_node

    data = np.zeros((1, 5))
    res = model(data)

    assert_array_equal(res, data)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    res = model(data)

    with model.with_state(state={plus_node.name: np.zeros_like(plus_node.state())}):
        assert_array_equal(plus_node.state(), np.zeros_like(plus_node.state()))

    with pytest.raises(TypeError):
        with model.with_state(state=np.zeros_like(plus_node.state())):
            pass


def test_model_run(plus_node, minus_node):
    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = merge(branch1, branch2)

    data = np.zeros((3, 5))
    res = model.run(data)

    expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
    expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus)
            assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
        else:
            assert_array_equal(arr, expected_minus)
            assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

    res = model.run(data, reset=True)

    expected_plus = np.array([[2] * 5, [4] * 5, [6] * 5])
    expected_minus = np.array([[-2] * 5, [0] * 5, [-2] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus)
            assert_array_equal(arr[-1][np.newaxis, :], plus_node.state())
        else:
            assert_array_equal(arr, expected_minus)
            assert_array_equal(arr[-1][np.newaxis, :], minus_node.state())

    res = model.run(data, stateful=False)

    expected_plus2 = np.array([[8] * 5, [10] * 5, [12] * 5])
    expected_minus2 = np.array([[0] * 5, [-2] * 5, [0] * 5])

    for name, arr in res.items():
        assert name in [out.name for out in model.output_nodes]
        if name == "PlusNode-0":
            assert_array_equal(arr, expected_plus2)
            assert_array_equal(expected_plus[-1][np.newaxis, :], plus_node.state())
        else:
            assert_array_equal(arr, expected_minus2)
            assert_array_equal(expected_minus[-1][np.newaxis, :], minus_node.state())


def test_model_run_on_sequences(plus_node, minus_node):
    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    data = np.zeros((5, 3, 5))
    res = model.run(data)

    assert set(res.keys()) == {plus_node.name, minus_node.name}
    assert len(res[plus_node.name]) == 5
    assert len(res[minus_node.name]) == 5
    assert res[plus_node.name][0].shape == (3, 5)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = branch1 & branch2

    data = [np.zeros((3, 5)), np.zeros((8, 5))]
    res = model.run(data)

    assert set(res.keys()) == {plus_node.name, minus_node.name}
    assert len(res[plus_node.name]) == 2
    assert len(res[minus_node.name]) == 2
    assert res[plus_node.name][0].shape == (3, 5)
    assert res[plus_node.name][1].shape == (8, 5)


def test_offline_fit_simple_model(offline_node, offline_node2, plus_node, minus_node):
    model = plus_node >> offline_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([6.5]))

    X = np.ones((3, 5, 5)) * 0.5
    Y = np.ones((3, 5, 5))

    model.fit(X, Y)

    assert_array_equal(offline_node.b, np.array([94.5]))

    model.fit(X, Y, reset=True)

    assert_array_equal(offline_node.b, np.array([19.5]))

    res = model.run(X[0], reset=True)

    exp = np.tile(np.array([22.0, 24.5, 27.0, 29.5, 32.0]), 5).reshape(5, 5).T

    assert_array_equal(exp, res)


def test_offline_fit_complex(
    basic_offline_node, offline_node2, plus_node, minus_node, feedback_node
):
    model = [plus_node >> basic_offline_node, plus_node] >> minus_node >> offline_node2

    X = np.ones((5, 5, 5)) * 0.5
    Y_1 = np.ones((5, 5, 5))
    Y_2 = np.ones((5, 5, 10))  # after concat

    model.fit(X, Y={"BasicOffline-0": Y_1, "Offline2-0": Y_2})

    res = model.run(X[0])

    assert res.shape == (5, 10)


def test_online_train_simple(online_node, plus_node):
    model = plus_node >> online_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.train(X, Y)

    assert_array_equal(online_node.b, np.array([42.5]))

    model.train(X, Y, reset=True)

    assert_array_equal(online_node.b, np.array([85]))


def test_online_train_teacher_nodes(online_node, plus_node, minus_node):
    X = np.ones((5, 5)) * 0.5
    model = plus_node >> online_node

    with pytest.raises(RuntimeError):
        model.train(X, minus_node)  # Impossible to init node nor infer shape

    model = plus_node >> [minus_node, online_node]

    minus_node.output_dim = 5

    model.train(X, minus_node)

    assert_array_equal(online_node.b, np.array([54.0]))

    model.train(X, minus_node, reset=True)

    assert_array_equal(online_node.b, np.array([108.0]))


def test_model_return_states():
    off = Offline(name="offline")
    plus = PlusNode(name="plus")
    minus = MinusNode(name="minus")
    inverter = Inverter(name="inv")

    model = plus >> [minus, off >> inverter]

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    res = model.run(X)

    assert set(res.keys()) == {"minus", "inv"}

    res = model.run(X, return_states="all")

    assert set(res.keys()) == {"minus", "inv", "offline", "plus"}

    res = model.run(X, return_states=["offline"])

    assert set(res.keys()) == {"offline"}


def test_multiinputs():
    import numpy as np

    from reservoirpy.nodes import Input, Reservoir

    source1, source2 = (
        Input(
            name="s1",
            input_dim=5,
        ),
        Input(
            name="s2",
            input_dim=3,
        ),
    )
    res1, res2 = Reservoir(100), Reservoir(100)
    # model = source1 >> [res1, res2] & source2 >> [res1, res2]
    model = [source1, source2] >> res1 & [source1, source2] >> res2
    outputs = model.run({"s1": np.ones((10, 5)), "s2": np.ones((10, 3))})
