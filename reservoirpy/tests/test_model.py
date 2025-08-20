# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.nodes.io import Input, Output

from ..model import Model
from ..ops import merge
from .dummy_nodes import Inverter, MinusNode, Offline, OnlineUnsupervised, PlusNode


def test_node_initialize():
    plus_node = PlusNode()
    minus_node = MinusNode()
    x = np.ones((10, 2))

    # model1 = plus_node >> minus_node
    model1 = Model([plus_node, minus_node], [(plus_node, 0, minus_node)])
    model1.initialize(x)

    # model2 = minus_node >> plus_node
    model2 = Model([plus_node, minus_node], [(minus_node, 0, plus_node)])
    model2.initialize(x)

    assert set(model1.nodes) == set(model2.nodes)

    # 2-circular graph
    with pytest.raises(RuntimeError):
        _model3 = Model(
            [plus_node, minus_node],
            [(minus_node, 0, plus_node), (plus_node, 0, minus_node)],
        )

    # 1-circular graph
    with pytest.raises(RuntimeError):
        _model4 = Model([plus_node, minus_node], [(plus_node, 0, plus_node)])


def test_multi_input():
    plus_node = PlusNode()
    x = np.ones((5,))

    # Basic multi-input
    input1, input2 = Input(name="Input1"), Input(name="Input2")
    model5 = Model(
        [input1, input2, plus_node], [(input1, 0, plus_node), (input2, 0, plus_node)]
    )
    model5.initialize({"Input1": x, "Input2": x})
    assert input1.input_dim == input2.input_dim == 5

    # multiple input but not named
    input1, input2 = Input(name="Input1"), Input()
    model5 = Model(
        [input1, input2, plus_node], [(input1, 0, plus_node), (input2, 0, plus_node)]
    )
    with pytest.raises(ValueError):
        model5.initialize({"Input1": x, "Input2": x})


def test_multi_output():
    plus_node = PlusNode()
    x = np.ones((5,))

    # Basic multi-output
    output1, output2 = Output(name="Output1"), Output(name="Output2")
    model5 = Model(
        [output1, output2, plus_node],
        [(plus_node, 0, output1), (plus_node, 0, output2)],
    )
    # res = model5(x)
    # assert output1.input_dim == output2.input_dim == 5
    # assert isinstance(res, dict)
    # assert "Output1" in res
    # assert "Output2" in res
    model5.initialize(x)

    # multiple input but not named
    output1, output2 = Output(name="Output1"), Output()
    model5 = Model(
        [output1, output2, plus_node],
        [(plus_node, 0, output1), (plus_node, 0, output2)],
    )
    with pytest.raises(ValueError):
        model5.initialize(x)


def test_complex_node_link():
    x = np.ones((10, 2))
    A = PlusNode(name="A")
    B = PlusNode(name="B")
    C = PlusNode(name="C")
    D = PlusNode(name="D")
    E = PlusNode(name="E")
    F = PlusNode(name="F")
    In = Input(name="In")
    Out = Output(name="Out")

    path1, path2 = A >> F, B >> E
    path3 = In >> [A, B, C]
    path4 = A >> B >> C >> D >> E >> F >> Out
    model = path1 & path2 & path3 & path4

    assert model.nodes == [A, F, B, E, In, C, D, Out]
    assert len(model.edges) == 11
    assert model.inputs == [In]
    assert model.outputs == [Out]
    assert len(model.named_nodes) == 8 and model.named_nodes["A"] == A
    assert not model.is_trainable
    assert not model.is_multi_input
    assert not model.is_multi_output

    model.initialize(x)

    assert len(model.parents[B]) == 2
    assert len(model.children[In]) == 3
    assert len(model.execution_order) == len(model.nodes)

    expected_dims = [2, 12, 4, 10, 2, 6, 6, 12]
    assert all([node.initialized for node in model.nodes])
    assert [node.input_dim for node in model.nodes] == expected_dims


def test_model_complex():
    from ..nodes import Reservoir, Ridge

    r1 = Reservoir(10)
    r2 = Reservoir(10)
    read1 = Ridge()
    read2 = Ridge()
    read3 = Ridge()

    model = r1 >> read1 >> r2 >> read2 >> read3
    model &= r2 & [(r1 << r2), (r1 << read1), read1] & (r2 << read2) & r2

    assert set(model.edges) == {
        (r1, 0, read1),
        (read1, 0, r2),
        (r2, 0, read2),
        (read2, 0, read3),
        (r2, 1, r1),
        (read1, 1, r1),
        (read2, 1, r2),
    }

    r1 = Reservoir(10)
    r2 = Reservoir(10)
    read1 = Ridge()
    read2 = Ridge()

    model = (r1 >> read1) & (r2 >> read2)
    model &= [r1, r2] << read1

    assert set(model.edges) == {
        (r1, 0, read1),
        (r2, 0, read2),
        (read1, 1, r1),
        (read1, 1, r2),
    }

    r1 = Reservoir(10)
    read1 = Ridge()
    model = r1 << (r1 >> read1)

    assert set(model.edges) == {(r1, 0, read1), (read1, 1, r1)}

    model = [r2] & (r1 >> read1)
    model = (r2 >> read1) & [r1]
    model = r2 & (r1 >> read1)
    model = (r2 >> read1) & r1
    model = [r2] >> (r1 >> read1)
    model = (r2 >> read1) >> [r1]
    model = r2 >> (r1 >> read1)
    model = (r2 >> read1) >> r1
    model = [r2] << (r1 >> read1)
    model = (r2 >> read1) << [r1]
    model = r2 << (r1 >> read1)
    model = (r2 >> read1) << r1


def test_model_call():
    data = np.zeros((5,))

    plus_node = PlusNode(h=2, name="Plus")
    minus_node = MinusNode(h=2, name="Minus")
    model = plus_node >> minus_node
    res = model(data)
    assert_array_equal(res, data)

    input_node = Input()
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node
    model = branch1 & branch2
    res = model(data)
    assert model.outputs == [plus_node, minus_node]
    assert_array_equal(res["Plus"], data + 2)
    assert_array_equal(res["Minus"], data - 2)

    plus_node = PlusNode(h=2, name="Plus")
    minus_node = MinusNode(h=2, name="Minus")
    model = plus_node >> minus_node
    res = model(None)
    assert res.shape == (0,)

    plus_node1 = PlusNode(h=1, name="Plus1")
    plus_node2 = PlusNode(h=2, name="Plus2")
    minus_node = MinusNode(h=2, name="Minus")
    model = (plus_node1 >> minus_node) & (plus_node2 >> minus_node)
    res = model({"Plus1": data, "Plus2": data})
    assert res.shape == (10,)


def test_model_run():
    input_node = Input()
    plus_node = PlusNode(name="plus")
    minus_node = MinusNode(name="minus")
    branch1 = input_node >> plus_node
    branch2 = input_node >> minus_node

    model = merge(branch1, branch2)

    data = np.zeros((3, 5))
    res = model.run(data)

    expected_plus = np.ones((3, 5))
    expected_minus = -np.ones((3, 5))

    for name, arr in res.items():
        assert name in [out.name for out in model.outputs]
        if name == "plus":
            assert_array_equal(arr, expected_plus)
            assert_array_equal(arr[-1], plus_node.state["out"])
        else:
            assert_array_equal(arr, expected_minus)
            assert_array_equal(arr[-1], minus_node.state["out"])


def test_model_run_on_sequences():
    input_node = Input()
    plus_node = PlusNode(name="plus")
    minus_node = MinusNode(name="minus")
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


def test_offline_fit_simple_model():
    plus_node = PlusNode()
    offline_node = Offline()
    model = plus_node >> offline_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.fit(X, Y)

    assert offline_node.b == 25

    X = np.ones((3, 5, 5)) * 0.5
    Y = np.ones((3, 5, 5))

    model.fit(X, Y)

    assert offline_node.b == 75


def test_offline_fit_complex():
    plus_node = PlusNode()
    minus_node = MinusNode()
    offline_node1 = Offline(name="off1")
    offline_node2 = Offline(name="off2")
    model = [plus_node >> offline_node1, plus_node] >> minus_node >> offline_node2

    X = np.ones((5, 5, 5)) * 0.5
    Y_1 = np.ones((5, 5, 5))
    Y_2 = np.ones((5, 5, 10))  # after concat

    model.fit(X, y={"off1": Y_1, "off2": Y_2})

    assert plus_node.input_dim == 5
    assert plus_node.output_dim == 5
    assert minus_node.input_dim == 10
    assert minus_node.output_dim == 10
    assert offline_node1.input_dim == 5
    assert offline_node1.output_dim == 5
    assert offline_node2.input_dim == 10
    assert offline_node2.output_dim == 10

    res = model.run(X[0])

    assert res.shape == (5, 10)


def test_online_train_simple():
    plus_node = PlusNode()
    online_node = OnlineUnsupervised()
    model = plus_node >> online_node

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.partial_fit(X)

    assert online_node.b == 37.5

    model.partial_fit(X)

    assert online_node.b == 75.0

    # model.fit reinitializes variables
    model.fit(X)

    assert online_node.b == 37.5


def test_model_online():
    from reservoirpy.nodes import RLS, Output, Reservoir

    reservoir1 = Reservoir(10, name="r1")
    reservoir2 = Reservoir(10, name="r2")
    output = Output(name="out")
    readout = RLS(alpha=1e-4, name="rls")
    model = [reservoir1, reservoir2] >> readout & reservoir1 >> output

    X = np.ones((5, 5)) * 0.5
    Y = np.ones((5, 5))

    model.partial_fit({"r1": X, "r2": X}, Y)
    Wout1 = readout.Wout

    model_output = model.partial_fit({"r1": X, "r2": X}, Y)
    Wout2 = readout.Wout
    assert np.any(Wout1 != Wout2)

    assert len(model_output) == 2
    assert model_output["out"].shape == (5, 10)
    assert model_output["rls"].shape == (5, 5)


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


def test_multiinputs():
    import numpy as np

    from reservoirpy.nodes import Input, Reservoir

    source1, source2 = Input(name="s1"), Input(name="s2")
    res1, res2 = Reservoir(100, name="res1"), Reservoir(100, name="res2")
    # model = source1 >> [res1, res2] & source2 >> [res1, res2]
    model = [source1, source2] >> res1 & [source1, source2] >> res2
    outputs = model.run({"s1": np.ones((10, 5)), "s2": np.ones((10, 3))})
