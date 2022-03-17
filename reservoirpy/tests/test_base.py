# Author: Nathan Trouvain at 15/02/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_equal

from .._base import DistantFeedback, check_n_sequences, check_one_sequence, check_xy
from .dummy_nodes import *


def idfn(val):
    if isinstance(val, np.ndarray):
        return str(val.shape)
    if isinstance(val, list):
        return f"list[{len(val)}]"
    if isinstance(val, dict):
        return str(val)
    else:
        return val


@pytest.mark.parametrize(
    "x,kwargs,expects",
    [
        (np.ones((1, 5)), {}, np.ones((1, 5))),
        (np.ones((5,)), {}, np.ones((1, 5))),
        (np.ones((1, 5)), {"expected_dim": 6}, ValueError),
        ("foo", {}, TypeError),
        (1, {}, np.ones((1, 1))),
        ([np.ones((1, 5)), np.ones((1, 6))], {}, TypeError),
        (np.ones((5, 5)), {}, np.ones((5, 5))),
        (np.ones((2, 5)), {}, np.ones((2, 5))),
        (np.ones((2, 5)), {"expected_dim": 5}, np.ones((2, 5))),
        (np.ones((2, 5)), {"expected_dim": 2}, ValueError),
        (np.ones((2, 5, 2)), {"expected_dim": 2}, ValueError),
        (np.ones((2, 5, 2)), {"expected_dim": (5, 2)}, np.ones((2, 5, 2))),
        (
            [np.ones((2, 5)), np.ones((2, 6))],
            {},
            TypeError,
        ),
    ],
    ids=idfn,
)
def test_check_one_sequence(x, kwargs, expects):
    if isinstance(expects, (np.ndarray, list)):
        x = check_one_sequence(x, **kwargs)
        assert_equal(x, expects)
    else:
        with pytest.raises(expects):
            x = check_one_sequence(x, **kwargs)


@pytest.mark.parametrize(
    "x,kwargs,expects",
    [
        (np.ones((1, 5)), {}, np.ones((1, 5))),
        (np.ones((5,)), {}, np.ones((1, 5))),
        (np.ones((1, 5)), {"expected_dim": (6,)}, ValueError),
        ("foo", {}, TypeError),
        (1, {}, np.ones((1, 1))),
        (
            [np.ones((1, 5)), np.ones((1, 6))],
            {},
            [np.ones((1, 5)), np.ones((1, 6))],
        ),
        (
            [np.ones((5,)), np.ones((6,))],
            {},
            [np.ones((1, 5)), np.ones((1, 6))],
        ),
        (
            [np.ones((1, 5)), np.ones((1, 6))],
            {"expected_dim": (5, 6)},
            [np.ones((1, 5)), np.ones((1, 6))],
        ),
        (
            [np.ones((1, 5)), np.ones((1, 6))],
            {"expected_dim": (6, 5)},
            ValueError,
        ),
        ([np.ones((1, 5)), np.ones((1, 6))], {"expected_dim": (5,)}, ValueError),
        ([np.ones((1, 5)), np.ones((1, 6))], {"expected_dim": (5, 8)}, ValueError),
        (["foo", np.ones((1, 6))], {}, TypeError),
        ([1, np.ones((1, 6))], {}, [np.ones((1, 1)), np.ones((1, 6))]),
        (np.ones((5, 5)), {}, np.ones((5, 5))),
        (np.ones((2, 5)), {}, np.ones((2, 5))),
        (np.ones((2, 5)), {"expected_dim": (5,)}, np.ones((2, 5))),
        (np.ones((2, 5)), {"expected_dim": (2,)}, ValueError),
        (
            [np.ones((2, 5)), np.ones((2, 6))],
            {},
            [np.ones((2, 5)), np.ones((2, 6))],
        ),
        (
            [np.ones((5, 5)), np.ones((5, 6))],
            {"expected_dim": (5, 6)},
            [np.ones((5, 5)), np.ones((5, 6))],
        ),
        ([np.ones((5, 5)), np.ones((5, 6))], {"expected_dim": (5,)}, ValueError),
        ([np.ones((8, 5)), np.ones((8, 6))], {"expected_dim": (5, 8)}, ValueError),
        ([np.ones((7, 5)), np.ones((8, 6))], {"expected_dim": (5, 6)}, ValueError),
        (
            [np.ones((2, 7, 5)), np.ones((2, 8))],
            {"expected_dim": ((7, 5), 8)},
            [np.ones((2, 7, 5)), np.ones((2, 8))],
        ),
        ([np.ones((2, 5)) for _ in range(5)], {}, [np.ones((2, 5)) for _ in range(5)]),
        ([np.ones((5,)) for _ in range(5)], {}, [np.ones((1, 5)) for _ in range(5)]),
        (
            [np.ones((2, 5, 7)) for _ in range(5)],
            {"expected_dim": ((5, 7),)},
            [np.ones((2, 5, 7)) for _ in range(5)],
        ),
        (
            [np.ones((5, 2, 7)), np.ones((5, 2, 6))],
            {"expected_dim": (7, 6)},
            [np.ones((5, 2, 7)), np.ones((5, 2, 6))],
        ),
        (
            [
                [np.ones((i, 7)) for i in range(10)],
                [np.ones((i, 6)) for i in range(10)],
            ],
            {},
            [
                [np.ones((i, 7)) for i in range(10)],
                [np.ones((i, 6)) for i in range(10)],
            ],
        ),
        (
            [
                [np.ones((i, 7)) for i in range(10)],
                [np.ones((i, 6)) for i in range(10)],
            ],
            {"expected_dim": (7, 6)},
            [
                [np.ones((i, 7)) for i in range(10)],
                [np.ones((i, 6)) for i in range(10)],
            ],
        ),
        (np.ones((5, 7, 6)), {"expected_dim": (7, 6)}, ValueError),
    ],
    ids=idfn,
)
def test_check_n_sequences(x, kwargs, expects):
    if isinstance(expects, (np.ndarray, list)):
        x = check_n_sequences(x, **kwargs)
        assert_equal(x, expects)
    else:
        with pytest.raises(expects):
            x = check_n_sequences(x, **kwargs)


@pytest.mark.parametrize(
    "caller,x,y,kwargs,expects",
    [
        (PlusNode(), np.ones((1, 5)), None, {}, (np.ones((1, 5)), None)),
        (PlusNode(), np.ones((5,)), None, {}, (np.ones((1, 5)), None)),
        (PlusNode(), np.ones((1, 5)), None, {"input_dim": (6,)}, ValueError),
        (PlusNode(input_dim=6), np.ones((1, 5)), None, {}, ValueError),
        (
            PlusNode(),
            np.ones((1, 5)),
            np.ones((1, 5)),
            {},
            (np.ones((1, 5)), np.ones((1, 5))),
        ),
        (
            PlusNode(),
            np.ones((5,)),
            np.ones((6,)),
            {},
            (np.ones((1, 5)), np.ones((1, 6))),
        ),
        (
            Offline(),
            np.ones((1, 5)),
            np.ones((1, 6)),
            {"output_dim": (7,)},
            ValueError,
        ),
        (PlusNode(output_dim=7), np.ones((1, 5)), np.ones((1, 6)), {}, ValueError),
        (
            PlusNode(name="plus0") & MinusNode(name="minus0"),
            {"plus0": np.ones((1, 5)), "minus0": np.ones((1, 6))},
            None,
            {},
            ({"plus0": np.ones((1, 5)), "minus0": np.ones((1, 6))}, None),
        ),
        (
            PlusNode(name="plus1") & MinusNode(name="minus1"),
            np.ones((1, 5)),
            None,
            {},
            ({"plus1": np.ones((1, 5)), "minus1": np.ones((1, 5))}, None),
        ),
        (
            PlusNode(name="plus2") & MinusNode(name="minus2"),
            {"plus2": np.ones((1, 5))},
            None,
            {},
            ValueError,
        ),
        (OnlineNode(output_dim=6), np.ones((1, 5)), np.ones((1, 5)), {}, ValueError),
        (
            OnlineNode(),
            np.ones((1, 5)),
            PlusNode(input_dim=5, output_dim=5).initialize(),
            {},
            (np.ones((1, 5)), None),
        ),
        (OnlineNode(), PlusNode(), np.ones((1, 5)), {}, TypeError),
        (Offline(), np.ones((1, 5)), PlusNode(), {}, TypeError),
        (MinusNode(), PlusNode(), np.ones((1, 5)), {}, TypeError),
        (
            OnlineNode(output_dim=6),
            np.ones((1, 5)),
            PlusNode(input_dim=5, output_dim=5).initialize(),
            {},
            ValueError,
        ),
        (
            PlusNode(name="plus3") & MinusNode(name="minus3"),
            {"plus3": np.ones((1, 5)), "minus3": np.ones((1, 6))},
            np.ones((1, 5)),
            {},
            ({"plus3": np.ones((1, 5)), "minus3": np.ones((1, 6))}, None),
        ),
        (
            PlusNode(name="plus4") >> OnlineNode(name="online4"),
            np.ones((1, 5)),
            PlusNode(),
            {},
            ({"plus4": np.ones((1, 5))}, None),
        ),
        (
            PlusNode(name="plus5") >> OnlineNode(name="online51")
            & MinusNode(name="minus5") >> OnlineNode(name="online52"),
            {"plus5": np.ones((1, 5)), "minus5": np.ones((1, 5))},
            {"online51": PlusNode(), "online52": np.ones((1, 5))},
            {},
            (
                {"plus5": np.ones((1, 5)), "minus5": np.ones((1, 5))},
                {"online52": np.ones((1, 5))},
            ),
        ),
        (
            PlusNode(name="plus6") >> OnlineNode(name="online61")
            & MinusNode(name="minus6") >> Offline(name="offline62"),
            {"plus6": np.ones((1, 5)), "minus6": np.ones((1, 5))},
            {"offline62": PlusNode(), "online61": np.ones((1, 5))},
            {},
            TypeError,
        ),
        (
            PlusNode(name="plus8") >> OnlineNode(name="online81")
            & MinusNode(name="minus8") >> Offline(name="offline82"),
            {"plus8": PlusNode(), "minus8": np.ones((1, 5))},
            {"offline82": PlusNode(), "online81": np.ones((1, 5))},
            {},
            TypeError,
        ),
        (
            PlusNode(name="plus7")
            >> Offline().fit(np.ones((10, 5)), np.ones((10, 5)))
            >> Offline(name="off7"),
            np.ones((1, 5)),
            {"off7": np.ones((1, 5))},
            {},
            ({"plus7": np.ones((1, 5))}, {"off7": np.ones((1, 5))}),
        ),
    ],
    ids=idfn,
)
def test_check_xy(caller, x, y, kwargs, expects):
    if isinstance(expects, (np.ndarray, list, dict, tuple)):
        x, y = check_xy(caller, x, y, **kwargs)
        assert_equal(x, expects[0])
        if y is not None:
            assert_equal(y, expects[1])
    else:
        with pytest.raises(expects):
            x = check_xy(caller, x, y, **kwargs)


def test_distant_feedback(plus_node, feedback_node):

    sender = PlusNode(input_dim=5, output_dim=5)
    fb = DistantFeedback(sender, feedback_node)

    fb.initialize()

    assert sender.is_initialized
    assert_equal(sender.state_proxy(), fb())

    fb = DistantFeedback(plus_node, feedback_node)

    with pytest.raises(RuntimeError):
        fb.initialize()

    plus = PlusNode(input_dim=5, output_dim=5) >> Inverter()
    minus = MinusNode(output_dim=5)
    sender = plus >> minus
    fb = DistantFeedback(sender, feedback_node)

    fb.initialize()

    assert sender.is_initialized
    assert_equal(minus.state_proxy(), fb())

    fb = DistantFeedback(plus_node, feedback_node)

    with pytest.raises(RuntimeError):
        fb.initialize()

    plus = PlusNode()
    minus = MinusNode()
    sender = plus >> minus
    fb = DistantFeedback(sender, feedback_node)

    with pytest.raises(RuntimeError):
        fb.initialize()
