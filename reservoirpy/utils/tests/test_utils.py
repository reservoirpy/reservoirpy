# Author: Nathan Trouvain at 25/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import defaultdict

import pytest
from tqdm import tqdm

from reservoirpy.utils import (
    _obj_from_kwargs,
    progress,
    safe_defaultdict_copy,
    verbosity,
)


def test_verbosity():
    v = verbosity()
    from reservoirpy.utils import VERBOSITY

    assert v == VERBOSITY
    verbosity(0)
    from reservoirpy.utils import VERBOSITY

    assert VERBOSITY == 0


def test_progress():
    verbosity(0)
    a = [1, 2, 3]
    it = progress(a)
    assert id(it) == id(a)

    verbosity(1)
    it = progress(a)
    assert isinstance(it, tqdm)


def test_defaultdict_copy():

    a = defaultdict(list)

    a["a"].extend([1, 2, 3])
    a["b"] = 2

    b = safe_defaultdict_copy(a)

    assert list(b.values()) == [
        [1, 2, 3],
        [
            2,
        ],
    ]
    assert list(b.keys()) == ["a", "b"]

    a = dict()

    a["a"] = [1, 2, 3]
    a["b"] = 2

    b = safe_defaultdict_copy(a)

    assert list(b.values()) == [
        [1, 2, 3],
        [
            2,
        ],
    ]
    assert list(b.keys()) == ["a", "b"]


def test_obj_from_kwargs():
    class A:
        def __init__(self, a=0, b=2):
            self.a = a
            self.b = b

    a = _obj_from_kwargs(A, {"a": 1})
    assert a.a == 1
    assert a.b == 2
