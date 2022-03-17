# Author: Nathan Trouvain at 15/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from reservoirpy.nodes import Identity, ReLU, Sigmoid, Softmax, Softplus, Tanh


@pytest.mark.parametrize(
    "node",
    (Tanh(), Softmax(), Softplus(), Sigmoid(), Identity(), ReLU(), Softmax(beta=2.0)),
)
def test_activation(node):
    x = np.ones((1, 10))
    out = node(x)
    assert out.shape == x.shape
