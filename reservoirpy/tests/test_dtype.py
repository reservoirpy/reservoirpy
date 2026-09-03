# Licence: MIT License

import numpy as np
import pytest

from ..nodes import Reservoir
from ..nodes.local_plasticity_reservoir import LocalPlasticityReservoir

DTYPE_NODES = [Reservoir, LocalPlasticityReservoir]


@pytest.mark.parametrize("Node", DTYPE_NODES)
def test_dtype_reaches_state_run_and_reset(Node):
    node = Node(20, dtype=np.float32)
    out = node.run(np.ones((8, 2), dtype=np.float32))

    assert node.W.dtype == np.float32
    assert node.state["out"].dtype == np.float32
    assert np.asarray(out).dtype == np.float32

    node.reset()
    assert node.state["out"].dtype == np.float32


@pytest.mark.parametrize("Node", DTYPE_NODES)
def test_default_dtype_is_unchanged(Node):
    node = Node(20)
    node.run(np.ones((8, 2)))
    assert node.state["out"].dtype == np.float64
    node.reset()
    assert node.state["out"].dtype == np.float64


def test_reset_preserves_state_dtype_for_any_node():
    # Node.reset() must not silently change the dtype of the state it restores,
    # whatever that dtype is.
    node = Reservoir(15, dtype=np.float32)
    node.run(np.ones((5, 1), dtype=np.float32))
    before = {k: v.dtype for k, v in node.state.items()}
    node.reset()
    after = {k: v.dtype for k, v in node.state.items()}
    assert before == after


def test_run_output_matches_state_dtype():
    node = Reservoir(15, dtype=np.float32)
    out = node.run(np.ones((5, 1), dtype=np.float32))
    assert np.asarray(out).dtype == node.state["out"].dtype
