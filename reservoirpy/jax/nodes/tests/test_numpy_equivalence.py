# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""
Each Jax node must give the same results as its NumPy counterpart.

Both versions of a node are built with the same parameters and seed (hence the same matrices), given the same data,
and their results are compared:
  - nodes without training: the outputs of run;
  - trainable nodes: the learned parameters and the outputs of run after fit, for each form of input that fit accepts,
    then after a second fit on other data (which checks that the node uses its new parameters).
"""
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy.jax.nodes as jax_nodes
import reservoirpy.nodes as numpy_nodes
from reservoirpy.node import TrainableNode

rng = np.random.default_rng(0)
X, Y = rng.normal(size=(200, 5)), rng.normal(size=(200, 2))
X_OTHER, Y_OTHER = rng.normal(size=(200, 5)), rng.normal(size=(200, 2))


def numpy_and_jax(name, **params):
    """The node `name`, built with reservoirpy.nodes and with reservoirpy.jax.nodes."""
    return getattr(numpy_nodes, name)(**params), getattr(jax_nodes, name)(**params)

def assert_same(jax_result, numpy_result):
    assert_allclose(np.asarray(jax_result, dtype=float), np.asarray(numpy_result, dtype=float), rtol=1e-4, atol=1e-5)



# Nodes without training
NODES = {
    "Reservoir": dict(units=100, sr=0.9, lr=0.7, input_scaling=0.5, bias=0.1, seed=0),
    "ES2N": dict(units=100, sr=0.9, proximity=0.5, input_scaling=0.5, seed=0),
    "LIF": dict(units=100, sr=0.9, lr=0.1, inhibitory=0.2, input_scaling=0.5, seed=0),
    "NVAR": dict(delay=3, order=2, strides=2),
}

@pytest.mark.parametrize("name", NODES)
def test_run(name):
    numpy_node, jax_node = numpy_and_jax(name, **NODES[name])

    assert_same(jax_node.run(X), numpy_node.run(X))


# Trainable nodes
@dataclass
class Trainable:
    node: str
    params: dict
    learned: tuple  # attributes set by fit
    supervised: bool = True


TRAINABLE = {
    "IPReservoir-tanh": Trainable(
        "IPReservoir",
        dict(units=100, sr=0.9, activation="tanh", mu=0.0, sigma=0.5, learning_rate=1e-2, seed=0),
        learned=("a", "b"),
        supervised=False,
    ),
    "IPReservoir-sigmoid": Trainable(
        "IPReservoir",
        dict(units=100, sr=0.9, activation="sigmoid", mu=0.2, learning_rate=1e-2, seed=0),
        learned=("a", "b"),
        supervised=False,
    ),
    "Ridge": Trainable("Ridge", dict(ridge=1e-3), learned=("Wout", "bias")),
    "LMS": Trainable("LMS", dict(learning_rate=1e-2), learned=("Wout", "bias")),
    "RLS": Trainable("RLS", dict(alpha=1e-1), learned=("Wout", "bias")),
}


# Forms of input accepted by fit, made from x and y: (x, y, warmup)
def one_timeseries(x, y):
    return x, y, 0
def array_of_timeseries(x, y):
    return np.stack([x[:100], x[100:]]), np.stack([y[:100], y[100:]]), 10
def list_of_timeseries_of_different_lengths(x, y):
    return [x[:120], x[120:]], [y[:120], y[120:]], 10


@pytest.mark.parametrize(
    "fit_input", [one_timeseries, array_of_timeseries, list_of_timeseries_of_different_lengths], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("case", TRAINABLE)
def test_fit(case, fit_input):
    trainable = TRAINABLE[case]
    numpy_node, jax_node = numpy_and_jax(trainable.node, **trainable.params)

    # fit, then fit again on other data
    for x, y in [(X, Y), (X_OTHER, Y_OTHER)]:
        x, y, warmup = fit_input(x, y)
        if not trainable.supervised:
            y = None
        numpy_node.fit(x, y, warmup=warmup)
        jax_node.fit(x, y, warmup=warmup)

        for attribute in trainable.learned:
            assert_same(getattr(jax_node, attribute), getattr(numpy_node, attribute))

        numpy_node.reset()
        jax_node.reset()
        assert_same(jax_node.run(X[:50]), numpy_node.run(X[:50]))


# Check if some nodes haven't been tested yet
def test_every_trainable_node_is_tested():
    trainable_nodes = { name for name, node in vars(jax_nodes).items() if isinstance(node, type) and issubclass(node, TrainableNode)}
    tested = {trainable.node for trainable in TRAINABLE.values()}

    assert trainable_nodes <= tested, f"No fit test for {trainable_nodes - tested}"
