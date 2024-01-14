import numpy as np

from ..delay import Delay
from ..readouts import Ridge


def test_no_delay():
    delay_node = Delay(delay=0)

    x = np.array([0.2, 0.3])
    y = delay_node(x)
    assert np.all(x == y)

    x = np.linspace(1, 12, num=12).reshape(-1, 2)
    y = delay_node.run(x)
    assert np.all(x == y)


def test_1_delay():
    delay_node = Delay(delay=1)

    x1 = np.array([0.2, 0.3])
    y = delay_node(x1)
    assert np.all(y == 0.0)

    x2 = np.linspace(1, 12, num=12).reshape(-1, 2)
    y = delay_node.run(x2)
    assert np.all(y[0] == x1)
    assert np.all(y[1:] == x2[:-1])


def test_large_delay():
    delay_node = Delay(delay=1_000)

    x = np.array([0.2, 0.3])
    y = delay_node(x)
    assert np.all(y == 0.0)
    assert np.all(delay_node.buffer[0] == x)
    assert np.all(delay_node.buffer[-1] == 0.0)

    delay_node.run(np.zeros((999, 2)))
    y = delay_node(np.zeros((1, 2)))
    assert np.all(y == x)


def test_multiseries_delay():
    delay_node = Delay(delay=2)
    readout = Ridge(ridge=1e-3)
    model = delay_node >> readout

    x = list(np.fromfunction(lambda i, j, k: i + j, (2, 4, 2)))
    y = list(np.fromfunction(lambda i, j, k: i + j, (2, 4, 1)))

    model.fit(x, y, warmup=2)
