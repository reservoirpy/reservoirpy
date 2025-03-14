# Author: Paul Bernard on 10/01/2024 <paul.bernard@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from collections import deque
from functools import partial

import numpy as np

from ..node import Node


def forward(node: Node, x, **kwargs):
    node.buffer.appendleft(x)
    output = node.buffer.pop()

    return output


def initialize(node: Node, x=None, y=None, initial_values=None, *args, **kwargs):
    if node.input_dim is not None:
        dim = node.input_dim
    else:
        dim = x.shape[1]

    node.set_input_dim(dim)
    node.set_output_dim(dim)

    if initial_values is None:
        initial_values = np.zeros((node.delay, node.input_dim), dtype=node.dtype)
    node.set_param("buffer", deque(initial_values, maxlen=node.delay + 1))


class Delay(Node):
    """Delays the data transmitted through this node without transformation.

    :py:attr:`Delay.params` **list**

    ============= ======================================================================
    ``buffer``    (:py:class:`collections.deque`) Buffer of the values coming next.
    ============= ======================================================================

    :py:attr:`Delay.hypers` **list**

    ============= ======================================================================
    ``delay``     (int) Number of timesteps before outputting the input.
    ============= ======================================================================

    Parameters
    ----------
    delay: int, defaults to 1.
        Number of timesteps before outputting the input.
    initial_values: array of shape (delay, input_dim), defaults to
        `np.zeros((delay, input_dim))`.
        Initial outputs of the node.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    dtype : Numpy dtype, defaults to np.float64
        Numerical type for node parameters.
    name : str, optional
        Node name.

    Examples
    --------
    >>> x = np.arange(10.).reshape(-1, 1)
    >>> from reservoirpy.nodes import Delay
    >>> delay_node = Delay(
    ...     delay=3,
    ...     initial_values=np.array([[-3.0], [-2.0], [-1.0]])
    ... )
    >>> out = delay_node.run(x)
    >>> print(out.T)
    [[-1. -2. -3.  0.  1.  2.  3.  4.  5.  6.]]
    >>> print(delay_node.buffer)
    deque([array([[9.]]), array([[8.]]), array([[7.]])], maxlen=4)
    """

    def __init__(
        self,
        delay=1,
        initial_values=None,
        input_dim=None,
        dtype=None,
        **kwargs,
    ):
        if input_dim is None and type(initial_values) is np.ndarray:
            input_dim = initial_values.shape[-1]

        super(Delay, self).__init__(
            hypers={"delay": delay},
            params={"buffer": None},
            forward=forward,
            initializer=partial(initialize, initial_values=initial_values),
            input_dim=input_dim,
            **kwargs,
        )
