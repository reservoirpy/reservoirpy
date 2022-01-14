# Author: Nathan Trouvain at 10/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest

from reservoirpy.node import Node


def plus_forward(node: Node, x: np.ndarray):
    return x + node.c + node.h + node.state()


def plus_initialize(node: Node, x=None, **kwargs):
    node.set_input_dim(x.shape[1])
    node.set_output_dim(x.shape[1])
    node.set_param("c", 1)


class PlusNode(Node):

    def __init__(self):
        super().__init__(params={"c": None}, hypers={"h": 1},
                         forward=plus_forward, initializer=plus_initialize)


def minus_forward(node: Node, x):
    return x - node.c - node.h - node.state()


def minus_initialize(node: Node, x=None, **kwargs):
    node.set_input_dim(x.shape[1])
    node.set_output_dim(x.shape[1])
    node.set_param("c", 1)


class MinusNode(Node):

    def __init__(self):
        super().__init__(params={"c": None}, hypers={"h": 1},
                         forward=minus_forward, initializer=minus_initialize)


def fb_forward(node: Node, x):
    return node.feedback() + x + 1


def fb_initialize(node: Node, x=None, **kwargs):
    node.set_input_dim(x.shape[1])
    node.set_output_dim(x.shape[1])


def fb_initialize_fb(node: Node, fb=None):
    node.set_feedback_dim(fb.shape[1])


class FBNode(Node):

    def __init__(self):
        super().__init__(initializer=fb_initialize,
                         fb_initializer=fb_initialize_fb,
                         forward=fb_forward)


def inv_forward(node: Node, x):
    return -x


def inv_initialize(node: Node, x=None, **kwarg):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


class Inverter(Node):

    def __init__(self):
        super(Inverter, self).__init__(initializer=inv_initialize,
                                       forward=inv_forward)


def off_forward(node: Node, x):
    return x + node.b


def off_partial_backward(node: Node, X_batch, Y_batch=None):
    db = np.mean(np.abs(X_batch - Y_batch))
    b = node.get_buffer("b")
    b += db


def off_backward(node: Node, X=None, Y=None):
    b = node.get_buffer("b")
    node.set_param("b", np.array(b).copy())


def off_initialize(node: Node, x=None, y=None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


def off_initialize_buffers(node: Node):
    node.create_buffer("b", (1,))


class Offline(Node):

    def __init__(self):
        super(Offline, self).__init__(params={"b": 0},
                                      forward=off_forward,
                                      partial_backward=off_partial_backward,
                                      backward=off_backward,
                                      buffers_initializer=off_initialize_buffers,
                                      initializer=off_initialize)


def off2_forward(node: Node, x):
    return x + node.b


def off2_partial_backward(node: Node, X_batch, Y_batch=None):
    db = np.mean(np.abs(X_batch - Y_batch))
    b = node.get_buffer("b")
    b += db


def off2_backward(node: Node, X=None, Y=None):
    b = node.get_buffer("b")
    node.set_param("b", np.array(b).copy())


def off2_initialize(node: Node, x=None, y=None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


def off2_initialize_buffers(node: Node):
    node.create_buffer("b", (1,))


class Offline2(Node):

    def __init__(self):
        super(Offline2, self).__init__(params={"b": 0},
                                       forward=off2_forward,
                                       partial_backward=off2_partial_backward,
                                       backward=off2_backward,
                                       initializer=off2_initialize,
                                       buffers_initializer=off2_initialize_buffers)


def sum_forward(node: Node, x):
    if isinstance(x, list):
        x = np.concatenate(x, axis=0)
    return np.sum(x, axis=0)


def sum_initialize(node: Node, x=None, **kwargs):
    if x is not None:
        if isinstance(x, list):
            x = np.concatenate(x, axis=0)
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


class Sum(Node):

    def __init__(self):
        super(Sum, self).__init__(forward=sum_forward,
                                  initializer=sum_initialize)


def unsupervised_forward(node: Node, x):
    return x + node.b


def unsupervised_partial_backward(node: Node, X_batch, Y_batch=None):
    b = np.mean(X_batch)
    node.set_buffer("b", node.get_buffer("b") + b)


def unsupervised_backward(node: Node, X=None, Y=None):
    b = node.get_buffer("b")
    node.set_param("b", np.array(b).copy())


def unsupervised_initialize(node: Node, x=None, y=None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


def unsupervised_initialize_buffers(node: Node):
    node.create_buffer("b", (1,))


class Unsupervised(Node):

    def __init__(self):
        super(Unsupervised, self).__init__(params={"b": 0},
                                           forward=unsupervised_forward,
                                           partial_backward=unsupervised_partial_backward,
                                           backward=unsupervised_backward,
                                           initializer=unsupervised_initialize,
                                           buffers_initializer=unsupervised_initialize_buffers)


def on_forward(node: Node, x):
    return x + node.b


def on_train(node: Node, x, y=None):
    if y is not None:
        node.set_param("b", node.b + np.mean(x+y))
    else:
        node.set_param("b", node.b + np.mean(x))


def on_initialize(node: Node, x=None, y=None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1])


class OnlineNode(Node):
    def __init__(self):
        super(OnlineNode, self).__init__(params={"b": np.array([0])},
                                           forward=on_forward,
                                           train=on_train,
                                           initializer=on_initialize)


def clean_registry(node_class):
    node_class._registry = []
    node_class._factory_id = -1


@pytest.fixture(scope="function")
def plus_node():
    clean_registry(PlusNode)
    return PlusNode()


@pytest.fixture(scope="function")
def minus_node():
    clean_registry(MinusNode)
    return MinusNode()


@pytest.fixture(scope="function")
def feedback_node():
    clean_registry(FBNode)
    return FBNode()


@pytest.fixture(scope="function")
def inverter_node():
    clean_registry(Inverter)
    return Inverter()


@pytest.fixture(scope="function")
def offline_node():
    clean_registry(Offline)
    return Offline()


@pytest.fixture(scope="function")
def offline_node2():
    clean_registry(Offline2)
    return Offline2()


@pytest.fixture(scope="function")
def sum_node():
    clean_registry(Sum)
    return Sum()


@pytest.fixture(scope="function")
def unsupervised_node():
    clean_registry(Unsupervised)
    return Unsupervised()


@pytest.fixture(scope="function")
def online_node():
    clean_registry(OnlineNode)
    return OnlineNode()
