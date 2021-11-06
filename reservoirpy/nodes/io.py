# Author: Nathan Trouvain at 12/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from contextlib import contextmanager

import numpy as np

from ..base import Node


def _io_initialize(io_node: "Node", x=None, **kwargs):
    if x is not None:
        if io_node.input_dim is None:
            io_node.set_input_dim(x.shape[1])
            io_node.set_output_dim(x.shape[1])
    else:
        if io_node.input_dim is None:
            raise RuntimeError(f"Impossible to infer shape of node {io_node}: "
                               f"no data was fed to the node. Try specify "
                               f"input dimension at node creation.")


def _input_forward(inp_node: "Input", x=None):
    if x is None and inp_node.has_feedback:
        x = inp_node.feedback()
    return x


def _probe_forward(probe, *args, **kwargs):
    result = probe.node._forward(probe.node, *args, **kwargs)
    if probe.is_connected:
        probe._store = np.r_(probe._store, result)
    return result


def _probe_train(probe, *args, **kwargs):
    probe.node._train(probe.node, *args, **kwargs)


def _probe_partial_backward(probe, *args, **kwargs):
    probe.node._partial_backward(probe.node, *args, **kwargs)


def _probe_backward(probe, *args, **kwargs):
    probe.node._backward(probe.node, *args, **kwargs)


def _probe_initializer(probe, *args, **kwargs):
    probe.node._initializer(probe.node, *args, **kwargs)
    probe.set_input_dim(probe.node.input_dim)
    probe.set_output_dim(probe.node.output_dim)

def _probe_fb_initializer(probe, *args, **kwargs):
    probe.node._feedback_initializer(*args, **kwargs)
    probe.set_feedback_dim(probe.node.feedback_dim)


def _probe_buffers_initializer(probe, *args, **kwargs):
    probe.node._buffers_initializer(*args, **kwargs)


class Input(Node):

    def __init__(self, input_dim=None, name=None):
        super(Input, self).__init__(forward=_input_forward,
                                    initializer=_io_initialize,
                                    input_dim=input_dim,
                                    output_dim=input_dim,
                                    name=name)


class Output(Node):

    def __init__(self, name=None):
        super(Output, self).__init__(forward=lambda inp, x: x,
                                     initializer=_io_initialize,
                                     name=name)


class Probe(Node):

    _store: np.ndarray
    _node: Node
    _is_connected: bool

    def __init__(self, node, name=None):

        self._node = node

        probe_train = None
        if node._train is not None:
            probe_train = _probe_train

        probe_partial = None
        if node._partial_backward is not None:
            probe_partial = _probe_partial_backward

        probe_backward = None
        if node._backward is not None:
            probe_backward = _probe_backward

        probe_fb_init = None
        if node._feedback_initializer is not None:
            probe_fb_init = _probe_fb_initializer

        probe_buffers_init = None
        if node._buffers_initializer is not None:
            probe_buffers_init = _probe_buffers_initializer

        super(Probe, self).__init__(name=name,
                                    forward=_probe_forward,
                                    train=probe_train,
                                    partial_backward=probe_partial,
                                    backward=probe_backward,
                                    initializer=_probe_initializer,
                                    fb_initializer=probe_fb_init,
                                    buffers_initializer=probe_buffers_init)
        self.connect(node)

    def __getattr__(self, item):
        param = self.__getattribute__("_node").get_param(item)
        if param is None and item not in self._node._params:
            raise AttributeError(f"No attribute named '{item}' "
                                 f"found in node {self.name}")
        return param

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def node(self):
        return self._node

    @property
    def name(self):
        return self._node.name

    @property
    def params(self):
        return self._node._params

    @property
    def hypers(self):
        return self._node._hypers

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def feedback_dim(self):
        return self._feedback_dim

    @property
    def is_initialized(self):
        return self._node.is_initialized

    @property
    def has_feedback(self):
        return self._node.has_feedback

    @property
    def is_trained_offline(self):
        return self._node.is_trained_offline

    @property
    def is_trained_online(self):
        return self._node.is_trained_online

    @property
    def is_trainable(self):
        return self._node.is_trainable

    @property
    def fitted(self):
        return self._node.fitted

    @is_trainable.setter
    def is_trainable(self, *args, **kwargs):
        self._node.is_trainable(*args, **kwargs)

    @property
    def is_fb_initialized(self):
        return self._node.is_fb_initialized

    def state(self):
        self._node.state()

    def state_proxy(self):
        self._node.state_proxy()

    def feedback(self):
        self._node.feedback()

    def set_state_proxy(self, *args, **kwargs):
        self._node.set_state_proxy(*args, **kwargs)

    def get_param(self, name):
        return self[name]

    def set_param(self, *args, **kwargs):
        self._node.set_param(*args, **kwargs)

    def create_buffer(self, *args, **kwargs):
        self._node.create_buffer(*args, **kwargs)

    def set_buffer(self, *args, **kwargs):
        self._node.set_buffer(*args, **kwargs)

    def get_buffer(self, *args, **kwargs):
        return self._node.get_buffer(*args, **kwargs)

    def reset_probe(self):
        self._store = self._node.state()
        return self

    def reset(self, *args, **kwargs):
        return self._node.reset(*args, **kwargs)

    def reset_feedback(self, *args, **kwargs):
        return self._node.reset_feedback(*args, **kwargs)

    def zero_state(self):
        return self._node.zero_state()

    def zero_feedback(self):
        return self._node.zero_feedback()

    @contextmanager
    def with_state(self, *args, **kwargs):
        with self._node.with_state(*args, **kwargs):
            yield self

    @contextmanager
    def with_feedback(self, *args, **kwargs):
        with self._node.with_feedback(*args, **kwargs):
            yield self

    def link_feedback(self, node):
        return self._node.link_feedback(node)

    def connect(self, node):
        self._node = node
        self.reset()
        self._is_connected = True
        return self
