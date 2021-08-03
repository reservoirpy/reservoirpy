# Author: Nathan Trouvain at 07/07/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Callable
from contextlib import contextmanager

import numpy as np

from .node import Node, Model
from .utils.validation import check_vector


"""""""""""
class FeedbackReceiver(Node):

    _feedback: Node
    _feedback_dim: int
    _feedback_initializer: Callable
    _is_fb_initialized: bool = False

    def __init__(self, fb_initializer, *args, **kwargs):
        self._feedback_initializer = fb_initializer
        self._feedback = None
        self._is_fb_initialized = False
        super(FeedbackReceiver, self).__init__(*args, **kwargs)

    def _check_feedback(self, fb):
        fb = check_vector(fb)

        if self._is_initialized:
            if fb.shape[1] != self.input_dim:
                raise ValueError(f"Impossible to call node {self.name}: node input "
                                 f"dimension is (1, {self.input_dim}) and input dimension "
                                 f"is {fb.shape}.")
        return fb

    @property
    def feedback_dim(self):
        return self._feedback_dim

    @property
    def has_feedback(self):
        return self._feedback is not None

    @property
    def is_fb_initialized(self):
        return self._is_fb_initialized

    def feedback(self):
        if not self._feedback.is_initialized:
            raise RuntimeError(f"Impossible to get feedback "
                               f"from node or model {self._feedback} "
                               f"to node {self.name}: {self._feedback.name} "
                               f"is not initialized.")

        if isinstance(self._feedback, Node):
            return self._feedback.state_proxy()
        else:
            if isinstance(self._feedback, Model):
                mapping = {c.name: p.state_proxy()
                           for p, c in self._feedback.edges
                           if p in self._feedback.input_nodes}
                return self._feedback.call(mapping)

    def zero_feedback(self):
        return self._feedback.zero_state()

    def initialize_feedback(self):
        self._feedback_initializer(self)
        self._is_fb_initialized = True

    def set_feedback_dim(self, value):
        if not self.is_fb_initialized:
            self._feedback_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def link_feedback(self, node):
        if not isinstance(node, Node):
            raise TypeError(f"Impossible to receive feedback from {node}: "
                            f"it is not a Node instance.")
        self._feedback = node
        return self

    def reset_feedback(self, to_feedback=None):
        if to_feedback is None:
            self._feedback.reset()
        else:
            self._feedback.reset(to_state=to_feedback)

    @contextmanager
    def with_feedback(self, feedback=None, stateful=False, reset=False):

        if self.has_feedback:
            current_fb = self._feedback

            if feedback is None:
                if reset:
                    feedback = self.zero_feedback()
                else:
                    feedback = current_fb

            if isinstance(feedback, Node):
                self._feedback = feedback
                yield self
                self._feedback = current_fb
            elif isinstance(feedback, np.ndarray):
                current_proxy = self._feedback._state_proxy
                self._feedback.set_state_proxy(feedback)
                yield self
                if not stateful:
                    self._feedback._state_proxy = current_proxy
            else:
                raise TypeError(f"Impossible to get feedback from {feedback}: "
                                f"it is neither a Node instance nor an array.")
        else:
            yield self
    def call(self, x, forced_feedback=None, from_state=None, stateful=True, reset=False):
        x = self._check_input(x)
        if forced_feedback is not None:
            if not isinstance(forced_feedback, Node):
                feedback = check_vector(forced_feedback)

        if not self._is_initialized:
            self.initialize(x, feedback=forced_feedback)

        with self.with_state(from_state, stateful=stateful, reset=reset):
            with self.with_feedback(forced_feedback, reset=reset):
                state = self._forward(self, x)

        return state

    def run(self, X, forced_feedbacks=None, from_state=None, stateful=True, reset=False):
        # TODO: check that X and feedbacks are arrays with correct shape
        if not self._is_initialized:
            # send a probe to infer shapes and initialize params
            if forced_feedbacks is None:
                self.call(X[0])
            else:
                self.call(X[0], forced_feedbacks[0])

        with self.with_state(from_state, stateful=stateful, reset=reset):
            states = np.zeros((X.shape[0], self.output_dim))
            for i, x in enumerate(X):
                if forced_feedbacks is None:
                    s = self.call(x)
                else:
                    s = self.call(x, forced_feedback=forced_feedbacks[i])
                states[i, :] = s

        return states

    def __lshift__(self, other):
        return self.link_feedback(other)
"""""""""""

class TrainableOffline(Node):

    _backward: Callable

    def __init__(self, backward, partial_backward=None, *args, **kwargs):
        self._backward = backward
        self._partial_backward = partial_backward
        super(TrainableOffline, self).__init__(*args, **kwargs)

    def _check_output(self, y):
        y = check_vector(y)

        if self._is_initialized:
            if y.shape[1] != self.output_dim:
                raise ValueError(f"Impossible to fit node {self.name}: node output "
                                 f"dimension is (1, {self.output_dim}) and target dimension "
                                 f"is {y.shape}.")
        return y

    def fit(self, x=None, y=None):
        if x is not None:
            x = self._check_input(x)
        if y is not None:
            y = self._check_output(y)

        if not self._is_initialized:
            self.initialize(x=x, y=y)

        self._backward(self, x, y)

        # in case of teacher forcing,
        # expose the target vector
        self.set_state_proxy(y)

        return self

    def partial_fit(self, x, y):
        x = self._check_input(x)
        y = self._check_output(y)

        if not self._is_initialized:
            self.initialize(x=x, y=y)

        if self._partial_backward is not None:
            self._partial_backward(self, x, y)

        # in case of teacher forcing,
        # expose the target vector
        self.set_state_proxy(y)

        return self


class TrainableOnline(Node):

    def fit(self, X, Y):
        ...

    def partial_fit(self, x, y):
        ...

    def call(self, x, y=None, fit=True):
        ...

    def run(self, X, Y=None, fit=True):
        ...
