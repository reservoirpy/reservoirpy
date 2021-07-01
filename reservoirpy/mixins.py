# Author: Nathan Trouvain at 14/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Callable


class FeedbackReceiver:

    _feedback: Callable
    _feedback_dim: int
    _feedback_initializer: Callable
    is_fb_initialized: bool

    def __init__(self, fb_initializer, *args, **kwargs):
        self._feedback_initializer = fb_initializer
        super(FeedbackReceiver, self).__init__(*args, **kwargs)

    @property
    def feedback(self):
        return self._feedback

    @property
    def feedabck_dim(self):
        return self._feedback_dim

    def initialize_feedback(self):
        self._feedback_initializer(self)

    def set_feedback_dim(self, value):
        if not self.is_initialized:
            self._feedback_dim = value
        else:
            raise TypeError(f"Output dimension of {self.name} is "
                            "immutable after initialization.")

    def link_feedback(self, node):
        if not isinstance(self, FeedbackReceiver):
            raise TypeError(f"Node {self.name} does not accept any "
                            f"feedback connections.")
        self._feedback = node.state
        return self

    def __lshift__(self, other):
        return self.link_feedback(other)
