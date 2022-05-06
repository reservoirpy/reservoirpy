# Author: Nathan Trouvain at 15/02/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Sequence, Union
from uuid import uuid4

import numpy as np

from .type import MappedData, Shape
from .utils import progress
from .utils.validation import check_vector, is_mapping


def _distant_model_inputs(model):
    """Get inputs for distant Nodes in a Model used as feedabck or teacher.
    These inputs should be already computed by other Nodes."""
    input_data = {}
    for p, c in model.edges:
        if p in model.input_nodes:
            input_data[c.name] = p.state_proxy()
    return input_data


def _remove_input_for_feedback(node) -> Union["Node", "Model"]:
    """Remove inputs nodes from feedback Model and gather remaining nodes
    into a new Model. Allow getting inputs for feedback model from its input
    nodes states."""
    from .model import Model

    all_nodes = set(node.nodes)
    input_nodes = set(node.input_nodes)
    filtered_nodes = list(all_nodes - input_nodes)
    filtered_edges = [edge for edge in node.edges if edge[0] not in input_nodes]

    # return a single Node if Model - Inputs = Node
    # else return Model - Inputs = Reduced Model
    if len(filtered_nodes) == 1:
        return list(filtered_nodes)[0]
    return Model(filtered_nodes, filtered_edges, name=str(uuid4()))


def check_one_sequence(
    x: Union[np.ndarray, Sequence[np.ndarray]],
    expected_dim=None,
    caller=None,
    allow_timespans=True,
):

    caller_name = caller.name + "is " if caller is not None else ""

    if expected_dim is not None and not hasattr(expected_dim, "__iter__"):
        expected_dim = (expected_dim,)

    x_new = check_vector(
        x, allow_reshape=True, allow_timespans=allow_timespans, caller=caller
    )
    data_dim = x_new.shape[1:]

    # Check x dimension
    if expected_dim is not None:
        if len(expected_dim) != len(data_dim):
            raise ValueError(
                f"{caller_name} expecting {len(expected_dim)} inputs "
                f"but received {len(data_dim)}: {x_new}."
            )
        for dim in expected_dim:
            if all([dim != ddim for ddim in data_dim]):
                raise ValueError(
                    f"{caller_name} expecting data of shape "
                    f"{expected_dim} but received shape {data_dim}."
                )
    return x_new


# expected_dim = ((m, n), o, (p, q, r), ...)
def check_n_sequences(
    x,
    expected_dim=None,
    allow_n_sequences=True,
    allow_n_inputs=True,
    allow_timespans=True,
    caller=None,
):
    if expected_dim is not None:
        if not hasattr(expected_dim, "__iter__"):
            expected_dim = (expected_dim,)
        n_inputs = len(expected_dim)

        # I
        if n_inputs > 1:
            if isinstance(x, (list, tuple)):
                x_new = [x[i] for i in range(len(x))]
                timesteps = []
                for i in range(n_inputs):
                    dim = (expected_dim[i],)
                    x_new[i] = check_n_sequences(
                        x[i],
                        expected_dim=dim,
                        caller=caller,
                        allow_n_sequences=allow_n_sequences,
                        allow_timespans=allow_timespans,
                        allow_n_inputs=allow_n_inputs,
                    )
                    if isinstance(x_new[i], (list, tuple)):
                        timesteps.append(tuple([x_.shape[0] for x_ in x_new[i]]))
                    else:
                        dim = dim[0]
                        if not hasattr(dim, "__len__"):
                            dim = (dim,)
                        if len(dim) + 2 > len(x_new[i].shape) >= len(dim) + 1:
                            timesteps.append((x_new[i].shape[0],))
                        else:
                            timesteps.append((x_new[i].shape[1],))

                if len(np.unique([len(t) for t in timesteps])) > 1 or any(
                    [
                        len(np.unique([t[i] for t in timesteps])) > 1
                        for i in range(len(timesteps[0]))
                    ]
                ):
                    raise ValueError("Inputs with different timesteps")
            else:
                raise ValueError("Expecting several inputs.")
        else:  # L
            dim = expected_dim[0]
            if not hasattr(dim, "__len__"):
                dim = (dim,)

            if isinstance(x, (list, tuple)):
                if not allow_n_sequences:
                    raise TypeError("No lists, only arrays.")
                x_new = [x[i] for i in range(len(x))]
                for i in range(len(x)):
                    x_new[i] = check_one_sequence(
                        x[i],
                        allow_timespans=allow_timespans,
                        expected_dim=dim,
                        caller=caller,
                    )
            else:
                if len(x.shape) <= len(dim) + 1:  # only one sequence
                    x_new = check_one_sequence(
                        x,
                        expected_dim=dim,
                        allow_timespans=allow_timespans,
                        caller=caller,
                    )
                elif len(x.shape) == len(dim) + 2:  # several sequences
                    if not allow_n_sequences:
                        raise TypeError("No lists, only arrays.")
                    x_new = x
                    for i in range(len(x)):
                        x_new[i] = check_one_sequence(
                            x[i],
                            allow_timespans=allow_timespans,
                            expected_dim=dim,
                            caller=caller,
                        )
                else:  # pragma: no cover
                    x_new = check_vector(
                        x,
                        allow_reshape=True,
                        allow_timespans=allow_timespans,
                        caller=caller,
                    )
    else:
        if isinstance(x, (list, tuple)):
            x_new = [x[i] for i in range(len(x))]
            for i in range(len(x)):
                if allow_n_inputs:
                    x_new[i] = check_n_sequences(
                        x[i],
                        allow_n_sequences=allow_n_sequences,
                        allow_timespans=allow_timespans,
                        allow_n_inputs=False,
                        caller=caller,
                    )
                elif allow_n_sequences:
                    x_new[i] = check_n_sequences(
                        x[i],
                        allow_n_sequences=False,
                        allow_timespans=allow_timespans,
                        allow_n_inputs=False,
                        caller=caller,
                    )
                else:
                    raise ValueError("No lists, only arrays.")
        else:
            x_new = check_one_sequence(
                x, allow_timespans=allow_timespans, caller=caller
            )

    return x_new


def _check_node_io(
    x,
    receiver_nodes=None,
    expected_dim=None,
    caller=None,
    io_type="input",
    allow_n_sequences=True,
    allow_n_inputs=True,
    allow_timespans=True,
):

    noteacher_msg = f"Nodes can not be used as {io_type}" + " for {}."
    notonline_msg = "{} is not trained online."

    x_new = None
    # Caller is a Model
    if receiver_nodes is not None:
        if not is_mapping(x):
            x_new = {n.name: x for n in receiver_nodes}
        else:
            x_new = x.copy()

        for node in receiver_nodes:
            if node.name not in x_new:
                # Maybe don't fit nodes a second time
                if io_type == "target" and node.fitted:
                    continue
                else:
                    raise ValueError(f"Missing {io_type} data for node {node.name}.")

            if (
                callable(x_new[node.name])
                and hasattr(x_new[node.name], "initialize")
                and hasattr(x_new[node.name], "is_initialized")
                and hasattr(x_new[node.name], "output_dim")
            ):
                if io_type == "target":
                    if node.is_trained_online:
                        register_teacher(
                            node,
                            x_new.pop(node.name),
                            expected_dim=node.output_dim,
                        )
                    else:
                        raise TypeError(
                            (noteacher_msg + notonline_msg).format(node.name, node.name)
                        )
                else:
                    raise TypeError(noteacher_msg.format(node.name))
            else:
                if io_type == "target":
                    dim = node.output_dim
                else:
                    dim = node.input_dim

                x_new[node.name] = check_n_sequences(
                    x_new[node.name],
                    expected_dim=dim,
                    caller=node,
                    allow_n_sequences=allow_n_sequences,
                    allow_n_inputs=allow_n_inputs,
                    allow_timespans=allow_timespans,
                )
    # Caller is a Node
    else:
        if (
            callable(x)
            and hasattr(x, "initialize")
            and hasattr(x, "is_initialized")
            and hasattr(x, "output_dim")
        ):
            if io_type == "target":
                if caller.is_trained_online:
                    register_teacher(
                        caller,
                        x,
                        expected_dim=expected_dim,
                    )
                else:
                    raise TypeError(
                        (noteacher_msg + notonline_msg).format(caller.name, caller.name)
                    )
            else:
                raise TypeError(noteacher_msg.format(caller.name))
        else:
            x_new = check_n_sequences(
                x,
                expected_dim=expected_dim,
                caller=caller,
                allow_n_sequences=allow_n_sequences,
                allow_n_inputs=allow_n_inputs,
                allow_timespans=allow_timespans,
            )

    # All x are teacher nodes, no data to return
    if is_mapping(x_new) and io_type == "target" and len(x_new) == 0:
        return None

    return x_new


def register_teacher(caller, teacher, expected_dim=None):

    target_dim = None
    if teacher.is_initialized:
        target_dim = teacher.output_dim

    if (
        expected_dim is not None
        and target_dim is not None
        and expected_dim != target_dim
    ):
        raise ValueError()

    caller._teacher = DistantFeedback(
        sender=teacher, receiver=caller, callback_type="teacher"
    )


def check_xy(
    caller,
    x,
    y=None,
    input_dim=None,
    output_dim=None,
    allow_n_sequences=True,
    allow_n_inputs=True,
    allow_timespans=True,
):
    """Prepare one step of input and target data for a Node or a Model.

    Preparation may include:
        - reshaping data to ([inputs], [sequences], timesteps, features);
        - converting non-array objects to array objects;
        - checking if n_features is equal to node input or output dimension.

    This works on numerical data and teacher nodes.

    Parameters
    ----------
    caller: Node or Model
        Node or Model requesting inputs/targets preparation.
    x : array-like of shape ([inputs], [sequences], timesteps, features)
        Input array or sequence of input arrays containing a single timestep of
        data.
    y : array-like of shape ([sequences], timesteps, features) or Node, optional
        Target array containing a single timestep of data, or teacher Node or
        Model
        yielding target values.
    input_dim, output_dim : int or tuple of ints, optional
        Expected input and target dimensions, if available.

    Returns
    -------
    array-like of shape ([inputs], 1, n), array-like of shape (1, n) or Node
        Processed input and target vectors.
    """

    if input_dim is None and hasattr(caller, "input_dim"):
        input_dim = caller.input_dim

    # caller is a Model
    if hasattr(caller, "input_nodes"):
        input_nodes = caller.input_nodes
    # caller is a Node
    else:
        input_nodes = None

    x_new = _check_node_io(
        x,
        receiver_nodes=input_nodes,
        expected_dim=input_dim,
        caller=caller,
        io_type="input",
        allow_n_sequences=allow_n_sequences,
        allow_n_inputs=allow_n_inputs,
        allow_timespans=allow_timespans,
    )

    y_new = y
    if y is not None:
        # caller is a Model
        if hasattr(caller, "trainable_nodes"):
            output_dim = None
            trainable_nodes = caller.trainable_nodes

        # caller is a Node
        else:
            trainable_nodes = None
            if output_dim is None and hasattr(caller, "output_dim"):
                output_dim = caller.output_dim

        y_new = _check_node_io(
            y,
            receiver_nodes=trainable_nodes,
            expected_dim=output_dim,
            caller=caller,
            io_type="target",
            allow_n_sequences=allow_n_sequences,
            allow_timespans=allow_timespans,
            allow_n_inputs=False,
        )

    return x_new, y_new


class DistantFeedback:
    def __init__(self, sender, receiver, callback_type="feedback"):
        self._sender = sender
        self._receiver = receiver
        self._callback_type = callback_type

        # used to store a reduced version of the feedback if needed
        # when feedback is a Model (inputs of the feedback Model are suppressed
        # in the reduced version, as we do not need then to re-run them
        # because we assume they have already run during the forward call)
        self._reduced_sender = None

        self._clamped = False
        self._clamped_value = None

    def __call__(self):
        if not self.is_initialized:
            self.initialize()
        return self.call_distant_node()

    @property
    def is_initialized(self):
        return self._sender.is_initialized

    @property
    def output_dim(self):
        return self._sender.output_dim

    @property
    def name(self):
        return self._sender.name

    def call_distant_node(self):
        """Call a distant Model for feedback or teaching
        (no need to run the input nodes again)"""
        if self._clamped:
            self._clamped = False
            return self._clamped_value

        if self._reduced_sender is not None:
            if len(np.unique([n._fb_flag for n in self._sender.nodes])) > 1:
                input_data = _distant_model_inputs(self._sender)

                if hasattr(self._reduced_sender, "nodes"):
                    return self._reduced_sender.call(input_data)
                else:
                    reduced_name = self._reduced_sender.name
                    return self._reduced_sender.call(input_data[reduced_name])
            else:
                fb_outputs = [n.state() for n in self._sender.output_nodes]
                if len(fb_outputs) > 1:
                    return fb_outputs
                else:
                    return fb_outputs[0]
        else:
            return self._sender.state_proxy()

    def initialize(self):
        """Initialize a distant Model or Node (used as feedback sender or teacher)."""
        msg = f"Impossible to get {self._callback_type} "
        msg += "from {} for {}: {} is not initialized or has no input/output_dim"

        reduced_model = None
        if hasattr(self._sender, "input_nodes"):
            for n in self._sender.input_nodes:
                if not n.is_initialized:
                    try:
                        n.initialize()
                    except RuntimeError:
                        raise RuntimeError(
                            msg.format(
                                self._sender.name,
                                self._receiver.name,
                                self._sender.name,
                            )
                        )

            input_data = _distant_model_inputs(self._sender)
            reduced_model = _remove_input_for_feedback(self._sender)

            if not reduced_model.is_initialized:
                if hasattr(reduced_model, "nodes"):
                    reduced_model.initialize(x=input_data)
                else:
                    reduced_name = reduced_model.name
                    reduced_model.initialize(x=input_data[reduced_name])
                self._sender._is_initialized = True
        else:
            try:
                self._sender.initialize()
            except RuntimeError:  # raise more specific error
                raise RuntimeError(
                    msg.format(
                        self._sender.name, self._receiver.name, self._sender.name
                    )
                )

        self._reduced_sender = reduced_model

    def zero_feedback(self):
        """A null feedback vector. Returns None if the Node receives
        no feedback."""
        if hasattr(self._sender, "output_nodes"):
            zeros = []
            for output in self._sender.output_nodes:
                zeros.append(output.zero_state())
            if len(zeros) == 1:
                return zeros[0]
            else:
                return zeros
        else:
            return self._sender.zero_state()

    def clamp(self, value):
        self._clamped_value = check_n_sequences(
            value,
            expected_dim=self._sender.output_dim,
            caller=self._sender,
            allow_n_sequences=False,
        )
        self._clamped = True


def call(node, x, from_state=None, stateful=True, reset=False):
    """One-step call, without input check."""
    with node.with_state(from_state, stateful=stateful, reset=reset):
        state = node._forward(node, x)
        node._state = state.astype(node.dtype)
        node._flag_feedback()

    return state


def train(
    node,
    X,
    Y=None,
    call_node=True,
    force_teachers=True,
    learn_every=1,
    from_state=None,
    stateful=True,
    reset=False,
):

    seq_len = X.shape[0]
    seq = (
        progress(range(seq_len), f"Training {node.name}")
        if seq_len > 1
        else range(seq_len)
    )

    with node.with_state(from_state, stateful=stateful, reset=reset):
        states = np.zeros((seq_len, node.output_dim))
        for i in seq:
            x = np.atleast_2d(X[i, :])

            y = None
            if node._teacher is not None:
                y = node._teacher()
            elif Y is not None:
                y = np.atleast_2d(Y[i, :])

            if call_node:
                s = call(node, x)
            else:
                s = node.state()

            if force_teachers:
                node.set_state_proxy(y)

            if i % learn_every == 0 or seq_len == 1:
                node._train(node, x=x, y=y)

            states[i, :] = s

    return states


class _Node(ABC):
    """Node base class for type checking and interface inheritance."""

    _factory_id = -1
    _registry = list()
    _name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._factory_id = -1
        cls._registry = list()

    def __repr__(self):
        klas = type(self).__name__
        hypers = [(str(k), str(v)) for k, v in self._hypers.items()]
        all_params = ["=".join((k, v)) for k, v in hypers]
        all_params += [f"in={self.input_dim}", f"out={self.output_dim}"]
        return f"'{self.name}': {klas}(" + ", ".join(all_params) + ")"

    def __setstate__(self, state):
        curr_name = state.get("name")
        if curr_name in type(self)._registry:
            new_name = curr_name + "-(copy)"
            state["name"] = new_name
        self.__dict__ = state

    def __del__(self):
        try:
            type(self)._registry.remove(self._name)
        except (ValueError, AttributeError):
            pass

    def __getattr__(self, item):
        if item in ["_params", "_hypers"]:
            raise AttributeError()
        if item in self._params:
            return self._params.get(item)
        elif item in self._hypers:
            return self._hypers.get(item)
        else:
            raise AttributeError(f"{self.name} has no attribute '{str(item)}'")

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.call(*args, **kwargs)

    def __rshift__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
        from .ops import link

        return link(self, other)

    def __rrshift__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
        from .ops import link

        return link(other, self)

    def __and__(self, other: Union["_Node", Sequence["_Node"]]) -> "Model":
        from .ops import merge

        return merge(self, other)

    def _get_name(self, name=None):
        if name is None:
            type(self)._factory_id += 1
            _id = self._factory_id
            name = f"{type(self).__name__}-{_id}"

        if name in type(self)._registry:
            raise NameError(
                f"Name '{name}' is already taken "
                f"by another node. Node names should "
                f"be unique."
            )

        type(self)._registry.append(name)
        return name

    @property
    def name(self) -> str:
        """Name of the Node or Model."""
        return self._name

    @name.setter
    def name(self, value):
        type(self)._registry.remove(self.name)
        self._name = self._get_name(value)

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters of the Node or Model."""
        return self._params

    @property
    def hypers(self) -> Dict[str, Any]:
        """Hyperparameters of the Node or Model."""
        return self._hypers

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    @abstractmethod
    def input_dim(self) -> Shape:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_dim(self) -> Shape:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_trained_offline(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_trained_online(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_trainable(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def fitted(self) -> bool:
        raise NotImplementedError()

    @is_trainable.setter
    @abstractmethod
    def is_trainable(self, value: bool):
        raise NotImplementedError()

    def get_param(self, name: str) -> Any:
        if name in self._params:
            return self._params.get(name)
        elif name in self._hypers:
            return self._hypers.get(name)
        else:
            raise NameError(f"No parameter named '{name}' found in node {self}")

    @abstractmethod
    def copy(
        self, name: str = None, copy_feedback: bool = False, shallow: bool = False
    ) -> "_Node":
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, x: MappedData = None, y: MappedData = None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, to_state: np.ndarray = None) -> "_Node":
        raise NotImplementedError()

    @contextmanager
    @abstractmethod
    def with_state(self, state=None, stateful=False, reset=False) -> Iterator["_Node"]:
        raise NotImplementedError()

    @contextmanager
    @abstractmethod
    def with_feedback(
        self, feedback=None, stateful=False, reset=False
    ) -> Iterator["_Node"]:
        raise NotImplementedError()
