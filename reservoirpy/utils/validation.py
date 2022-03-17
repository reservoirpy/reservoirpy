# Author: Nathan Trouvain at 21/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numbers
from typing import Any, Mapping

import numpy as np
from scipy.sparse import issparse

#
# def check_node_io(
#     node, x, expected_dim, allow_timespans=False, allow_list=True, io_type="input"
# ):
#
#     if io_type == "output":
#         msg = (
#             "Impossible to fit node {}: node expected output "
#             "dimension is (1, {}) and teacher vector dimension is {}."
#         )
#     else:
#         msg = (
#             "Impossible to call node {}: node input dimension is (1, {}) "
#             "and input dimension is {}."
#         )
#
#     if isinstance(x, np.ndarray):
#         x = check_vector(
#             x, allow_reshape=True, allow_timespans=allow_timespans, caller=node
#         )
#
#         if node.is_initialized:
#             if x.shape[1] != expected_dim:
#                 raise ValueError(msg.format(node.name, expected_dim, x.shape))
#
#     elif isinstance(x, list):
#         if allow_list:
#             for i in range(len(x)):
#                 x[i] = check_vector(
#                     x[i],
#                     allow_reshape=True,
#                     allow_timespans=allow_timespans,
#                     caller=node,
#                 )
#         else:
#             raise TypeError(
#                 f"Data type not understood. Expected a single "
#                 f"Numpy array but received list in {node}: {x}"
#             )
#     # if X is a teacher Node
#     elif (
#         io_type == "output"
#         and hasattr(x, "is_initialized")
#         and hasattr(x, "output_dim")
#     ):
#         if x.is_initialized and expected_dim is not None:
#             if x.output_dim != expected_dim:
#                 raise ValueError(msg.format(node.name, expected_dim, x.output_dim))
#     # if X is a teacher Node
#     elif (
#         io_type == "input" and hasattr(x, "is_initialized") and hasattr(x, "output_dim")
#     ):
#         raise ValueError(
#             "Impossible to use a Node as input X. Nodes can only "
#             "be used to generate targets Y."
#         )
#     return x

#
# def check_node_state(node, s):
#     s = check_vector(s, allow_timespans=False, caller=node)
#
#     if not node.is_initialized:
#         raise RuntimeError(
#             f"Impossible to set state of node {node.name}: node "
#             f"is not initialized yet."
#         )
#
#     if s.shape[1] != node.output_dim:
#         raise ValueError(
#             f"Impossible to set state of node {node.name}: "
#             f"dimension mismatch between state vector ("
#             f"{s.shape[1]}) "
#             f"and node output dim ({node.output_dim})."
#         )
#     return s


def is_sequence_set(seq: Any) -> bool:
    return isinstance(seq, list) or (isinstance(seq, np.ndarray) and seq.ndim > 2)


def is_array(obj: Any) -> bool:
    return obj is not None and isinstance(obj, np.ndarray) or issparse(obj)


def is_mapping(obj):
    return isinstance(obj, Mapping) or (
        (hasattr(obj, "items") and hasattr(obj, "get"))
        or (
            not (isinstance(obj, list) or isinstance(obj, tuple))
            and hasattr(obj, "__getitem__")
            and not hasattr(obj, "__array__")
        )
    )


def add_bias(X):
    if isinstance(X, np.ndarray):
        X = np.atleast_2d(X)
        return np.hstack([np.ones((X.shape[0], 1)), X])
    elif isinstance(X, list):
        new_X = []
        for x in X:
            x = np.atleast_2d(x)
            new_X.append(np.hstack([np.ones((x.shape[0], 1)), x]))
        return new_X


def check_vector(array, allow_reshape=True, allow_timespans=True, caller=None):
    msg = "."
    if caller is not None:
        if hasattr(caller, "name"):
            msg = f" in {caller.name}."
        else:
            msg = f"in {caller}."

    if not isinstance(array, np.ndarray):
        # maybe a single number, make it an array
        if isinstance(array, numbers.Number):
            array = np.asarray(array)
        else:
            msg = (
                f"Data type '{type(array)}' not understood. All vectors "
                f"should be Numpy arrays" + msg
            )
            raise TypeError(msg)

    if not (np.issubdtype(array.dtype, np.number)):
        msg = f"Impossible to operate on non-numerical data, in array: {array}" + msg
        raise TypeError(msg)

    if allow_reshape:
        array = np.atleast_2d(array)

    if not allow_timespans:
        if array.shape[0] > 1:
            msg = (
                f"Impossible to operate on multiple timesteps. Data should"
                f" have shape (1, n) but is {array.shape}" + msg
            )
            raise ValueError(msg)

    # TODO: choose axis to expand and/or np.atleast_2d

    return array
