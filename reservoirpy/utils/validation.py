# Author: Nathan Trouvain at 21/06/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
from typing import Union, Sequence, Any

import numpy as np


def add_bias(X):
    if isinstance(X, np.ndarray):
        if X.ndim < 2:
            X = X.reshape(1, -1)
        return np.hstack([np.ones((X.shape[0], 1)), X])
    elif isinstance(X, list):
        new_X = []
        for x in X:
            new_X.append(np.hstack([np.ones((x.shape[0], 1)), x]))
        return new_X


def _check_values(array_or_list: Union[Sequence, np.ndarray], value: Any):
    """ Check if the given array or list contains the given value. """
    if value == np.nan:
        assert np.isnan(array_or_list).any() == False, \
               f"{array_or_list} should not contain NaN values."
    if value is None:
        if type(array_or_list) is list:
            assert np.count_nonzero(array_or_list == None) == 0, \
                   f"{array_or_list} should not contain None values."
        elif type(array_or_list) is np.array:
            # None is transformed to np.nan when it is in an array
            assert np.isnan(array_or_list).any() == False, \
                   f"{array_or_list} should not contain NaN values."


def check_vector(array, allow_reshape=True):
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Data type '{type(array)}' not understood. All sequences of data "
                        f"should be Numpy arrays, or lists of Numpy arrays.")

    if not(np.issubdtype(array.dtype, np.number)):
        raise TypeError(f"Impossible to operate on non-numerical data, in array: {array}")

    if allow_reshape:
        if array.ndim < 2:
            array = array.reshape(1, -1)

    return array


def check_input_lists(X, dim_in, Y=None, dim_out=None):

    if isinstance(X, np.ndarray):
        X = [X]

    if Y is not None:
        if isinstance(Y, np.ndarray):
            Y = [Y]
        if not(len(X) == len(Y)):
            raise ValueError(f"Inconsistent number of inputs and targets: "
                             f"found {len(X)} input sequences, but {len(Y)} "
                             f"target sequences.")

    for i in range(len(X)):
        x = check_vector(X[i], allow_reshape=False)

        if x.ndim != 2:
            raise ValueError(f"Input {i} has shape {x.shape} but should "
                             f"be 2-dimensional, with first axis representing "
                             f"time and second axis representing features.")

        if x.shape[1] != dim_in:
            raise ValueError(f"Input {i} has {x.shape[1]} features but ESN expects "
                             f"{dim_in} features as input.")

        if Y is not None:
            y = check_vector(Y[i], allow_reshape=False)
            if y.ndim != 2:
                raise ValueError(f"Target {i} has shape {y.shape} but should "
                                 f"be 2-dimensional, with first axis representing "
                                 f"time and second axis representing features.")

            if x.shape[0] != y.shape[0]:
                raise ValueError(f"Inconsistent inputs and targets lengths: "
                                 f"input {i} has length {x.shape[0]} but "
                                 f"corresponding target {i} has length "
                                 f"{y.shape[0]}.")

            if dim_out is not None:
                if y.shape[1] != dim_out:
                    raise ValueError(f"Target {i} has {y.shape[1]} features but ESN expects "
                                     f"{dim_out} features as feedback.")

    return X, Y

