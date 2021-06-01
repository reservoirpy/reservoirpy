# Author: Nathan Trouvain at 22/04/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import numpy as np


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


def _format_data(X, y=None):

    newX, newy = X, y

    if isinstance(X, np.ndarray):
        _check_array(X)
        if X.ndim > 3:
            raise ValueError(f"Found array with {X.ndim} dimensions. "
                             "Input data must be of shape "
                             "(n_samples, n_timesteps, n_features) "
                             "or (n_timesteps, n_features).")
        if X.ndim <= 2:
            newX = [X]

    elif isinstance(X, Sequence):
        for i, x in enumerate(X):
            if not(isinstance(x, np.ndarray)):
                raise ValueError(f"Found object of type {type(x)} at index {i} "
                                 "in data. All data samples must be Numpy arrays.")
            else:
                _check_array(x)

    if y is not None:
        if isinstance(y, np.ndarray):
            if y.ndim > 3:
                raise ValueError(f"Found array with {X.ndim} dimensions. "
                                 "Input data must be of shape "
                                 "(n_samples, n_timesteps, n_features) "
                                 "or (n_timesteps, n_features).")
            if y.ndim <= 2:
                newy = [y]

        elif isinstance(y, Sequence):
            for i, yy in enumerate(y):
                if not(isinstance(yy, np.ndarray)):
                    raise ValueError(f"Found object of type {type(yy)} at index {i} "
                                     "in targets. All targets must be Numpy arrays.")
                else:
                    _check_array(yy)

    return newX, newy


def _add_bias(vector, bias=1.0, pos="first"):
    if pos == "first":
        return np.hstack((np.ones((vector.shape[0], 1)) * bias, vector))
    if pos == "last":
        return np.hstack((vector, bias))


def _check_array(array: np.ndarray):
    """ Check if the given array or list contains the given value. """
    if issparse(array):
        _check_array(array.data)
    else:
        if not np.isfinite(np.asanyarray(array)).any():
            num = np.sum(~np.isfinite(np.asarray(array)))
            raise ValueError(f"Found {num} NaN or inf value in array : {array}.")

        if isinstance(array, list):
            if np.count_nonzero(np.asarray(array) == None) != 0:
                raise ValueError(f"Found None in array : {array}")


def _check_vector(vector):

    if not isinstance(vector, np.ndarray):
        if issparse(vector):
            checked_vect = vector
        else:
            try:
                checked_vect = np.asanyarray(vector)
            except Exception:
                raise ValueError("Trying to use a data structure that is "
                                 "not an array. Consider converting "
                                 f"your data to numpy array format : {vector}")
    else:
        checked_vect = vector

    if issparse(checked_vect):
        pass
    elif not np.issubdtype(checked_vect.dtype, np.number):
        try:
            checked_vect = checked_vect.astype(float)
        except Exception:
            raise ValueError("Trying to use non numeric data (of dtype "
                             f"{vector.dtype}) : {vector}")

    if checked_vect.ndim == 1:
        checked_vect = checked_vect.reshape(1, -1)
    elif checked_vect.ndim > 2:
        raise ValueError("Trying to use data that is more than "
                         f"2-dimensional ({vector.shape}): {vector}")
    return checked_vect


def _pack(arg):
    if isinstance(arg, list):
        if len(arg) >= 2:
            return arg
        else:
            return arg[0]
    else:
        return arg
