import os
import time
import json

from typing import Sequence, Union, Any

import dill
import numpy as np

from scipy import sparse
from scipy.sparse import issparse
from numpy.random import Generator, RandomState, MT19937, default_rng

import reservoirpy

from ._types import RandomSeed


def _random_generator(seed: RandomSeed) -> Generator:
    """

    Returns
    -------
    Generator:
        A `numpy.random.Generator` instance.
    """
    if isinstance(seed, Generator):
        return seed
    # provided to support legacy RandomState generator
    # of Numpy. It is not the best thing to do however
    # and recommend the user to keep using integer seeds
    # and proper Numpŷ Generator API.
    if isinstance(seed, RandomState):
        mt19937 = MT19937()
        mt19937.state = seed.get_state()
        return Generator(mt19937)
    else:
        return default_rng(seed)


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


def _pack(arg):
    if isinstance(arg, list):
        if len(arg) >= 2:
            return arg
        else:
            return arg[0]
    else:
        return arg


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


def _add_bias(vector, bias=1.0, pos="first"):
    if pos == "first":
        return np.hstack((np.ones((vector.shape[0], 1)) * bias, vector))
    if pos == "last":
        return np.hstack((vector, bias))


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


def _save(esn, directory: str):
    """ Base utilitary for saving an ESN model, based on the ESN class.

    Arguments:
        esn {ESN} -- ESN model to save.
        directory {str or Path} -- Directory to store the model.
    """
    # create new directory
    savedir = os.path.join(directory)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    else:
        raise OSError(f"Directory '{savedir}' already exists.")

    current_time = time.time()

    # store matrices
    Win_path = f"esn-Win-{current_time}.npy"
    Wout_path = None
    Wfb_path = None

    if sparse.issparse(esn.W):
        W_path = f"esn-W-{current_time}.npz"
        sparse.save_npz(os.path.join(savedir, W_path), esn.W)
    else:
        W_path = f"esn-W-{current_time}.npy"
        np.save(os.path.join(savedir, W_path), esn.W)
    np.save(os.path.join(savedir, Win_path), esn.Win)

    dim_out = None
    if esn.Wout is not None:
        Wout_path = f"esn-Wout-{current_time}.npy"
        np.save(os.path.join(savedir, Wout_path), esn.Wout)
        dim_out = esn.dim_out

    fbfunc = None
    fbfunc_info = None
    if esn.Wfb is not None:
        Wfb_path = f"esn-Wfb-{current_time}.npy"
        np.save(os.path.join(savedir, Wfb_path), esn.Wfb)
        # fbfunc is serialized and stored
        fbfunc_info = {
            "cls": str(esn.fbfunc.__class__),
            "name": esn.fbfunc.__name__
        }
        fbfunc = f"fbfunc_save-{current_time}"
        dim_out = esn.dim_out
        with open(os.path.join(savedir, fbfunc), "wb+") as f:
            dill.dump(esn.fbfunc, f)

    reg_model = {"type": "pinv"}
    if esn.ridge is not None:
        reg_model = {"type": "ridge", "coef": esn.ridge}
    elif esn.sklearn_model is not None:
        reg_model = {"type": "sklearn", "path": f"sklearn_func_save-{current_time}"}
        # reg_model is serialized and stored
        with open(os.path.join(savedir, reg_model), "wb+") as f:
            dill.dump(esn.sklearn_model, f)

    # a copy of the ESN class is also serialized.
    # allow to load an ESN without necesseraly using the same version of Reservoirpy.
    cls_path = f"cls_bin-{current_time}"
    with open(os.path.join(savedir, cls_path), "wb+") as f:
        dill.dump(esn.__class__, f)

    attr = {
        "cls": esn.__class__.__name__,
        "cls_bin": cls_path,
        "version": reservoirpy.__version__,
        "serial": current_time,
        "attr": {
            "W": W_path,
            "Win": Win_path,
            "Wfb": Wfb_path,
            "Wout": Wout_path,
            "N": esn.N,
            "lr": esn.lr,
            "in_bias": esn.in_bias,
            "dim_inp": esn.dim_inp,
            "dim_out": dim_out,
            "typefloat": esn.typefloat.__name__,
            "reg_model": reg_model,
            "fbfunc": fbfunc
        },
        "misc": {
            "fbfunc_info": fbfunc_info
        }
    }

    # save a summary file
    with open(os.path.join(savedir, "esn.json"), "w+") as f:
        json.dump(attr, f)


def _new_from_save(base_cls, restored_attr):

    obj = object.__new__(base_cls)
    for name, attr in restored_attr.items():
        obj.__setattr__(name, attr)
    obj.reg_model = obj._get_regression_model(obj.ridge, obj.reg_model)

    return obj


def load(directory: str):
    """Load an ESN model.

    Parameters
    ----------
        directory : str or Path
            Saved model directory.

    Returns
    -------
        :py:class:`reservoirpy.ESN`
            Loaded ESN.
    """
    with open(os.path.join(directory, "esn.json"), "r") as f:
        attr = json.load(f)

    model_attr = attr["attr"]

    if os.path.splitext(model_attr["W"])[1] == "npy":
        model_attr["W"] = np.load(os.path.join(directory, model_attr["W"]))
    else:
        model_attr["W"] = sparse.load_npz(os.path.join(directory, model_attr["W"]))

    model_attr["Win"] = np.load(os.path.join(directory, model_attr["Win"]))

    if model_attr["Wout"] is not None:
        model_attr["Wout"] = np.load(os.path.join(directory, model_attr["Wout"]))
    if model_attr["Wfb"] is not None:
        model_attr["Wfb"] = np.load(os.path.join(directory, model_attr["Wfb"]))
        with open(os.path.join(directory, model_attr["fbfunc"]), "rb") as f:
            model_attr["fbfunc"] = dill.load(f)

    if model_attr["reg_model"]["type"] == "ridge":
        model_attr["ridge"] = model_attr["reg_model"]["coef"]
        model_attr["reg_model"] = None
    elif model_attr["reg_model"]["type"] == "sklearn":
        with open(os.path.join(directory, model_attr["reg_model"]["path"]), "rb") as f:
            model_attr["reg_model"] = dill.load(f)
        model_attr["ridge"] = None
    else:
        model_attr["reg_model"] = None
        model_attr["ridge"] = None

    model_attr["typefloat"] = np.float64

    with open(os.path.join(directory, attr["cls_bin"]), 'rb') as f:
        base_cls = dill.load(f)

    model = _new_from_save(base_cls, model_attr)

    return model
