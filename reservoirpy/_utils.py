import os
import time
import json
from typing import Sequence, Union, Any

import dill
import numpy as np

from scipy import sparse

import reservoirpy


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
            "fbfunc": fbfunc,
            "noise_in": esn.noise_in,
            "noise_out": esn.noise_out,
            "noise_rc": esn.noise_rc,
            "seed": esn.seed,
            "sklearn_model": esn.sklearn_model
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

    if os.path.splitext(model_attr["W"])[1] == ".npy":
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
