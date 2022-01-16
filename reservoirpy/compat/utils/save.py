import os
import time
import json

import dill
import numpy as np

from scipy import sparse

from ..._version import __version__
from .. import regression_models


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

    sklearn_model = None
    if getattr(esn.model, "model", None) is not None:
        sklearn_model = f"sklearn_func_save-{current_time}"
        # scikit-learn model is serialized and stored
        # will require scikit-learn to be imported when loading ESN.
        with open(os.path.join(savedir, sklearn_model), "wb+") as f:
            dill.dump(esn.model.model, f)

    # a copy of the ESN class is also serialized.
    # allow to load an ESN without necesseraly using
    # the same version of Reservoirpy.
    cls_path = f"cls_bin-{current_time}"
    with open(os.path.join(savedir, cls_path), "wb+") as f:
        dill.dump(esn.__class__, f)

    attr = {
        "cls": esn.__class__.__name__,
        "cls_bin": cls_path,
        "version": __version__,
        "serial": current_time,
        "attr": {
            "_W": W_path,
            "_Win": Win_path,
            "_Wfb": Wfb_path,
            "_Wout": Wout_path,
            "_N": esn._N,
            "lr": esn.lr,
            "_input_bias": esn.input_bias,
            "_dim_in": esn._dim_in,
            "_dim_out": dim_out,
            "_ridge": esn.ridge,
            "typefloat": esn.typefloat.__name__,
            "sklearn_model": sklearn_model,
            "fbfunc": fbfunc,
            "noise_in": esn.noise_in,
            "noise_out": esn.noise_out,
            "noise_rc": esn.noise_rc,
            "seed": esn.seed,
            "model": esn.model.__class__.__name__
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
        try:
            obj.__setattr__(name, attr)
        except AttributeError as e:
            print(e)
            print(name, attr)

    obj.model = getattr(regression_models, obj.model)(obj._ridge, obj.sklearn_model)

    del obj.sklearn_model
    del obj._ridge

    return obj


def load(directory: str):
    """Load an ESN model in v0.2 format.

    Warning
    -------

        v0.2 models are deprecated. Consider using :py:func:`load_compat` to
        translate saved models from v0.2 to new Node API (see :ref:`node`)
        introduced in v0.3.

    Parameters
    ----------
        directory : str or Path
            Saved model directory.

    Returns
    -------
        :py:class:`compat.ESN`
            Loaded ESN.
    """
    with open(os.path.join(directory, "esn.json"), "r") as f:
        attr = json.load(f)

    model_attr = attr["attr"]

    if os.path.splitext(model_attr["_W"])[1] == ".npy":
        model_attr["_W"] = np.load(os.path.join(directory, model_attr["_W"]))
    else:
        model_attr["_W"] = sparse.load_npz(os.path.join(directory, model_attr["_W"]))

    model_attr["_Win"] = np.load(os.path.join(directory, model_attr["_Win"]))

    if model_attr["_Wout"] is not None:
        model_attr["_Wout"] = np.load(os.path.join(directory, model_attr["_Wout"]))
    if model_attr["_Wfb"] is not None:
        model_attr["_Wfb"] = np.load(os.path.join(directory, model_attr["_Wfb"]))
        with open(os.path.join(directory, model_attr["fbfunc"]), "rb") as f:
            model_attr["fbfunc"] = dill.load(f)

    elif model_attr["sklearn_model"] is not None:
        with open(os.path.join(directory, model_attr["sklearn_model"]), "rb") as f:
            model_attr["sklearn_model"] = dill.load(f)

    model_attr["typefloat"] = getattr(np, model_attr["typefloat"])

    with open(os.path.join(directory, attr["cls_bin"]), 'rb') as f:
        base_cls = dill.load(f)

    if model_attr.get("activation") is None:
        model_attr["activation"] = np.tanh

    model = _new_from_save(base_cls, model_attr)

    return model
