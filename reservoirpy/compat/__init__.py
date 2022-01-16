"""
==================================================
ReservoirPy v0.2 tools (:mod:`reservoirpy.compat`)
==================================================

ReservoirPy v0.2 tools, kept for compatibility with older projects.

Warning
-------

    ReservoirPy v0.2 tools are deprecated. No removal date has been set,
    however, we encourage users to switch to newer versions (v0.3) of the
    library. The function :py:func:`load_compat` can help you load v0.2 models
    using the new Node API (see :ref:`node`), introduced in version v0.3.

Load and backport
=================

.. autosummary::
    :toctree: generated/

    load_compat - Load v0.2 models into v0.3
    load - Load old v0.2 models

ESN and ESNOnline
=================

.. autosummary::
    :toctree: generated/

    ESN - ESN model with parallelized training
    ESNOnline - ESN with FORCE learning

Regression models
=================

.. autosummary::
    :toctree: generated/

    regression_models.RidgeRegression - Tikhonov regression solver
"""
import json
import pathlib
import dill
import re

from typing import Union

import numpy as np

from scipy import sparse

from . import regression_models

from ._esn import ESN
from ._esn_online import ESNOnline

from .utils.save import load
from ..nodes import ESN as ESN_v3
from ..nodes import Reservoir, Ridge
from ..activationsfunc import identity


def _load_files_from_v2(dirpath):
    dirpath = pathlib.Path(dirpath)
    matrices = dict()
    config = dict()
    for filename in dirpath.iterdir():
        ext = filename.suffix
        # all rpy <= 0.2.4 versions have file names starting with esn
        is_esn_obj = "esn" in filename.name[:3]
        if is_esn_obj:
            if ext in (".npy", ".npz"):
                matrix_name = ("Win", "W", "Wfb", "Wout", "_W", "_Win",
                               "_Wfb", "_Wout")
                match = re.findall("_?W.*?(?=-)", filename.name)
                for name in matrix_name:
                    if name in match:
                        n = name
                        if name.startswith("_"):
                            n = name[1:]
                        matrices[n] = _load_matrix_v2(filename)
            elif ext == ".json":
                with filename.open(mode="r+") as fp:
                    config = json.load(fp)

    fns = dict()
    for attr, value in config.items():
        if attr == "fbfunc" and value is not None:
            filename = pathlib.Path(value)
            if filename.exists():
                fns["fbfunc"] = dill.load(filename)

    return matrices, fns, config


def _load_matrix_v2(filename):
    ext = filename.suffix
    mat = None
    if ext == ".npy":
        mat = np.load(str(filename))
    elif ext == ".npz":  # maybe a scipy sparse array
        try:
            mat = sparse.load_npz(str(filename))
        except Exception as e:
            mat = np.load(str(filename))
            keys = list(mat.keys())
            sparse_keys = ('indices', 'indptr',
                           'format', 'shape', 'data')
            if any([k in sparse_keys for k in keys]):
                raise e
            elif len(keys) == 1:  # Only one array per file
                mat = mat[keys[0]]
            else:
                raise TypeError("Unknown array format "
                                f"in file {filename}.")
    return mat


def load_compat(directory: Union[str, pathlib.Path]) -> ESN_v3:
    """Load a ReservoirPy v0.2.4 and lower ESN model as a
    ReservoirPy v0.3 model.

    .. warning::
        Models and Nodes should now
        be saved using Python serialization utilities
        `pickle`.

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
        reservoirpy.nodes.ESN
            A ReservoirPy v0.3 ESN instance.
    """
    dirpath = pathlib.Path(directory)
    if not dirpath.exists():
        raise NotADirectoryError(f"'{directory}' not found.")

    matrices, fns, config = _load_files_from_v2(dirpath)

    attr = config.get("attr", config)

    version = config.get("version")

    msg = "Impossible to load ESN from version {} of " \
          "reservoirpy: unknown model {}"
    ridge = 0.0
    if attr.get("sklearn_model") is not None:
        raise TypeError(msg.format(version, attr["sklearn_model"]))
    elif attr.get("_ridge") is not None:
        ridge = attr["_ridge"]

    if attr.get("reg_model") is not None:
        reg_model = attr["reg_model"]
        if reg_model["type"] not in ("ridge", "pinv"):
            raise TypeError(msg.format(version, attr["type"]))
        elif reg_model["type"] == "ridge":
            ridge = reg_model.get("coef", 0.0)

    feedback = False
    if matrices.get("Wfb") is not None:
        feedback = True

    output_dim = attr.get("dim_out", attr.get("_dim_out"))

    reservoir = Reservoir(units=attr.get("N", attr.get("_N")),
                          lr=attr["lr"],
                          input_bias=attr.get("in_bias",
                                              attr.get("_input_bias")),
                          W=matrices["W"],
                          Win=matrices["Win"],
                          Wfb=matrices.get("Wfb"),
                          fb_activation=fns.get("fbfunc", identity),
                          noise_in=attr.get("noise_in", 0.0),
                          noise_rc=attr.get("noise_rc", 0.0),
                          noise_fb=attr.get("noise_out", 0.0),
                          noise_type="uniform",
                          seed=attr.get("seed"))

    readout = Ridge(output_dim=output_dim,
                    ridge=ridge, Wout=matrices.get("Wout"))

    model = ESN_v3(reservoir=reservoir, readout=readout, feedback=feedback)

    return model


__all__ = [
    "ESN", "ESNOnline", "load_compat", "regression_models", "load"
    ]
