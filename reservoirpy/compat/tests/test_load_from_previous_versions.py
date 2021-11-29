# Author: Nathan Trouvain at 23/11/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pytest
import os
import shutil
from contextlib import contextmanager

import numpy as np
from scipy import sparse

from .. import load_compat
from .. import ESN
from ... import activationsfunc as F


@contextmanager
def create_old_model(sparse_W=True, input_bias=True, feedback=False,
                     train=False, fbfunc=F.identity):

    parent_dir = os.path.dirname(os.path.realpath(__file__))

    d = parent_dir + "dummy_model"

    if os.path.exists(d):
        shutil.rmtree(d)

    in_dim = 10 + int(input_bias)
    wfb = None

    if sparse_W:
        w = sparse.random(100, 100, format="csr", dtype=np.float64)
    else:
        w = np.random.normal(0, 0.1, size=(100, 100))

    win = np.random.normal(0, 0.1, size=(100, in_dim))
    if feedback:
        wfb = np.random.normal(0, 0.1, size=(100, 10))

    esn = ESN(lr=0.3, W=w, Win=win, input_bias=input_bias, Wfb=wfb,
              ridge=1e-8, fbfunc=fbfunc)

    if train:
        X, Y = [np.ones((100, 10))], [np.ones((100, 10))]
        esn.train(X, Y)

    esn.save(d)

    yield d, esn

    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.mark.parametrize("sparse,bias,feedback,train,fbfunc",
                         ((True, False, False, False, F.identity),
                          (True, False, False, True, F.identity),
                          (True, True, False, False, F.identity),
                          (True, False, True, True, F.sigmoid),
                          (True, True, True, False, F.tanh),
                          (False, True, False, False, F.identity),
                          (False, False, True, True, F.softmax)))
def test_load_files_from_v2(sparse, bias, feedback, fbfunc, train):
    with create_old_model(sparse_W=sparse, input_bias=bias,
                          feedback=feedback, train=train, fbfunc=fbfunc) as m:

        dirname, esn = m[0], m[1]

        esn2 = load_compat(dirname)

        X = np.ones((100, 10))

        if train is False and feedback is False:
            with pytest.raises(RuntimeError):
                esn2.run(X)

            esn2 = load_compat(dirname)
            esn2.fit(X, X)
            res = esn2.run(X)
            assert res.shape == (100, 10)
        else:
            res = esn2.run(X)
            assert res.shape == (100, 10)
