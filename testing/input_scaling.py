# import pdb
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import numpy as np
# import sys
# sys.path.insert(0, '../')
# from reservoirpy.datasets import japanese_vowels
# from reservoirpy import set_seed, verbosity
# from reservoirpy.observables import nrmse, rsquare
# from reservoirpy.nodes import Reservoir, RidgeRegression, Input
# from sklearn.metrics import accuracy_score
# from reservoirpy.mat_gen import fast_spectral_initialization, generate_input_weights

# import ipywidgets as widgets
# set_seed(42)
# verbosity(0)

# X_train, Y_train, X_test, Y_test = japanese_vowels()

# hyper = {
#     "N": 500,
#     "sr": 0.7,
#     "leak": 9e-2,
#     "iss": 1e-3,
#     "isd": 5e-3,
#     "isd2": 5e-3,
#     "ridge": 1e-3, #1e-8
#     "n_mfcc": 13,
#     "mfcc": True,
#     "d": True,
#     "dd": True,
#     "alpha":1.0,
#     "seed": 12345689
# }

# def get_weights(hyper):
#     rng = np.random.default_rng(hyper["seed"])
#     Wins = []
#     input_bias = True
#     import pdb;pdb.set_trace()
#     if hyper["mfcc"]:
#         win = generate_input_weights(hyper["N"], hyper["n_mfcc"], hyper["iss"], proba=0.2,
#                                      input_bias=input_bias, seed=rng)
#         input_bias = False
#         Wins.append(win)
#     if hyper["d"]:
#         wd = generate_input_weights(hyper["N"], hyper["n_mfcc"], hyper["isd"], proba=0.2,
#                                     input_bias=input_bias, seed=rng)
#         input_bias = False
#         Wins.append(wd)
#     if hyper["dd"]:
#         wdd = generate_input_weights(hyper["N"], hyper["n_mfcc"], hyper["isd2"], proba=0.2,
#                                      input_bias=input_bias, seed=rng)
#         input_bias = False
#         Wins.append(wdd)

#     Win = np.hstack(Wins)

#     W = fast_spectral_initialization(hyper["N"], sr=hyper["sr"], seed=rng, proba=0.2)
#     return Win, W

# source = Input()
# W, W_in = get_weights(hyper)
# reservoir = Reservoir(N=hyper["N"], sr=hyper["sr"], lr=hyper["lr"], W=W, Win=Win)
# readout = RidgeRegression(alpha=hyper['alpha'])
# model = source >> reservoir >> readout

# states_train = []
# for x in X_train:
#     states = reservoir.run(x, reset=True)
#     states_train.append(states[-1, np.newaxis])

# # the ridge regression class expects inputs and outputs as a numpy 2d matrix
# # thus we do a bit of pre-processing where we convert the states_train list 
# # and Y_train into a numpy array and remove any unecessary dimensions
# readout.fit(np.array(states_train).squeeze(), np.array(Y_train).squeeze())

# Y_test = np.array(Y_test).squeeze()
# Y_pred_class = np.argmax(y_pred, axis=1)
# Y_test_class = np.argmax(Y_test, axis=1)
# score = accuracy_score(Y_test_class, Y_pred_class)
# print("Accuracy: ", f"{score * 100:.3f} %")

import glob
import json
from pathlib import Path
import pprint
from typing import Sequence, Union, Tuple, Dict, Any

import numpy as np
from numpy.random import SeedSequence
from joblib import delayed, Parallel
from scipy import linalg
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics

import sys
sys.path.insert(0, '../')
from reservoirpy.mat_gen import fast_spectral_initialization, generate_input_weights
# from reservoirpy import ESN
# from canapy import Config
# from canapy.sequence import lcs, lev_sim, group


def esn_model(config, seed=None):

    sr = config["sr"]
    iss = config["iss"]
    isd = config["isd"]
    isd2 = config["isd2"]
    N = config["N"]
    leak = config["leak"]
    ridge = config["ridge"]
    n_mfcc = config["n_mfcc"]
    mfcc = config["mfcc"]
    d = config["d"]
    dd = config["dd"]

    rng = np.random.default_rng(seed)

    Wins = []
    input_bias = True
    if mfcc:
        win = generate_input_weights(N, n_mfcc, iss, proba=0.2,
                                     input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(win)
    if d:
        wd = generate_input_weights(N, n_mfcc, isd, proba=0.2,
                                    input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(wd)
    if dd:
        wdd = generate_input_weights(N, n_mfcc, isd2, proba=0.2,
                                     input_bias=input_bias, seed=rng)
        input_bias = False
        Wins.append(wdd)

    Win = np.hstack(Wins)

    W = fast_spectral_initialization(N, sr=sr, seed=rng, proba=0.2)

    return Win

if __name__ == "__main__":
	# config = lambda x: x
	CONF = {
    "N": 1000,
    "sr": 0.7,
    "leak": 9e-2,
    "iss": 1e-3,
    "isd": 5e-3,
    "isd2": 5e-3,
    "ridge": 1e-3, #1e-8
    "n_mfcc": 13,
    "mfcc": True,
    "d": True,
    "dd": True,
    "seed": 12345689
}

	Win = esn_model(config=CONF, seed=12345689)
	