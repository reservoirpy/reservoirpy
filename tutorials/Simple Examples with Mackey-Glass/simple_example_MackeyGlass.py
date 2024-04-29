"""Echo State Network demo with ReservoirPy on a chaotic timeseries prediction
task.

This script provides some Reservoir Computing and ReservoirPy basics, applying
them on a benchmarking classic: the Mackey-Glass timeseries prediction task.

To go further, we encourage you to follow the "Introduction to Reservoir
Computing" notebook that can be found in the eponym repertory,
within the `tutorials/` repertory.

This is an adaptation of Mantas Lukoševičius [1]_ 2012 great tutorial on
Reservoir Computing, that can also be found in this repository under the name
"minimalESN_MackeyGlass".

References:
-----------
    .. [1] Mantas Lukoševičius personal website: https://mantas.info/
"""

# Author: Xavier Hinaut <xavier.hinaut@inria.fr> and
# Nathan Trouvain at 01/12/2021 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import matplotlib.pyplot as plt
import numpy as np

import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes import Input, Reservoir, Ridge

# Set a particular seed for the random generator (for example seed = 42)
# NB: reservoir performances should be averaged across at least 30 random
# instances (with the same set of parameters)
# If None, then a random seed will be used.
SEED = 42
VERBOSE = True

# Dataset parameters
TIMESTEPS = 10_000
TAU = 17
NORMALIZE = True

TRAIN_SIZE = 2000
TEST_SIZE = 2000
HORIZON = 1  # horizon p of the forecast (predict X[t+p] from X[t])


# ---- A look into Echo State Networks parameters ----

# Input dimension
INPUT_BIAS = True  # add a constant input to 1
N_INPUTS = 1  # input dimension (optional, can be inferred at runtime)
N_OUTPUTS = 1  # output dimension (optional, can be inferred at runtime)

# Reservoir parameter
UNITS = 300  # number of recurrent units
LEAK_RATE = 0.3  # leaking rate (=1/time_constant_of_neurons)
RHO = 1.25  # Scaling of recurrent matrix
INPUT_SCALING = 1.0  # Scaling of input matrix

# Connectivity
# Connectivity defines the probability that two neurons in the
# reservoir are being connected
# (the two neurons can be a neuron and itself)
RC_CONNECTIVITY = 0.2  # Connectivity of recurrent matrix W
INPUT_CONNECTIVITY = 1.0  # Connectivity of input matrix
FB_CONNECTIVITY = 1.0  # Connectivity of feedback matrix

# Readout parameters
REGULARIZATION_COEF = 1e-8
WARMUP = 100

# ---- Model architecture ----

# Enable feedback
FEEDBACK = False
# Adds the input values to the regression along the reservoir states
INPUT_TO_READOUT = False

if __name__ == "__main__":
    rpy.set_seed(SEED)
    # Set verbosity of ReservoirPy objects to 0 (no use here)
    rpy.verbosity(0)

    # ---- Loading data ----
    data = mackey_glass(TIMESTEPS, tau=TAU)

    if VERBOSE:
        print("Data dimensions", data.shape)
        print(f"max: {data.max()} min: {data.min()}")
        print(f"mean: {data.mean()}, std: {data.std()}")

    # Normalize data
    if NORMALIZE:
        data = (data - data.min()) / (data.max() - data.min())
        if VERBOSE:
            print("Normalization...")
            print(f"max: {data.max()} min: {data.min()}")
            print(f"mean: {data.mean()}, std: {data.std()}")

    plt.figure()
    plt.plot(data[:1000])
    plt.title("A sample of input data")
    plt.show()

    # ---- Prepare dataset ----

    X = data[:TRAIN_SIZE]
    y = data[HORIZON : TRAIN_SIZE + HORIZON]

    X_test = data[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]
    y_test = data[TRAIN_SIZE + HORIZON : TRAIN_SIZE + TEST_SIZE + HORIZON]

    if VERBOSE:
        print("X, y dimensions", X.shape, y.shape)
        print("X_test, y_test dimensions", X_test.shape, y_test.shape)

    plt.figure()
    plt.plot(X, y)
    plt.xlabel("$X[t]$")
    plt.ylabel("$y[t]=X[t+p]$")
    if NORMALIZE:
        plt.ylim([-1.1, 1.1])
    plt.title("Recurrence plot of training data: input(t+1) vs. input(t)")
    plt.show()

    plt.figure()
    plt.plot(X, label="X")
    plt.plot(y, label="y")
    plt.xlabel("t")
    if NORMALIZE:
        plt.ylim([-1.1, 1.1])
    plt.legend()
    plt.title("$X[t] and y[t]=X[t+p]$")
    plt.show()

    # ---- Generating random weight matrices with toolbox methods ----
    # (this is optional)

    # !! uncomment if you want to test the more
    # detailed matrix generation method !!

    from reservoirpy import mat_gen

    W = mat_gen.normal(
        UNITS,
        connectivity=RC_CONNECTIVITY,
        sr=RHO,
        seed=SEED,
    )

    Win = mat_gen.bernoulli(
        UNITS,
        N_INPUTS,
        input_scaling=INPUT_SCALING,
        connectivity=INPUT_CONNECTIVITY,
        input_bias=INPUT_BIAS,
        seed=SEED,
    )

    Wfb = mat_gen.bernoulli(
        UNITS,
        N_OUTPUTS,
        connectivity=FB_CONNECTIVITY,
        input_bias=False,
        seed=SEED,
    )

    # ---- Generating random weight matrices with custom method ----
    # (this is also optional)

    rng = np.random.default_rng(SEED)

    W = rng.random((UNITS, UNITS)) - 0.5
    Win = rng.random((UNITS, N_INPUTS + int(INPUT_BIAS))) - 0.5
    Wfb = rng.random((UNITS, N_OUTPUTS)) - 0.5

    # Delete the fraction of connections given the connectivity
    # (i.e. proba of non-zero connections in the reservoir):
    mask = rng.random((UNITS, UNITS))  # create a mask Uniform[0;1]
    W[mask > RC_CONNECTIVITY] = 0  # set to zero some connections

    mask = rng.random((UNITS, Win.shape[1]))
    Win[mask > INPUT_CONNECTIVITY] = 0

    mask = rng.random((UNITS, Wfb.shape[1]))
    Wfb[mask > FB_CONNECTIVITY] = 0

    # Scaling of matrices

    # Scaling of input matrix
    Win = INPUT_SCALING * Win

    # Scaling of recurrent matrix using a specific spectral radius
    # First compute the spectral radius of these weights:

    if VERBOSE:
        print("Computing spectral radius...")

    from reservoirpy.observables import spectral_radius

    # Spectral radius is the maximum absolute norm of the eigenvectors of W.
    # original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    original_spectral_radius = spectral_radius(W)

    if VERBOSE:
        print("Default spectral radius before scaling:", original_spectral_radius)

    # Rescale them to reach the requested spectral radius:
    W = W * (RHO / original_spectral_radius)

    if VERBOSE:
        print("Spectral radius after scaling:", spectral_radius(W))

    # ---- Create an Echo State Network ----

    # Create a reservoir
    reservoir = Reservoir(
        UNITS,
        lr=LEAK_RATE,
        sr=RHO,
        input_bias=INPUT_BIAS,
        input_scaling=INPUT_SCALING,
        rc_connectivity=RC_CONNECTIVITY,
        input_connectivity=INPUT_CONNECTIVITY,
        fb_connectivity=FB_CONNECTIVITY,
        name="reservoir",
    )

    # If you want to use custom matrices, then use:
    custom_reservoir = Reservoir(
        W=W,
        lr=LEAK_RATE,
        input_bias=INPUT_BIAS,
        Win=Win,
        Wfb=Wfb,
        name="custom-reservoir",
    )

    # create a readout layer equipped with an offline learning rule
    readout = Ridge(ridge=REGULARIZATION_COEF, name="readout")

    if FEEDBACK:
        reservoir = reservoir << readout

    if INPUT_TO_READOUT:
        # Connect the inputs directly to the readout
        inputs = Input(name="input")
        esn = [inputs >> reservoir, inputs] >> readout
    else:
        esn = reservoir >> readout

    # ---- Let's have a look at internal states ----

    internal_trained = reservoir.run(X)

    # Reset the reservoir state: we want to start training from scratch
    reservoir.reset()

    plt.figure()
    plt.plot(internal_trained[:200, :21])
    plt.title(
        "Activations $\\mathbf{x}(n)$ from Reservoir "
        "Neurons ID 0 to 20 for 200 time steps"
    )
    plt.show()

    # ---- Train the ESN ----

    esn = esn.fit(X, y, warmup=WARMUP)

    # ---- Evaluate the ESN ----

    y_pred = esn.run(X_test)

    from reservoirpy.observables import mse, nrmse, rmse

    # Mean Squared Error
    # https://en.wikipedia.org/wiki/Mean_squared_error
    mse_score = mse(y_test, y_pred)

    # Root Mean Squared Error
    # https://en.wikipedia.org/wiki/Root-mean-square_deviation
    rmse_score = rmse(y_test, y_pred)

    # Normalised RMSE (based on mean of training data)
    nmrse_mean = nrmse(y_test, y_pred, norm_value=y.mean())

    # Normalised RMSE (based on max - min of training data)
    nmrse_maxmin = nrmse(y_test, y_pred, norm_value=y.max() - y.min())

    print("\n********************")
    print(f"Errors computed over {TEST_SIZE} time steps")
    print("\nMean Squared error (MSE):\t%.4e" % mse_score)
    print("Root Mean Squared error (RMSE):\t%.4e" % rmse_score)
    print("Normalized RMSE (based on mean):\t%.4e" % nmrse_mean)
    print("Normalized RMSE (based on max - min):\t%.4e" % nmrse_maxmin)
    print("********************")

    plt.figure(figsize=(12, 4))
    plt.plot(y_pred, color="red", lw=1.5, label="Predictions")
    plt.plot(y_test, color="blue", lw=0.75, label="Ground truth")
    plt.title("Output predictions against real timeseries")
    plt.legend()
    plt.show()
