import matplotlib.pyplot as plt
import numpy as np

import reservoirpy as rpy
from reservoirpy.datasets import mackey_glass
from reservoirpy.nodes import ESN, Reservoir, Ridge, Input

## Set a particular seed for the random generator (for example seed = 42),
# or use a "random" one (seed = None)
# NB: reservoir performances should be averaged accross at least 30 random
# instances (with the same set of parameters)
SEED = 42  # None #42
VERBOSE = True

if __name__ == '__main__':

    rpy.set_seed(SEED)  # random.seed(seed)
    rpy.verbosity(0)

    # ---- Loading data ----
    units = 10000
    tau = 17
    data = mackey_glass(10000, tau=tau)

    normalize = True

    if VERBOSE:
        print("data dimensions", data.shape)
        print("max", data.max(), "min", data.min())
        print("mean", data.mean(), "std", data.std())

    # normalizing data
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
        if VERBOSE:
            print("max", data.max(), "min", data.min())
            print("mean", data.mean(), "std", data.std())

    # plot some of it
    plt.figure()
    plt.plot(data[:1000])
    plt.title('A sample of input data')
    plt.show()

    # ---- A look into Echo State Networks parameters ----

    # Input dimension
    input_bias = True  # add a constant input to 1
    n_inputs   = 1     # input dimension (optional, can be infered at runtime)
    n_outputs  = 1    # output dimension (optional, can be infered at runtime)

    # Reservoir parameter
    units              = 300  # number of recurrent units
    leak_rate          = 0.3  # leaking rate (=1/time_constant_of_neurons)
    rho                = 1.25  # Scaling of recurrent matrix
    input_scaling      = 1.  # Scaling of input matrix
    rc_connectivity    = 0.2  # Sparsity of recurrent matrix W: % of
    input_connectivity = 1.  # Sparsity of input matrix
    fb_connectivity    = 1.  # Sparsity of feedback matrix

    # Readout parameters
    regularization_coef = 1e-8
    warmup = 100

    # Enable feedback
    feedback = False

    # ---- Generating random weight matrices with toolbox methods ----
    # (this is optional)

    # !! uncomment if you want to test the more
    # detailed matrix generation method !!

    # from reservoirpy import mat_gen
    # W = mat_gen.generate_internal_weights(units, spectral_radius=rho,
    #                                       proba=rc_connectivity, seed=SEED)
    #
    # Win = mat_gen.generate_input_weights(units, n_inputs,
    #                                      input_scaling=input_scaling,
    #                                      proba=input_connectivity,
    #                                      input_bias=input_bias,
    #                                      seed=SEED)
    #
    # Wfb = mat_gen.generate_input_weights(units, n_outputs,
    #                                      proba=fb_connectivity,
    #                                      input_bias=False,
    #                                      seed=SEED)

    # ---- Generating random weight matrices with custom method ----
    # (this is also optional)

    rng = np.random.default_rng(SEED)

    W = rng.random((units, units)) - 0.5
    Win = rng.random((units, n_inputs + int(input_bias))) - 0.5
    Wfb = rng.random((units, n_outputs)) - 0.5

    # delete the fraction of connections given the sparsity (i.e. proba of
    # non-zero connections):
    mask = rng.random((units, units))  # create a mask Uniform[0;1]
    W[mask > rc_connectivity] = 0  # set to zero some connections

    # given by the mask
    mask = rng.random((units, Win.shape[1]))
    Win[mask > input_connectivity] = 0

    mask = rng.random((units, Wfb.shape[1]))
    Wfb[mask > fb_connectivity] = 0

    # Scaling of matrices

    # Scaling of input matrix
    Win = Win * input_scaling

    # Scaling of recurrent matrix using a specific spectral radius
    # First compute the spectral radius of these weights:

    if VERBOSE:
        print('Computing spectral radius...')

    from reservoirpy.observables import spectral_radius

    #original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    original_spectral_radius = spectral_radius(W)

    if VERBOSE:
        print("Default spectral radius before scaling:",
              original_spectral_radius)

    # Rescale them to reach the requested spectral radius:
    W = W * (rho / original_spectral_radius)

    if VERBOSE:
        print("Spectral radius after scaling:", spectral_radius(W))

    # ---- Prepare dataset ----

    train_size = 2000
    test_size  = 2000
    horizon    = 1

    X = data[:train_size]
    y = data[horizon:  train_size+horizon]

    X_test = data[train_size: train_size+test_size]
    y_test = data[train_size+horizon: train_size+test_size+horizon]

    if VERBOSE:
        print("X, y dimensions", X.shape, y.shape)
        print("X_test, y_test dimensions", X_test.shape, y_test.shape)

    plt.figure()
    plt.plot(X, y)
    plt.xlabel("$X[t]$")
    plt.ylabel("$y[t]=X[t+p]$")
    if normalize:
        plt.ylim([-1.1, 1.1])
    plt.title('Recurrence plot of training data: input(t+1) vs. input(t)')
    plt.show()

    plt.figure()
    plt.plot(X, label="X")
    plt.plot(y, label="y")
    plt.xlabel("t")
    if normalize:
        plt.ylim([-1.1, 1.1])
    plt.legend()
    plt.title("$X[t] and y[t]=X[t+p]$")
    plt.show()

    # ---- Create an Echo State Network ----

    reservoir = Reservoir(units,
                          lr=leak_rate,
                          sr=rho,
                          input_bias=input_bias,
                          input_scaling=input_scaling,
                          rc_connectivity=rc_connectivity,
                          input_connectivity=input_connectivity,
                          fb_connectivity=fb_connectivity,
                          name="reservoir")

    readout = Ridge(ridge=regularization_coef,
                    transient=warmup,
                    name="readout")

    if feedback:
        reservoir = reservoir << readout

    esn = reservoir >> readout

    # (optional) Connect the inputs directly to the readout
    # inputs = Input(name="input")
    # esn = [inputs >> reservoir, inputs] >> readout

    # ---- Let's have a look at internal states ----

    internal_trained = reservoir.run(X)

    # Reset the reservoir state: we want to start training from scratch
    reservoir.reset()

    plt.figure()
    plt.plot(internal_trained[:200, :21])
    plt.title('Activations $\\mathbf{x}(n)$ from Reservoir '
              'Neurons ID 0 to 20 for 200 time steps')
    plt.show()

    # ---- Train the ESN ----

    esn = esn.fit(X, y)

    # ---- Evaluate the ESN ----

    y_pred = esn.run(X_test)

    from reservoirpy.observables import mse, rmse, nrmse

    ## printing errors made on test set
    # mse = sum( np.square( y_test[:] - y_pred[0] ) ) / errorLen
    # print( 'MSE = ' + str( mse ))
    mse_score = mse(y_test, y_pred)  # Mean Squared Error: see
    # https://en.wikipedia.org/wiki/Mean_squared_error
    rmse_score = rmse(y_test, y_pred)# Root Mean Squared Error: see
    # https://en.wikipedia.org/wiki/Root-mean-square_deviation for more info
    nmrse_mean = nrmse(y_test, y_pred, norm_value=y.mean())  # Normalised RMSE (based on mean)
    nmrse_maxmin = nrmse(y_test, y_pred, norm_value=y.max()-y.min())# Normalised RMSE (based on max - min)

    print("\n********************")
    print(f"Errors computed over {test_size} time steps")
    print("\nMean Squared error (MSE):\t%.4e" % mse_score)
    print("Root Mean Squared error (RMSE):\t%.4e" % rmse_score)
    print("Normalized RMSE (based on mean):\t%.4e" % nmrse_mean)
    print("Normalized RMSE (based on max - min):\t%.4e" % nmrse_maxmin)
    print("********************")

    plt.figure(figsize=(12, 4))
    plt.plot(y_pred, color='red', lw=1.5, label="Predictions")
    plt.plot(y_test, color='blue', lw=0.75, label="Ground truth")
    plt.title("Output predictions against real timeseries")
    plt.legend()
    plt.show()
