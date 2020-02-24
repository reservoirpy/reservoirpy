# ReservoirPy (FunkyReservoir)
A simple and flexible code for Reservoir Computing architectures like Echo State Networks.

This toolbox works for Python 3 (and should be compatible for Python2). We just updated it from Python 2 to 3, so tell us if you have any issue with it.

## Quick try
#### Chaotic timeseries prediction (MackeyGlass)
Run and analyse these two files to see how to make timeseries prediction with Echo State Networks:
- simple_example_MackeyGlass.py (using the ESN class)

    ```bash
    python simple_example_MackeyGlass.py
    ```

- minimalESN_MackeyGlass.py (without the ESN class)

    ```bash
    python minimalESN_MackeyGlass.py
    ```

## How to use the ESN class
You can generate and train a reservoir to predict the MackeyGlass timeseries in a few steps:
1. Define the number of dimension of input, recurrent and outputs layers:

    ```python
    n_inputs = 1
    input_bias = True # add a constant input to 1
    n_outputs = 1
    N = 300 # number of recurrent units
    ```

2. Define the random input Win and recurrent W matrices (quick method with generation tools):

    ```python
    import mat_gen
    W = mat_gen.generate_internal_weights(N=N, spectral_radius=1.0, proba=1.0, Wstd=1.0) # Normal distribution with mean 0 and standard deviation 0
    Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=n_inputs, input_scaling=1.0, proba=1.0, input_bias=input_bias)
    ```

3. (instead of previous step) Define yourself the random input Win and recurrent W matrices (customize method):

    ```python
    import numpy as np

    # Generating matrices Win and W
    W = np.random.rand(N,N) - 0.5
    if input_bias:
        Win = np.random.rand(N,dim_inp+1) - 0.5
    else:
        Win = np.random.rand(N,dim_inp) - 0.5

    # Apply mask to make matrices sparse
    proba_non_zero_connec_W = 0.2 # set the probability of non-zero connections
    mask = np.random.rand(N,N) # create a mask with Uniform[0;1] distribution
    W[mask > proba_non_zero_connec_W] = 0 # apply mask on W: set to zero some connections given by the mask
    mask = np.random.rand(N,Win.shape[1]) # Do the same for input matrix
    Win[mask > proba_non_zero_connec_Win] = 0

    # Scaling matrices Win and W
    input_scaling = 1.0 # Define the scaling of the input matrix Win
    Win = Win * input_scaling # Apply scaling
    spectral_radius = 1.0 # Define the scaling of the recurrent matrix W
    print 'Computing spectral radius ...',
    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / original_spectral_radius) # Rescale W to reach the requested spectral radius
    ```

4. Define the Echo State Network (ESN):
     ```python
     import ESN
     reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None)
     ```

5. Define your input/output training and testing data:

    In this step, we load the dataset to perform the prediction of the chaotic MackeyGlass timeseries, and we split the data into the different subsets.

    ```python
    data = np.loadtxt('MackeyGlass_t17.txt')
    train_in = data[None,0:trainLen].T # input data (TRAINING PHASE)
    train_out = data[None,0+1:trainLen+1].T # output to be predicted (TRAINING PHASE)
    test_in = data[None,trainLen:trainLen+testLen].T # input data (TESTING PHASE)
    test_out = data[None,trainLen+1:trainLen+testLen+1].T # output to be predicted (TESTING PHASE)
    ```

6. Train the ESN:

    Be careful to give lists for the input and output (i.e. teachers) training data. Here we are training with only one timeseries, but you actually can provide a list of timeseries segments to train from.

    **wash_nr_time_step** defines the initial warming-up period: corresponding reservoir states are discarded for training.

    ```python
    internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=100)
    ```

7. Test the ESN (i.e. predict the next value in the timeseries):
    ```python
    output_pred, internal_pred = reservoir.run(inputs=[test_in,], reset_state=False)
    ```

8. Compute the error made on test data:

    ```python
    print("\nRoot Mean Squared error:")
    print(np.sqrt(np.mean((output_pred[0] - test_out)**2))/testLen)
    ```

9. Plot the internal states of the ESN and the outputs for test data.

    If the training was sucessful, predicted output and real curves should overlap:

    ```python
    import matplotlib.pyplot as plt
    ## Plot internal states of the ESN
    plt.figure()
    plt.plot( internal_trained[0][:200,:12])
    plt.ylim([-1.1,1.1])
    plt.title('Activations $\mathbf{x}(n)$ from Reservoir Neurons ID 0 to 11 for 200 time steps')

    ## Plot the predicted output with the real one.
    plt.figure(figsize=(12,4))
    plt.plot(test_out,  color='0.75', lw=1.0)
    plt.plot(output_pred[0], color='0.00', lw=1.5)
    plt.title("Ouput predictions against real timeseries")
    plt.legend(["real timeseries", "output predictions"])
    # plt.ylim(-1.1,1.1)
    plt.show()
    ```

If you want to have more information on all the steps and more option (for example, have a reservoir with output feedback), please have a look at **simple_example_MackeyGlass.py**.
