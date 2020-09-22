# ReservoirPy (v0.2)
**A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

ReservoriPy is a simple user-friendly library based on Python scientific modules. It provides a flexible interface to implement efficient Reservoir Computing (RC) architectures with a particular focus on Echo State Networks (ESN). Advanced features of ReservoirPy allow to improve computation time efficiency on a simple laptop compared to basic Python implementation. Some of its features are: offline and online training, parallel implementation, sparse matrix computation, fast spectral initialization, etc. Moreover, graphical tools are included to easily explore hyperparameters with the help of the hyperopt library.          

This library works for Python 3 (and should be compatible for Python2). We just updated it from Python 2 to 3, so tell us if you have any issue with it.

## Installation

This library is not yet available through PyPI, but can be installed with `pip` using a local clone of this repository:

```bash
# clone or download the repository
# in "path/to/local/copy"
git clone https://github.com/neuronalX/reservoirpy.git

# install the repository
pip install path/to/local/copy/reservoirpy/.
```

## Versions
**To enable last features of ReservoirPy, you migth want to download a specific Git branch.**

Available versions and corresponding branch:
- v0.1.x : `v0.1`
- v0.2.x (last stable) : `master`
- v0.2.x (dev) : `v0.2-dev`
- (comming soon) v0.3.0 : `v0.3`

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

## Preprint with tutorials
Tutorial on ReservoirPy can be found in this [preprint (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

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

3. Define the Echo State Network (ESN):
     ```python
     from reservoirpy import ESN
     reservoir = ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None)
     ```

4. Define your input/output training and testing data:

    In this step, we load the dataset to perform the prediction of the chaotic MackeyGlass timeseries, and we split the data into the different subsets.

    ```python
    data = np.loadtxt('MackeyGlass_t17.txt')
    train_in = data[None,0:trainLen].T # input data (TRAINING PHASE)
    train_out = data[None,0+1:trainLen+1].T # output to be predicted (TRAINING PHASE)
    test_in = data[None,trainLen:trainLen+testLen].T # input data (TESTING PHASE)
    test_out = data[None,trainLen+1:trainLen+testLen+1].T # output to be predicted (TESTING PHASE)
    ```

5. Train the ESN:

    Be careful to give lists for the input and output (i.e. teachers) training data. Here we are training with only one timeseries, but you actually can provide a list of timeseries segments to train from.

    **wash_nr_time_step** defines the initial warming-up period: corresponding reservoir states are discarded for training.

    ```python
    internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=100)
    ```

6. Test the ESN (i.e. predict the next value in the timeseries):
    ```python
    output_pred, internal_pred = reservoir.run(inputs=[test_in,])
    ```

7. Compute the error made on test data:

    ```python
    print("\nRoot Mean Squared error:")
    print(np.sqrt(np.mean((output_pred[0] - test_out)**2))/testLen)
    ```

8. Plot the internal states of the ESN and the outputs for test data.

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

## Explore Hyper-Parameters with Hyperopt
Tutorial on how to explore hyperparameters with ReservoirPy and Hyperopt can be found in this [preprint (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

More info on hyperopt: [Official website](http://hyperopt.github.io/hyperopt/)

## Cite
Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
