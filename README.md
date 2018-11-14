# FunkyReservoir
A simple and flexible code for Reservoir Computing architectures like Echo State Networks.
This toolbox works for Python 2. Please send a request if you need it for Python 3, because the update to Python 3 should not be long.

## Quick try
#### Chaotic timeseries prediction (MackeyGlass)
Run and analyse these two files to see how to make timeseries prediction with Echo State Networks:
- simple_example_MackeyGlass.py (using the ESN class)
    `bash
    python imple_example_MackeyGlass.py
    `
- minimalESN_MackeyGlass.py (without the ESN class)
```bash
python imple_example_MackeyGlass.py
```

## Using the ESN class for your own application
You can do this in a few steps:
1. Define the number of dimension of input, recurrent and outputs layers:
    `python
    n_inputs = 1
    input_bias = True # add a constant input to 1
    n_outputs = 1
    N = 300 # number of recurrent units
    `

2. Define yourself the random input Win and recurrent W matrices:
    `python
    import numpy as np
    Win = (np.random.rand(N,1+dim_inp)-0.5) * input_scaling
    W = np.random.rand(N,N)-0.5
    print 'Computing spectral radius...',
    original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (spectral_radius / original_spectral_radius) # rescale W to reach the requested spectral radius
    `
Or use the tools available in mat_gen.py for automatic generation method:
    `python
    import mat_gen
    W = mat_gen.generate_internal_weights(N=N, spectral_radius=1.0, proba=0.2, Wstd=1.0)
    Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=n_inputs, input_scaling=1.0, proba=0.2, input_bias=input_bias)
    `

3. Define the Echo State Network (ESN):
     `python
     import ESN
     reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None)
     `

4. Define your input/output training and testing data. Here want to perform the prediction of the chaotic MackeyGlass timeseries:
    `python
    data = np.loadtxt('MackeyGlass_t17.txt')
    train_in = data[None,0:trainLen].T
    train_out = data[None,0+1:trainLen+1].T
    test_in = data[None,trainLen:trainLen+testLen].T
    test_out = data[None,trainLen+1:trainLen+testLen+1].T
    `

5. Train the ESN:
Be careful to give lists for the input and output (i.e. teachers) training data. Here we are training with only one timeseries, but you actually can provide a list of timeseries segments to train from.
**wash_nr_time_step** defines the initial warming-up period: corresponding reservoir states are discarded for training.
    `python
    internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=100)
    `

6. Test the ESN (i.e. predict the next value in the timeseries):
    `python
    output_pred, internal_pred = reservoir.run(inputs=[test_in,], reset_state=False)
    `

7. Compute the error made on test data:

    `python
    print("\nRoot Mean Squared error:")
    # print(np.sqrt(np.mean((output_pred[0] - test_out)**2)))
    print(np.sqrt(np.mean((output_pred[0] - test_out)**2))/errorLen)
    `

8. Plot the internal states of the ESN and the outputs for test data. If the training was sucessful, predicted output and real curves should overlap:

  `python
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
  `
