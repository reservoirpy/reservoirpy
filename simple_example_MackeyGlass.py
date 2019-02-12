import numpy as np
import matplotlib.pyplot as plt
#from pyESN import ESN
import ESN
# import ESNold as ESN
# import reservoir as ESN

def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""

    # Set the seed
    if seed is None:
        import time
        seed = int((time.time()*10**6) % 4294967295)
    try:
        np.random.seed(seed)
    except Exception as e:
        print( "!!! WARNING !!!: Seed was not set correctly.")
        print( "!!! Seed that we tried to use: "+str(seed))
        print( "!!! Error message: "+str(e))
        seed = None
    print( "Seed used for random values:", seed)
    return seed

## Set a particular seed for the random generator (for example seed = 42), or use a "random" one (seed = None)
# NB: reservoir performances should be averaged accross at least 30 random instances (with the same set of parameters)
seed = 42 #None #42

set_seed(seed) #random.seed(seed)

## load the data and select which parts are used for 'warming', 'training' and 'testing' the reservoir
# 30 seems to be enough for initLen with leak_rate=0.3 and reservoir size (resSize) = 300
initLen = 100 # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = initLen + 1900 # number of time steps during which we train the network
testLen = 2000 # number of time steps during which we test/run the network

data = np.loadtxt('MackeyGlass_t17.txt')
print( "data dimensions", data.shape)

# plot some of it
plt.figure(0)
plt.plot(data[0:1000])
plt.ylim([-1.1,1.1])
plt.title('A sample of input data')

# generate the ESN reservoir
# inSize = outSize = 1 #input/output dimension
# resSize = 300 #reservoir size (for prediction)
# resSize = 1000 #reservoir size (for generation)
# spectral_radius = 1.25
# input_scaling = 1.


n_inputs = 1
input_bias = True # add a constant input to 1
n_outputs = 1
n_reservoir = 300 # number of recurrent units
leak_rate = 0.3 # leaking rate (=1/time_constant_of_neurons)
spectral_radius = 1.25 # Scaling of recurrent matrix
input_scaling = 1. # Scaling of input matrix
proba_non_zero_connec_W = 0.2 # Sparsity of recurrent matrix: Perceptage of non-zero connections in W matrix
proba_non_zero_connec_Win = 1. # Sparsity of input matrix
proba_non_zero_connec_Wfb = 1. # Sparsity of feedback matrix
regularization_coef =  1e-8 #None # regularization coefficient, if None, pseudo-inverse is use instead of ridge regression
# out_func_activation = lambda x: x

N = n_reservoir#100
dim_inp = n_inputs #26

## Generating random weight matrices with toolbox methods
#import mat_gen
# W = mat_gen.generate_internal_weights(N=self.N, spectral_radius=self.sr, proba=self.w_proba,
# #                                            seed=seed, verbose=verbose)
#                                 Wstd=self.Wstd, seed=current_seed, verbose=verbose)
# Win = mat_gen.generate_input_weights(nbr_neuron=self.N, dim_input=self.stim_sent_train[0].shape[1], #TODO stim_sent_train[0].shape
#                                 input_scaling=self.iss, proba=self.w_in_proba, input_bias=self.input_bias, seed=current_seed, verbose=verbose)
# Wfb = mat_gen.generate_input_weights(nbr_neuron=self.N, dim_input=self.dim_output, #TODO stim_sent_train[0].shape
#                                 input_scaling=self.fbscale, proba=self.fbproba, input_bias=None, seed=current_seed, verbose=verbose)

### Generating random weight matrices with custom method
W = np.random.rand(N,N) - 0.5
if input_bias:
    Win = np.random.rand(N,dim_inp+1) - 0.5
else:
    Win = np.random.rand(N,dim_inp) - 0.5
Wfb = np.random.rand(N,n_outputs) - 0.5

# # Mantas way
# Win = (np.random.rand(N,1+dim_inp)-0.5) * input_scaling
# W = np.random.rand(N,N)-0.5

## delete the fraction of connections given the sparsity (i.e. proba of non-zero connections):
mask = np.random.rand(N,N) # create a mask Uniform[0;1]
W[mask > proba_non_zero_connec_W] = 0 # set to zero some connections given by the mask
mask = np.random.rand(N,Win.shape[1])
Win[mask > proba_non_zero_connec_Win] = 0
# mask = np.random.rand(N,Wfb.shape[1])
# Wfb[mask > proba_non_zero_connec_Wfb] = 0

## SCALING of matrices
# scaling of input matrix
Win = Win * input_scaling
# scaling of recurrent matrix
# compute the spectral radius of these weights:
print( 'Computing spectral radius...')
original_spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
#TODO: check if this operation is quicker: max(abs(linalg.eig(W)[0])) #from scipy import linalg
print( "default spectral radius before scaling:", original_spectral_radius)
# rescale them to reach the requested spectral radius:
W = W * (spectral_radius / original_spectral_radius)
print( "spectral radius after scaling", np.max(np.abs(np.linalg.eigvals(W))))


reservoir = ESN.ESN(lr=leak_rate, W=W, Win=Win, input_bias=input_bias, ridge=regularization_coef, Wfb=None, fbfunc=None)


train_in = data[None,0:trainLen]
train_out = data[None,0+1:trainLen+1]
test_in = data[None,trainLen:trainLen+testLen]
test_out = data[None,trainLen+1:trainLen+testLen+1]

# train_in, train_out =  np.vstack([data[0:trainLen],data[0:trainLen]]), np.vstack([data[0+1:trainLen+1], data[0+1:trainLen+1]])
# test_in, test_out =  np.vstack([data[trainLen:trainLen+testLen],data[trainLen:trainLen+testLen]]) , np.vstack([data[trainLen+1:trainLen+testLen+1],data[trainLen+1:trainLen+testLen+1]])

# train_in, train_out =  np.atleast_2d(data[0:trainLen]), np.atleast_2d(data[0+1:trainLen+1])
# test_in, test_out =  np.atleast_2d(data[trainLen:trainLen+testLen]), np.atleast_2d(data[trainLen+1:trainLen+testLen+1])


# rearange inputs in correct dimensions
train_in, train_out = train_in.T, train_out.T
test_in, test_out = test_in.T, test_out.T

# Dimensions of input/output train/test data
print( "train_in, train_out dimensions", train_in.shape, train_out.shape)
print( "test_in, test_out dimensions", test_in.shape, test_out.shape)

plt.figure()
plt.plot(train_in, train_out)
plt.ylim([-1.1,1.1])
plt.title('Recurrence plot of training data: input(t+1) vs. input(t)')
plt.figure()
plt.plot(train_in)
plt.plot(train_out)
plt.ylim([-1.1,1.1])
plt.legend(['train_in','train_out'])
plt.title('train_in & train_out')
# plt.figure()
# plt.plot(test_in)
# plt.plot(test_out)
# plt.ylim([-1.1,1.1])
# plt.legend(['test_in','test_out'])
# plt.title('test_in & test_out')


internal_trained = reservoir.train(inputs=[train_in,], teachers=[train_out,], wash_nr_time_step=initLen, verbose=False)
output_pred, internal_pred = reservoir.run(inputs=[test_in,], reset_state=False)
errorLen = len(test_out[:]) #testLen #2000

## printing errors made on test set
# mse = sum( np.square( test_out[:] - output_pred[0] ) ) / errorLen
# print( 'MSE = ' + str( mse ))
mse = np.mean((test_out[:] - output_pred[0])**2) # Mean Squared Error: see https://en.wikipedia.org/wiki/Mean_squared_error
rmse = np.sqrt(mse) # Root Mean Squared Error: see https://en.wikipedia.org/wiki/Root-mean-square_deviation for more info
nmrse_mean = abs(rmse / np.mean(test_out[:])) # Normalised RMSE (based on mean)
nmrse_maxmin = rmse / abs(np.max(test_out[:]) - np.min(test_out[:])) # Normalised RMSE (based on max - min)
print("\n********************")
print("Errors computed over %d time steps" % (errorLen))
print("\nMean Squared error (MSE):\t\t%.4e" % (mse) )
print("Root Mean Squared error (RMSE):\t\t%.4e\n" % rmse )
print("Normalized RMSE (based on mean):\t%.4e" % (nmrse_mean) )
print("Normalized RMSE (based on max - min):\t%.4e" % (nmrse_maxmin) )
print("********************\n")

plt.figure()
plt.plot( internal_trained[0][:200,:12])
plt.ylim([-1.1,1.1])
plt.title('Activations $\mathbf{x}(n)$ from Reservoir Neurons ID 0 to 11 for 200 time steps')

plt.figure(figsize=(12,4))
plt.plot(test_out,  color='0.75', lw=1.0)
plt.plot(output_pred[0], color='0.00', lw=1.5)
plt.title("Ouput predictions against real timeseries")
plt.legend(["real timeseries", "output predictions"])
# plt.ylim(-1.1,1.1)
plt.show()
