from collections import defaultdict

import tqdm
import numpy as np

from reservoirpy import ESN, ESNOnline
from reservoirpy.mat_gen import generate_input_weights
from reservoirpy.mat_gen import fast_spectral_initialization
from reservoirpy.activationsfunc import get_function
from reservoirpy.datasets import mackey_glass


def init_esn(training_type):

    # Common parameters
    n_inputs = 1
    input_bias = True  # add a constant input to 1
    n_outputs = 1
    n_reservoir = 300  # number of recurrent units
    leak_rate = 0.6  # leaking rate (=1/time_constant_of_neurons)
    spectral_radius = 0.5  # Scaling of recurrent matrix
    input_scaling = 1.  # Scaling of input matrix
    regularization_coef = 0.02

    W = fast_spectral_initialization(n_reservoir, sr=spectral_radius)
    Win = generate_input_weights(n_reservoir, n_inputs, input_scaling=input_scaling, input_bias=input_bias)

    if training_type == 'online':
        Wout = np.zeros((n_outputs, n_reservoir + 1))
        esn = ESNOnline(leak_rate, W, Win, Wout,
                        alpha_coef=regularization_coef, input_bias=input_bias)

    elif training_type == 'online_feedback':
        Wout = np.zeros((n_outputs, n_reservoir + 1))
        Wfb = generate_input_weights(n_reservoir, n_outputs, input_bias=False)
        fbfunc = get_function("identity")
        esn = ESNOnline(leak_rate, W, Win, Wout,
                        alpha_coef=regularization_coef,
                        input_bias=input_bias,
                        Wfb=Wfb,
                        fbfunc=fbfunc)

    elif training_type == 'offline':
        esn = ESN(leak_rate, W, Win,
                  input_bias=input_bias, ridge=regularization_coef)
    else:
        raise RuntimeError(f"training_type = [{training_type}] unknown")

    return esn


def evaluate(esn, training_type, result_dict):

    # Run on test set
    output_pred, internal_pred = esn.run([test_in])

    # Compute evaluation metric
    mse = np.mean((test_out[:] - output_pred[0])**2)
    rmse = np.sqrt(mse)
    nmrse_mean = abs(rmse / np.mean(test_out[:]))
    nmrse_maxmin = rmse / abs(np.max(test_out[:]) - np.min(test_out[:]))

    # Save to result_dict
    result_dict[f'mse_{training_type}'].append(mse)
    result_dict[f'rmse_{training_type}'].append(rmse)
    result_dict[f'nmrse_mean_{training_type}'].append(nmrse_mean)
    result_dict[f'nmrse_maxmin_{training_type}'].append(nmrse_maxmin)


data = mackey_glass(2000)

# load the data and select which parts are used for 'warming', 'training' and 'testing' the reservoir
initLen = 20  # number of time steps during which internal activations are washed-out during training
# we consider trainLen including the warming-up period (i.e. internal activations that are washed-out when training)
trainLen = int(0.6*len(data))  # number of time steps during which we train the network
testLen = len(data) - trainLen - 1  # number of time steps during which we test/run the network

print("data dimensions", data.shape)
print("data not normalized",data)
print("max", data.max())
print("min", data.min())
print("mean", data.mean())
print("std", data.std())

# normalizing data
data = data / (data.max() - data.min())
print("data normalized",data)
print("max", data.max())
print("min", data.min())
print("mean", data.mean())
print("std", data.std())

# Split data
train_in = data[0:trainLen]
train_out = data[0+1:trainLen+1]
test_in = data[trainLen:trainLen+testLen]
test_out = data[trainLen+1:trainLen+testLen+1]

# parameters of test runs
eval_metrics = defaultdict(list)
nb_runs = 30

for i in tqdm.trange(nb_runs):
    # Online training without feedback
    esn_online = init_esn('online')
    esn_online.train([train_in], [train_out], wash_nr_time_step=initLen, verbose=False)
    evaluate(esn_online, 'online', eval_metrics)
    # Online training with feedback
    esn_online_feedback = init_esn('online_feedback')
    esn_online_feedback.train([train_in], [train_out], wash_nr_time_step=initLen, verbose=False)
    evaluate(esn_online_feedback, 'online_feedback', eval_metrics)
    # Offline training
    esn_offline = init_esn('offline')
    esn_offline.train([train_in], [train_out], wash_nr_time_step=initLen, verbose=False)
    evaluate(esn_offline, 'offline', eval_metrics)

print("\nEvaluation results")
for k, v in eval_metrics.items():
    print(f"{k}: {np.mean(v)}(mean), {np.std(v)}(std)")
