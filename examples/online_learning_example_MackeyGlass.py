import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')

from reservoirpy import ESNOnline
from reservoirpy.mat_gen import generate_internal_weights, generate_input_weights




def init_network(input_size, output_size, 
                 reservoir_size = 300, 
                 spectral_radius = 1.25,
                 leaking_rate = 0.3,
                 average_nb_connexions = None, 
                 use_feedback = False,
                 use_raw_input = True,
                 feedback_scaling = 1.,
                 input_scaling = 1.,
                 input_sparsity = 0.1,
                 input_bias = True,
                 
                 alpha_coef = 1e-6):
    
    if average_nb_connexions is None:
        average_nb_connexions = 0.8*reservoir_size
    
    reservoir_weights = generate_internal_weights(reservoir_size, spectral_radius=spectral_radius,
                                                    proba=average_nb_connexions/reservoir_size)
                                      
    if input_bias:
        input_size += 1
    input_weights = generate_input_weights(reservoir_size, input_size,
                                           input_scaling=input_scaling, proba=1.0 - input_sparsity)  
    
    if use_feedback:
        feedback_weights = generate_input_weights(reservoir_size, output_size,
                                                  input_scaling=feedback_scaling, proba=1.0)
        fbfunc = lambda x: x
    else:
        feedback_weights = None
        fbfunc = None
    
    return ESNOnline(lr = leaking_rate,
                     W = reservoir_weights,
                     Win = input_weights,
                     output_size = output_size,
                     alpha_coef = alpha_coef,
                     use_raw_input = use_raw_input,
                     input_bias = input_bias,
                     Wfb = feedback_weights,
                     fbfunc = fbfunc)    


def generate_outputs(nw, nb_outputs = 1, initial_value = None):
    """
        Makes run the network nb_outputs time : if network needs inputs, its last output is used.
        If initial_value is not None, it is used as first input to feed the network
        
        Output : 
            numpy array of network outputs
    """
    
    outputs = []
    
    if initial_value != None:
        next_input = initial_value
    else:
        next_input = getattr(nw, 'output_values').ravel()
    
    for i in range(nb_outputs):
        _, next_input = nw.compute_output(next_input if getattr(nw, 'dim_inp') != 0 else np.array([]))
        outputs.append(next_input)
    
    return np.asarray(outputs)



if __name__ == '__main__':

    train_size = 5000
    test_size = 1000
  
    test_sequence = np.tile(np.loadtxt('MackeyGlass_t17.txt'), 10)        
    test_sequence_darray = [np.array([test_data]) for test_data in test_sequence]
    
    nw = init_network(1, 1, 
                      reservoir_size = 400, 
                      spectral_radius = 1.3, 
                      average_nb_connexions = 3, 
                      use_feedback = False, 
                      leaking_rate = 0.3,
                      use_raw_input = False,
                      alpha_coef = 1e-8)
        
    nw.train(test_sequence_darray[0:train_size], test_sequence_darray[1:train_size+1], wash_nr_time_step = int(0.05 * train_size))
    
    outputs = generate_outputs(nw, nb_outputs=test_size)
    
    fig, ax = plt.subplots(figsize=(30., 20.))    
    ax.plot(test_sequence[train_size + 1 : train_size + test_size + 1])
    ax.plot(outputs)
    ax.legend(["target signal", "generated signal"])
    plt.show()
