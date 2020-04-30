# -*- coding: utf-8 -*-
"""
Created on 11 juil. 2012
Modified 2018

@author: Xavier HINAUT
xavier.hinaut #/at\# inria.fr
"""
import time
import warnings

import numpy as np
from scipy import linalg
from scipy import sparse
import random


def fast_spectra_initialization(N,
        spectral_radius=None,
        proba=0.1,
        seed=None,
        verbose=False,
        sparsity_type='csr',
        typefloat=np.float64):
    """
    The fast spectral radius (FSI) approach for weights initialization introduced in 
    [C. Gallicchio, A. Micheli, L. Pedrelli. Fast Spectral Radius Initialization for Recurrent Neural Networks. (2020)]

    Inputs :
        - N: number of neurons
        - spectral_radius: SR
        - proba: probability of non-zero connections (sparsity), usually between 0.05 to 0.30
        - seed: if not None, set the seed of the numpy.random generator to the given value.        
        - verbose: print( in the console detailed information.
        - sparsity_type: the type of sparse matrix.

    """
    
    if seed is not None:
        np.random.seed(seed)

    a = -(6*spectral_radius)/(np.sqrt(12) * np.sqrt((proba * N)))
    if proba<1:
        return sparse.random(N, N, density=proba, format=sparsity_type, data_rvs=lambda s: np.random.uniform(a, -a, size=s))
    
    else:
        return np.random.uniform(a,-a, size=(N,N))              
        
        
def generate_internal_weights(N,
                              spectral_radius=None,
                              proba=0.1,
                              Wstd=1.0,
                              seed=None,
                              randomize_seed_afterwards=False,
                              verbose=False,
                              sparsity_type=None,
                              typefloat=np.float64):
    """
    Method that generate the weight matrix that will be used for the internal connections of the Reservoir.

    Inputs :
        - N: number of neurons
        - spectral_radius: SR
        - proba: probability of non-zero connections (sparsity), usually between 0.05 to 0.30
        - verbose: print( in the console detailed information.
        - seed: if not None, set the seed of the numpy.random generator to the given value.
        - sparsity_type: the type of sparse matrix.        
        - randomize_seed_afterwards: as the module mdp.numx.random may not be used only by this method,
            the user may want to run several experiments with the same seed only for this method
            (generating the internal weights of the Reservoir), but have random seed for all other
            methods that will use mdp.numx.random.
    """
    if seed is not None:
        # mdp.numx.random.seed(seed)
        np.random.seed(seed)
    # mask = 1*(mdp.numx_rand.random((N,N))<proba)
    # mat = mdp.numx.random.normal(0, 1, (N,N)) #equivalent to mdp.numx.random.randn(n, m) * sd + mu
    # w = mdp.numx.multiply(mat, mask)
    if sparsity_type is not None:
        w = sparse.random(N, N, density=proba, format=sparsity_type, data_rvs=lambda s: np.random.uniform(-1, 1, size=s))
        rhoW = max(abs(sparse.linalg.eigs(w, k=1, which='LM', return_eigenvectors=False)))
    else:    
        mask = 1 * (np.random.rand(N, N) < proba)
        mat = np.random.normal(0, Wstd, (N,N)) #equivalent to mdp.numx.random.randn(n, m) * sd + mu
        w = np.multiply(mat, mask,dtype=typefloat)
        # Computing the spectral radius of W matrix
        rhoW = max(abs(linalg.eig(w)[0]))
    if verbose:
        # print( "Spectra radius of generated matrix before applying another spectral radius: "+str(Oger.utils.get_spectral_radius(w)))
        print( "Spectra radius of generated matrix before applying another spectral radius: "+str(rhoW))
    if spectral_radius is not None:
        w *= spectral_radius / rhoW
        if verbose:
            if csr_matrix:
                rhoW_after = max(abs(sparse.linalg.eigs(w, k=1, which='LM', return_eigenvectors=False)))
            else:
                rhoW_after = max(abs(linalg.eig(w)[0]))
            print( "Spectra radius matrix after applying another spectral radius: "+str(rhoW_after))
    if randomize_seed_afterwards:
        """ redifine randomly the seed in order to not fix the seed also for other methods that are using numpy.random methods.
        """
        warnings.warn("Have to check if you really want to randomize the seed, \
            because this makes the whole experiment not reproducible.", UserWarning)
        # mdp.numx.random.seed(int(time.time()*10**6))
        np.seed(int(time.time()*10**6))
        
    return w


def generate_input_weights(nbr_neuron,
                           dim_input,
                           input_scaling=None,
                           proba=0.1,
                           input_bias=False,
                           seed=None,
                           randomize_seed_afterwards=False,
                           verbose=False,
                           typefloat=np.float64):
    """
    Method that generate the weight matrix that will be used for the input connections of the Reservoir.

    Inputs :
        - nbr_neuron: number of neurons
        - dim_input: dimension of the inputs
        - input_scaling: ISS
        - proba: probability of non-zero connections (sparsity), usually between 0.05 to 0.30
        - verbose: print( in the console detailed information.
        - seed: if not None, set the seed of the numpy.random generator to the given value.
        - randomize_seed_afterwards: as the module mdp.numx.random may not be used only by this method,
            the user may want to run several experiments with the same seed only for this method
            (generating the internal weights of the Reservoir), but have random seed for all other
            methods that will use mdp.numx.random.
    """
    if seed is not None:
        # mdp.numx.random.seed(seed)
        np.random.seed(seed)
    # mask = 1*(mdp.numx_rand.random((nbr_neuron, dim_input))<proba)
    # mat = mdp.numx.random.randint(0, 2, (nbr_neuron, dim_input)) * 2 - 1
    # w = mdp.numx.multiply(mat, mask)
    if input_bias:
        dim_input += 1
    mask = 1 * (np.random.rand(nbr_neuron, dim_input) < proba)
    mat = np.random.randint(0, 2, (nbr_neuron, dim_input)) * 2 - 1
    w = np.multiply(mat, mask,dtype=typefloat)
    if input_scaling is not None:
        w = input_scaling * w
    if randomize_seed_afterwards:
        """ redifine randomly the seed in order to not fix the seed also for other methods that are using numpy.random methods.
        """
        warnings.warn("Have to check if you really want to randomize the seed, \
            because this makes the whole experiment not reproducible.", UserWarning)
        # mdp.numx.random.seed(int(time.time()*10**6))
        np.random.seed(int(time.time()*10**6))
    return w
