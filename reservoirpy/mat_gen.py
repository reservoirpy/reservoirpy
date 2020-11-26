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
from scipy.sparse.linalg import eigs


def fast_spectral_initialization(N,
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
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    if seed is not None:
        np.random.seed(seed)

    if spectral_radius is None or proba == 0.:
        a = 1
    else:
        a = -(6 * spectral_radius) / (np.sqrt(12) * np.sqrt((proba * N)))

    if proba < 1:
        return sparse.random(N, N, density=proba, format=sparsity_type,
                             data_rvs=lambda s: np.random.uniform(a, -a, size=s))

    else:
        return np.random.uniform(a, -a, size=(N, N))


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
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    # ! this is deprecated in Numpy and should change
    if seed is not None:
        np.random.seed(seed)

    # TODO: put this in a function
    if sparsity_type is not None:
        w = sparse.random(N, N, density=proba, format=sparsity_type,
                          data_rvs=lambda s: np.random.uniform(-1, 1, size=s))
        rhoW = max(abs(eigs(w, k=1, which='LM', return_eigenvectors=False)))
    else:
        mask = 1 * (np.random.rand(N, N) < proba)
        mat = np.random.normal(0, Wstd, (N, N))
        w = np.multiply(mat, mask, dtype=typefloat)

        # Computing the spectral radius of W matrix
        rhoW = max(abs(linalg.eig(w)[0]))

    # TODO: use logging
    if verbose:
        print( "Spectra radius of generated matrix before applying another spectral radius: "+str(rhoW))

    if spectral_radius is not None:
        w *= spectral_radius / rhoW
        if verbose:
            if sparse.issparse(w):
                rhoW_after = max(abs(sparse.linalg.eigs(w, k=1, which='LM', return_eigenvectors=False)))
            else:
                rhoW_after = max(abs(linalg.eig(w)[0]))
            print("Spectra radius matrix after applying another spectral radius: "+str(rhoW_after))

    # ? remove this in future versions ?
    if randomize_seed_afterwards:  # pragma: no cover

        warnings.warn("Have to check if you really want to randomize the seed, \
            because this makes the whole experiment not reproducible.", UserWarning)

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
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    if seed is not None:
        np.random.seed(seed)

    if input_bias:
        dim_input += 1
    mask = 1 * (np.random.rand(nbr_neuron, dim_input) < proba)
    mat = np.random.randint(0, 2, (nbr_neuron, dim_input)) * 2 - 1
    w = np.multiply(mat, mask, dtype=typefloat)
    if input_scaling is not None:
        w = input_scaling * w

    # ? remove this in future versions ?
    if randomize_seed_afterwards:  # pragma: no cover
        warnings.warn("Have to check if you really want to randomize the seed, \
                      because this makes the whole experiment not reproducible.", \
                      UserWarning)
        np.random.seed(int(time.time()*10**6))

    return w


def _is_probability(proba):
    return 1. - proba >= 0. and proba >= 0.
