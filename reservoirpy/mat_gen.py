# -*- coding: utf-8 -*-
"""
Created on 11 juil. 2012
Modified 2018

@author: Xavier HINAUT
xavier.hinaut #/at/# inria.fr
"""
import time
import warnings

import numpy as np

from numpy.random import RandomState
from scipy import linalg
from scipy import sparse
from scipy.sparse.linalg import eigs


def fast_spectral_initialization(N: int,
                                 spectral_radius: float = None,
                                 proba: float = 0.1,
                                 seed: int = None,
                                 verbose: bool = False,
                                 sparsity_type: str = 'csr',
                                 typefloat=np.float64):
    """The fast spectral radius (FSI) approach for weights
    initialization introduced in [1]_.

    Parameters
    ----------
    N : int
        Number of reservoir units
    spectral_radius : float, optional
        Maximum desired eigenvalue of the
        reservoir weights matrix, by default None
    proba : float, optional
        Probability of non zero connection,
        density of the weight matrix, by default 0.1
    seed : int, optional
        Random state generator seed, for reproducibility,
        by default None
    verbose : bool, optional
    sparsity_type : {"csr", "csc", "coo"} optional
        Scipy sparse matrix format. "csr" by default.
    typefloat : np.float64, optional

    Returns
    -------
    np.ndarray or scipy.sparse matrix
        A reservoir weights matrix.

    Raises
    ------
    ValueError
        Invalid non zero connection probability.

    References:
    -----------
        .. _[1] C. Gallicchio, A. Micheli, L. Pedrelli.
        Fast Spectral Radius Initialization for Recurrent
        Neural Networks. (2020)

    Examples:
    ---------
        >>> from reservoirpy.mat_gen import fast_spectral_initialization
        >>> W = fast_spectral_initialization(5, proba=0.5, seed=42)
        >>> W.toarray()
        array([[ 0.13610996, -0.57035192,  0.        ,  0.        ,  0.        ],
               [ 0.26727631,  0.        , -0.22370579, -0.04951286,  0.        ],
               [ 0.60065244,  0.        , -0.02846888,  0.        ,  0.08786003],
               [ 0.        ,  0.39151551,  0.4157107 ,  0.        ,  0.41754172],
               [ 0.        ,  0.72101228,  0.        ,  0.        ,  0.        ]])
    """
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    rs = RandomState(seed)

    if spectral_radius is None or proba == 0.:
        a = 1
    else:
        a = -(6 * spectral_radius) / (np.sqrt(12) * np.sqrt((proba * N)))

    if proba < 1:
        return sparse.random(N, N, density=proba,
                             random_state=rs, format=sparsity_type,
                             data_rvs=lambda s: rs.uniform(a, -a, size=s))

    else:
        return np.random.uniform(a, -a, size=(N, N))


def generate_internal_weights(N: int,
                              spectral_radius: float = None,
                              proba: float = 0.1,
                              Wstd: float = 1.0,
                              sparsity_type: str = None,
                              seed: int = None,
                              verbose: bool = False,
                              typefloat=np.float64):
    """Method that generate the weight matrix that will be used
    for the internal connections of the reservoir.

    Weights will follow a normal distribution of mean 0 and
    scale Wstd (by default 1), and can then be rescale to


    Parameters
    ----------
    N : int
        Number of reservoir units
    spectral_radius : float, optional
        Maximum desired eigenvalue of the
        reservoir weights matrix, by default None
    proba : float, optional
        Probability of non zero connection,
        density of the weight matrix, by default 0.1
    Wstd : float, optional
        Standard deviation of internal weights, by default 1.0
    sparsity_type : str, optional
        If set ot one of the scipy.sparse matrix format,
        will generate internal weights matrix in a sparse
        format, by default None
    seed : int, optional
        Random state generator seed, for reproducibility,
        by default None
    verbose : bool, optional
    typefloat : [type], optional

    Returns
    -------
    np.ndarray or scipy.sparse matrix
        A reservoir weights matrix.

    Raises
    ------
    ValueError
        Invalid non zero connection probability.
    """
    if not _is_probability(proba):
        raise ValueError(f"proba = {proba} not in [0; 1].")

    rs = RandomState(seed)

    # TODO: put this in a function
    if sparsity_type is not None:
        w = sparse.random(N, N, density=proba, format=sparsity_type,
                          random_state=rs,
                          data_rvs=lambda s: rs.normal(0, Wstd, size=s))
        rhoW = max(abs(eigs(w, k=1, which='LM', return_eigenvectors=False)))
    else:
        mask = 1 * (rs.rand(N, N) < proba)
        mat = rs.normal(0, Wstd, (N, N))
        w = np.multiply(mat, mask, dtype=typefloat)

        # Computing the spectral radius of W matrix
        rhoW = max(abs(linalg.eig(w)[0]))

    # TODO: use logging
    if verbose:
        print("Spectra radius of generated matrix before "
              "applying another spectral radius: " + str(rhoW))

    if spectral_radius is not None:
        w *= spectral_radius / rhoW
        if verbose:
            if sparse.issparse(w):
                rhoW_after = max(abs(sparse.linalg.eigs(w,
                                                        k=1,
                                                        which='LM',
                                                        return_eigenvectors=False)))
            else:
                rhoW_after = max(abs(linalg.eig(w)[0]))
            print("Spectra radius matrix after applying "
                  "another spectral radius: " + str(rhoW_after))

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

    rs = RandomState(seed)

    if input_bias:
        dim_input += 1
    mask = 1 * (rs.rand(nbr_neuron, dim_input) < proba)
    mat = rs.randint(0, 2, (nbr_neuron, dim_input)) * 2 - 1
    w = np.multiply(mat, mask, dtype=typefloat)
    if input_scaling is not None:
        w = input_scaling * w

    # ? remove this in future versions ?
    if randomize_seed_afterwards:  # pragma: no cover
        warnings.warn("Have to check if you really want to randomize the seed, "
                      "because this makes the whole experiment not reproducible.",
                      UserWarning)
        np.random.seed(int(time.time()*10**6))

    return w


def _is_probability(proba):
    return 1. - proba >= 0. and proba >= 0.
