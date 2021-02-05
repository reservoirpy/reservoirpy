# ReservoirPy (v0.2)
**A simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

[![PyPI version](https://badge.fury.io/py/reservoirpy.svg)](https://badge.fury.io/py/reservoirpy)
[![Documentation Status](https://readthedocs.org/projects/reservoirpy/badge/?version=latest)](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/reservoirpy/reservoirpy.svg?branch=master)](https://travis-ci.org/reservoirpy/reservoirpy)

ReservoirPy is a simple user-friendly library based on Python scientific modules. It provides a flexible interface to implement efficient Reservoir Computing (RC) architectures with a particular focus on Echo State Networks (ESN). Advanced features of ReservoirPy allow to improve computation time efficiency on a simple laptop compared to basic Python implementation. Some of its features are: offline and online training, parallel implementation, sparse matrix computation, fast spectral initialization, etc. Moreover, graphical tools are included to easily explore hyperparameters with the help of the hyperopt library.

This library works for Python 3.6 and higher.

## Installation, examples and tutorials

[Go to the examples folder](./examples/) for intallation, examples, tutorials and Jupyter Notebooks.

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

## Explore Hyper-Parameters with Hyperopt
Tutorial on how to explore hyperparameters with ReservoirPy and Hyperopt can be found in this [preprint (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

[Turorial and Jupyter Notebook for hyper-parameter exploration](./examples/Optimization%20of%20hyperparameters)

More info on hyperopt: [Official website](http://hyperopt.github.io/hyperopt/)

## Cite
Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
