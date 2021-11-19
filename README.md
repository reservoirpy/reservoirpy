![ReservoirPy banner](static/rpy_banner_bw.png)

[![PyPI version](https://badge.fury.io/py/reservoirpy.svg)](https://badge.fury.io/py/reservoirpy)
[![Documentation Status](https://readthedocs.org/projects/reservoirpy/badge/?version=latest)](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/reservoirpy/reservoirpy.svg?branch=master)](https://travis-ci.org/reservoirpy/reservoirpy)


# ReservoirPy (v0.3.0-**beta**) 🌀 🧠
**Simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**


```python
from reservoirpy.nodes import Reservoir, Ridge, Input

data      = Input(input_dim=1)
reservoir = Reservoir(100, lr=0.1, sr=0.99)
readout   = Ridge(1, ridge=1e-3)

esn = data >> reservoir >> readout

forecast = esn.fit(X, y).run(timeseries)
```


ReservoirPy is a simple user-friendly library based on Python scientific modules. It provides a flexible interface to implement efficient Reservoir Computing (RC) architectures with a particular focus on Echo State Networks (ESN). Advanced features of ReservoirPy allow to improve computation time efficiency on a simple laptop compared to basic Python implementation. Some of its features are: offline and online training, parallel implementation, sparse matrix computation, fast spectral initialization, etc. Moreover, graphical tools are included to easily explore hyperparameters with the help of the hyperopt library.

This library works for Python 3.8 and higher.

## Offcial documentation 📖

See [the official ReservoirPy's documentation](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
to learn more about the main features of ReservoirPy, its API and the installation process.

## Examples and tutorials 🎓

[Go to the tutorial folder](./tutorials/) for tutorials in Jupyter Notebooks.

[Go to the examples folder](./examples/) for examples and papers with codes, also in Jupyter Notebooks.


## Quick try ⚡
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
A quick tutorial on how to explore hyperparameters with ReservoirPy and Hyperopt can be found in this [preprint (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

Take a look at our **advices and general method to explore hyperparameters** for reservoirs in our recent paper:
"Which Hype for My New Task? Hints and Random Search for Echo State Networks Hyperparameters", ICANN 2021
[HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [Preprint](https://hal.inria.fr/hal-03203318/)

[Turorial and Jupyter Notebook for hyper-parameter exploration](./examples/Optimization%20of%20hyperparameters)

More info on hyperopt: [Official website](http://hyperopt.github.io/hyperopt/)

## Cite
Nathan Trouvain, Luca Pedrelli, Thanh Trung Dinh, Xavier Hinaut. ReservoirPy: an Efficient and User-Friendly Library to Design Echo State Networks. 2020. ⟨hal-02595026⟩ https://hal.inria.fr/hal-02595026
