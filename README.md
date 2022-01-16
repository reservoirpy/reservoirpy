<div align="center">
  <img src="https://github.com/reservoirpy/reservoirpy/raw/master/static/rpy_banner_bw.png"><br>
</div>

[![PyPI version](https://badge.fury.io/py/reservoirpy.svg)](https://badge.fury.io/py/reservoirpy)
[![Documentation Status](https://readthedocs.org/projects/reservoirpy/badge/?version=latest)](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/reservoirpy/reservoirpy.svg?branch=master)](https://travis-ci.org/reservoirpy/reservoirpy)


# ReservoirPy (v0.3.0) 🌀🧠
**Simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**


```python
from reservoirpy.nodes import Reservoir, Ridge, Input

data      = Input(input_dim=1)
reservoir = Reservoir(100, lr=0.3, sr=1.1)
readout   = Ridge(1, ridge=1e-6)

esn = data >> reservoir >> readout

forecast = esn.fit(X, y).run(timeseries)
```

ReservoirPy is a simple user-friendly library based on Python scientific modules. 
It provides a flexible interface to implement efficient Reservoir Computing (RC) 
architectures with a particular focus on Echo State Networks (ESN). 
Advanced features of ReservoirPy allow to improve computation time efficiency 
on a simple laptop compared to basic Python implementation. 
Some of its features are: offline and online training, parallel implementation, 
sparse matrix computation, fast spectral initialization, etc. 
Moreover, graphical tools are included to easily explore hyperparameters 
with the help of the hyperopt library.

This library works for Python 3.8 and higher.

## Offcial documentation 📖

See [the official ReservoirPy's documentation](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
to learn more about the main features of ReservoirPy, its API and the installation process.

## Installation

To install it, use one of the following command:

```bash
pip install reservoirpy
```

or 

```bash
pip install reservoirpy==0.3.0
```

If you want to use the previous version 0.2.4, you can install ReservoirPy using:

```bash
pip install reservoirpy==0.2.4
```

If you want to enable the `hyper` package and its hyperparameter optimization helpers using
[hyperopt](http://hyperopt.github.io/hyperopt/), use:

```bash
pip install reservoirpy[hyper]
```

## Quick try ⚡

### An example on Chaotic timeseries prediction (MackeyGlass)

**Step 1: Load the dataset**

ReservoirPy comes with some handy data generator able to create synthetic timeseries
for well-known tasks such as Mackey-Glass timeseries forecasting.

```python
from reservoirpy.datasets import mackey_glass

X = mackey_glass(n_timesteps=2000)
```

**Step 2: Create an Echo State Network...**

...or any kind of model you wish to use to solve your task. In this simple
use case, we will try out Echo State Networks (ESNs), one of the 
most minimal architecture of Reservoir Computing machines.

An ESN is made of 
a *reservoir*, a random recurrent network used to encode our 
inputs in a "close-to-chaos" high dimensional space, and a *readout*, a simple
feed-forward layer of neurons in charge with *reading-out* the desired output from
the activations of the reservoir. 
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout   = Ridge(output_dim=1, ridge=1e-5)
```

We here obtain a reservoir with 100 neurons, a *spectral radius* of 1.25 and 
a *leak rate* of 0.3 (you can learn more about these hyperparameters going through
the tutorial 
[Introduction to Reservoir Computing](./tutorials/Introduction%20%20to%20Reservoir%20Computing)).
Our readout is just a layer of one single neuron, that we will next connect to the
reservoir neurons. Note that only the readout layer connections are trained! 
This is one of the cornerstone of all Reservoir Computing techniques. In our
case, we will train these connections using linear regression, with a regularization
coefficient of 10<sup>-5</sup>.

Now, let's connect everything using the `>>` operator. 

```python
esn = reservoir >> readout
```

That's it! Next step: fit the readout weights to perform the task we want.
We will train the ESN to make one-step-ahead forecasts of our timeseries.

**Step 3: Fit and run the ESN**

```python
predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])
```

Our ESN is now trained and ready to use. Let's evaluate its performances:

**Step 4: Evaluate the ESN**

```python
from reservoirpy.observables import rmse, rsquare
print("RMSE:", rmse(X[502:], predictions), 
      "R^2 score:", rsquare(X[502:], predictions))
```

Run and analyse these two files (in the "tutorials/Simple Examples with Mackey-Glass" folder) to see how to make timeseries prediction with Echo State Networks:
- simple_example_MackeyGlass.py (using the ESN class)

    ```bash
    python simple_example_MackeyGlass.py
    ```

- minimalESN_MackeyGlass.py (without the ESN class)

    ```bash
    python minimalESN_MackeyGlass.py
    ```
  
If you have some issues testing some examples, have a look at the [extended packages requirements in readthedocs](https://reservoirpy.readthedocs.io/en/latest/installation.html#additional-dependencies-and-requirements).
  
## Examples and tutorials 🎓

[Go to the tutorial folder](./tutorials/) for tutorials in Jupyter Notebooks.

[Go to the examples folder](./examples/) for examples and papers with codes, also in Jupyter Notebooks.

## Paper with tutorials
Tutorial on ReservoirPy can be found in this [Paper (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

## Explore Hyper-Parameters with Hyperopt
A quick tutorial on how to explore hyperparameters with ReservoirPy and Hyperopt can be found in this [paper (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

Take a look at our **advices and general method to explore hyperparameters** for reservoirs in our [recent paper: (Hinaut et al 2021)](https://hal.inria.fr/hal-03203318/) [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [HAL](https://hal.inria.fr/hal-03203318)

[Turorial and Jupyter Notebook for hyper-parameter exploration](./examples/Optimization%20of%20hyperparameters)

More info on hyperopt: [Official website](http://hyperopt.github.io/hyperopt/)

## Papers and projects using ReservoirPy
- Trouvain & Hinaut (2021) Canary Song Decoder: Transduction and Implicit Segmentation with ESNs and LTSMs. ICANN 2021 [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_6) [HAL](https://hal.inria.fr/hal-03203374) [PDF](https://hal.inria.fr/hal-03203374/document)
- Pagliarini et al. (2021) Canary Vocal Sensorimotor Model with RNN Decoder and Low-dimensional GAN Generator. ICDL 2021. [HTML](https://ieeexplore.ieee.org/abstract/document/9515607?casa_token=QbpNhxjtfFQAAAAA:3klJ9jDfA0EEbckAdPFeyfIwQf5qEicaKS-U94aIIqf2q5xkX74gWJcm3w9zxYy9SYOC49mQt6vF) 
- Pagliarini et al. (2021) What does the Canary Say? Low-Dimensional GAN Applied to Birdsong. HAL preprint. [HAL](https://hal.inria.fr/hal-03244723/) [PDF](https://hal.inria.fr/hal-03244723/document)
- Which Hype for My New Task? Hints and Random Search for Echo State Networks Hyperparameters. ICANN 2021 [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [HAL](https://hal.inria.fr/hal-03203318) [PDF](https://hal.inria.fr/hal-03203318)



## Cite
Trouvain, N., Pedrelli, L., Dinh, T. T., Hinaut, X. (2020) Reservoirpy: an efficient and user-friendly library to design echo state networks. In International Conference on Artificial Neural Networks (pp. 494-505). Springer, Cham. [HTML](https://link.springer.com/chapter/10.1007/978-3-030-61616-8_40) [HAL](https://hal.inria.fr/hal-02595026) [PDF](https://hal.inria.fr/hal-02595026/document)
