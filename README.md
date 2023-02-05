<div align="center">
  <!-- <img src="https://github.com/reservoirpy/reservoirpy/raw/master/static/rpy_banner_bw.png"><br> !-->
  <img src="./static/rpy_banner_bw_small-size.jpg"><br>
</div>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/reservoirpy)
[![PyPI version](https://badge.fury.io/py/reservoirpy.svg)](https://badge.fury.io/py/reservoirpy)
[![Documentation Status](https://readthedocs.org/projects/reservoirpy/badge/?version=latest)](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
[![Testing](https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/reservoirpy/reservoirpy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/reservoirpy/reservoirpy/branch/master/graph/badge.svg?token=JC8R1PB5EO)](https://codecov.io/gh/reservoirpy/reservoirpy)

# ReservoirPy (v0.3.6) 🌀🧠
**Simple and flexible code for Reservoir Computing architectures like Echo State Networks (ESN).**

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/reservoirpy/reservoirpy/HEAD)

```python
from reservoirpy.nodes import Reservoir, Ridge, Input

data = Input(input_dim=1)
reservoir = Reservoir(100, lr=0.3, sr=1.1)
readout = Ridge(ridge=1e-6)

esn = data >> reservoir >> readout

forecast = esn.fit(X, y).run(timeseries)
```

ReservoirPy is a simple user-friendly library based on Python scientific modules.
It provides a **flexible interface to implement efficient Reservoir Computing** (RC)
architectures with a particular focus on *Echo State Networks* (ESN).
Advanced features of ReservoirPy allow to improve computation time efficiency
on a simple laptop compared to basic Python implementation, with datasets of
any size.

Some of its features are: **offline and online training**, **parallel implementation**,
**sparse matrix computation**, fast spectral initialization, **advanced learning rules**
(e.g. *Intrinsic Plasticity*) etc. It also makes possible
to **easily create complex architectures with multiple reservoirs** (e.g. *deep reservoirs*),
readouts, and **complex feedback loops**.
Moreover, graphical tools are included to **easily explore hyperparameters**
with the help of the *hyperopt* library.
Finally, it includes several tutorials exploring exotic architectures
and examples of scientific papers reproduction.

This library works for **Python 3.8** and higher.

[Follow @reservoirpy](https://twitter.com/reservoirpy) updates and new releases on Twitter.

## Official documentation 📖

See [the official ReservoirPy's documentation](https://reservoirpy.readthedocs.io/en/latest/?badge=latest)
to learn more about the main features of ReservoirPy, its API and the installation process. Or you can access directly the [User Guide with tutorials](https://reservoirpy.readthedocs.io/en/latest/user_guide/index.html#user-guide).

## Quick example of how to code a deep reservoir
![Image](deep-reservoir.gif)

## Installation

```bash
pip install reservoirpy
```

(See below for more advanced installation options)


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
inputs in a high-dimensional (non-linear) space, and a *readout*, a simple
feed-forward layer of neurons in charge with *reading-out* the desired output from
the activations of the reservoir.
```python
from reservoirpy.nodes import Reservoir, Ridge

reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(output_dim=1, ridge=1e-5)
```

We here obtain a reservoir with 100 neurons, a *spectral radius* of 1.25 and
a *leak rate* of 0.3 (you can learn more about these hyperparameters going through
the tutorial
[Understand and optimize hyperparameters](./tutorials/4-Understand_and_optimize_hyperparameters.ipynb)).
Here, our readout layer is just a single unit, that we will receive connections from (all units of) the reservoir.
Note that only the readout layer connections are trained.
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

We train our ESN on the first 500 timesteps of the timeseries, with 100 steps used to warm up the reservoir states.

```python
esn.fit(X[:500], X[1:501], warmup=100)
```

Our ESN is now trained and ready to use. Let's run it on the remainder of the timeseries:

```python
predictions = esn.run(X[501:-1])
```

As a shortcut, both operations can be performed in just one line!

```python
predictions = esn.fit(X[:500], X[1:501]).run(X[501:-1])
```

Let's now evaluate its performances.

**Step 4: Evaluate the ESN**

```python
from reservoirpy.observables import rmse, rsquare

print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))
```

Run and analyse this simple file (in the "tutorials/Simple Examples with Mackey-Glass" folder) to see a complete example of timeseries prediction with ESNs:
- simple_example_MackeyGlass.py (using the ESN class)

    ```bash
    python simple_example_MackeyGlass.py
    ```

If you have some issues testing some examples, have a look at the [extended packages requirements in readthedocs](https://reservoirpy.readthedocs.io/en/latest/developer_guide/advanced_install.html?highlight=requirements#additional-dependencies-and-requirements).

## More installation options 

To install it, use one of the following command:

```bash
pip install reservoirpy
```
or

```bash
pip install reservoirpy==0.3.5
```

If you want to run the Python Notebooks of the _tutorials_ folder, install the packages in requirements file (warning: this may downgrade the version of hyperopt installed):
```bash
pip install -r tutorials/requirements.txt
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

## More examples and tutorials 🎓

[Go to the tutorial folder](./tutorials/) for tutorials in Jupyter Notebooks.

[Go to the examples folder](./examples/) for examples and papers with codes, also in Jupyter Notebooks.

## Paper with tutorials
Tutorial for ReservoirPy (v0.2) can be found in this [Paper (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

## Explore Hyper-Parameters with Hyperopt
A quick tutorial on how to explore hyperparameters with ReservoirPy and Hyperopt can be found in this [paper (Trouvain et al. 2020)](https://hal.inria.fr/hal-02595026).

Take a look at our **advices and our method to explore hyperparameters** for reservoirs in our [recent paper: (Hinaut et al 2021)](https://hal.inria.fr/hal-03203318/) [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [HAL](https://hal.inria.fr/hal-03203318)

[Turorial and Jupyter Notebook for hyper-parameter exploration](./tutorials/4-Understand_and_optimize_hyperparameters.ipynb)

More info on hyperopt: [Official website](http://hyperopt.github.io/hyperopt/)

## Papers and projects using ReservoirPy

If you want your paper to appear here, please contact us (see contact link below).

- Chaix-Eichel et al. (2022) From implicit learning to explicit representations. arXiv preprint arXiv:2204.02484. [arXiv](https://arxiv.org/abs/2204.02484) [PDF](https://arxiv.org/pdf/2204.02484)
- Trouvain & Hinaut (2021) Canary Song Decoder: Transduction and Implicit Segmentation with ESNs and LTSMs. ICANN 2021 [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_6) [HAL](https://hal.inria.fr/hal-03203374) [PDF](https://hal.inria.fr/hal-03203374/document)
- Pagliarini et al. (2021) Canary Vocal Sensorimotor Model with RNN Decoder and Low-dimensional GAN Generator. ICDL 2021. [HTML](https://ieeexplore.ieee.org/abstract/document/9515607?casa_token=QbpNhxjtfFQAAAAA:3klJ9jDfA0EEbckAdPFeyfIwQf5qEicaKS-U94aIIqf2q5xkX74gWJcm3w9zxYy9SYOC49mQt6vF)
- Pagliarini et al. (2021) What does the Canary Say? Low-Dimensional GAN Applied to Birdsong. HAL preprint. [HAL](https://hal.inria.fr/hal-03244723/) [PDF](https://hal.inria.fr/hal-03244723/document)
- Which Hype for My New Task? Hints and Random Search for Echo State Networks Hyperparameters. ICANN 2021 [HTML](https://link.springer.com/chapter/10.1007/978-3-030-86383-8_7) [HAL](https://hal.inria.fr/hal-03203318) [PDF](https://hal.inria.fr/hal-03203318)

## Contact
If you have a question regarding the library, please open an Issue. If you have more general question or feedback you can [contact us on twitter](https://twitter.com/reservoirpy) or by email to xavier dot hinaut the-famous-home-symbol inria dot fr.

## Citing ReservoirPy

Trouvain, N., Pedrelli, L., Dinh, T. T., Hinaut, X. (2020) Reservoirpy: an efficient and user-friendly library to design echo state networks. In International Conference on Artificial Neural Networks (pp. 494-505). Springer, Cham. [HTML](https://link.springer.com/chapter/10.1007/978-3-030-61616-8_40) [HAL](https://hal.inria.fr/hal-02595026) [PDF](https://hal.inria.fr/hal-02595026/document)

If you're using ReservoirPy in your work, please cite our package using the following bibtex entry:

```

@incollection{Trouvain2020,
  doi = {10.1007/978-3-030-61616-8_40},
  url = {https://doi.org/10.1007/978-3-030-61616-8_40},
  year = {2020},
  publisher = {Springer International Publishing},
  pages = {494--505},
  author = {Nathan Trouvain and Luca Pedrelli and Thanh Trung Dinh and Xavier Hinaut},
  title = {{ReservoirPy}: An Efficient and User-Friendly Library to Design Echo State Networks},
  booktitle = {Artificial Neural Networks and Machine Learning {\textendash} {ICANN} 2020}
}
```

<div align="left">
  <img src="./static/inr_logo_rouge.jpg" width=300><br>
</div>


This package is developped and supported by Inria at Bordeaux, France in [Mnemosyne](https://team.inria.fr/mnemosyne/) group. [Inria](https://www.inria.fr/en) is a French Research Institute in Digital Sciences (Computer Science, Mathematics, Robotics, ...).


