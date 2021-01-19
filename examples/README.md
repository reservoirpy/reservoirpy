
# ReservoirPy examples and tutorials

Here, you can find everything you need to dive into Reservoir Computing with simple examples written with ReservoirPy API.

## Summary

0. **Installation**

See below the "Getting started" section.

1. **[Introduction to Reservoir Computing](./Introduction%20%20to%20Reservoir%20Computing)**

An easy way to discover Reservoir Computing, with nice visualizations on a Jupyter notebook. Gives an extensive presentation of ReservoirPy main functionnalities.

2. **[Optimization of hyperparameters](Optimization%20of%20hyperparameters)**

How to combine *hyperopt* with ReservoirPy to find the bests parameters for a model.

3. **[Online learning](Online%20learning)**

A presentation of some other learning rules that can be appliyed to ESN, allowing to perform things like online learning.

4. **[Simple example on chaotic timeseries prediction](Simple%20Examples%20with%20Mackey-Glass)**

Example based on Mantas Lukoševičius's minimal example of MackeyGlass prediction and generation. The directory includes both the minimal ESN of Mantas (not using ReservoiPy) and the equivalent simple example using ReservoirPy.

## Getting started

Each tutorial may have its own dependencies, and therefore require some installation.
In general, you will need :
- Matplotlib and Seaborn, for visualization
- Pandas, to work with timeseries and high dimensional data
- Jupyter, to benefits from the power of Jupyter notebooks
- Sklearn, to have metrics from scikit-learn
- Hyperopt, to optimise hyperparameters

```bash
pip install matplotlib seaborn pandas jupyter sklearn hyperopt
```

And of course, you will need to install ReservoirPy. As there is no stable realease of the library on PyPi yet, we advise to clone this repostiory and perform an editable installation with `pip` to install ReservoirPy:

```bash
git clone https://github.com/neuronalx/reservoirpy.git
pip install -e ./reservoirpy
```
