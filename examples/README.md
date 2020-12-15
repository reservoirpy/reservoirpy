
# ReservoirPy examples and tutorials

Here, you can find everything you need to dive into Reservoir Computing with simple examples written with ReservoirPy API.

## Summary

1. **Introduction to Reservoir Computing**

An easy way to discover Reservoir Computing, with nice visualizations on a Jupyter notebook. Gives an extensive presentation of ReservoirPy main functionnalities.

2. **Optimization of hyperparameters**

How to combine *hyperopt* with ReservoirPy to find the bests parameters for a model.

3. **Online learning**

A presentation of some other learning rules that can be appliyed to ESN, allowing to perform things like online learning.

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
