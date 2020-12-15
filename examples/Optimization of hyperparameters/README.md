
# Introduction to optimization of hyperparameters with *hyperopt* and ReservoirPy

This example covers usage of [*hyperopt*](http://hyperopt.github.io/hyperopt/) and ReservoirPy for optimization of ESN hyperparameters.

## Summary

In this example, you will:

- learn how to use ReservoirPy to help you manipulate *hyperopt* API;
- apply these tools to estimate the most relevant hyperparameters for a task;
- apply these techniques on real data from the real world.

By the end of this short tutorial, you will master:
- the `hyper` module, containing tools to manipulate *hyperopt*.

## Getting started

Using Jupyter is recommended to follow this tutorial. You can install it using:

```bash
pip install jupyter
```

Then, from within your virtual environment where Jupyter is installed, use:

```bash
jupyter notebook
```
at the root of the directory containing the Jupyter notebook (.ipynb file extension).

You will also need any recent version of hyperopt, seaborn, Pandas, Matplotlib and requests:

```bash
pip install hyperopt seaborn pandas matplotlib requests
```

Of course, you will also need to install ReservoirPy. Since there is not yet a stable release of ReservoirPy on PyPi, we still recommend cloning the respository and installing it in editable mode.

```bash
git clone https://github.com/neuronalx/reservoirpy.git
pip install -e ./reservoirpy
```