# ReservoirPy examples and tutorials

Here, you can find everything you need to dive into Reservoir Computing with simple tutorials written
with ReservoirPy API.

## Summary

**Installation**

See below the "Getting started" section.

1. **[Getting Started](./1-Getting_Started.ipynb)**

A short notebook to discover the basics of ReservoirPy, for beginners.

2. **[Advanced features](./2-Advanced_Features.ipynb)**

A notebook presenting more advanced manipulations with ESNs like parallelization,
feedback connections, deep ESNs, custom weigth matrices... for RC enthusiasts.

3. **[General Introduction to Reservoir Computing](./3-General_Introduction_to_Reservoir_Computing.ipynb)**

An easy way to discover Reservoir Computing in general, with nice visualizations in a Jupyter notebook.

- First part gives a good overview of ReservoirPy main functionnalities, with online and offline learning rules applied to ESNs
with choatic timeseries.

- Second part is composed of two short demos of ESNs applied to "real life" tasks, one concerning robotics and the other
concerning audio annotation for birdsongs.

2. **[Understand and optimize hyperparameters](4-Understand_and_optimize_hyperparameters.ipynb)**

A gentle introduction to some important hyperparameters defining Reservoir Computing architectures,
followed by a tutorial on how to combine *hyperopt* with ReservoirPy to find the bests parameters for a model.

4. **[Simple example on chaotic timeseries prediction](Simple%20Examples%20with%20Mackey-Glass)**

Example based on Mantas Lukoševičius's minimal example of MackeyGlass prediction and generation.
The directory includes both the minimal ESN of Mantas (not using ReservoiPy) and the equivalent simple example
using ReservoirPy. Kept for teaching purpose.

## Getting started

Each tutorial may have its own dependencies, and therefore require some installation.
In general, you will need :
- ReservoirPy,
- Matplotlib and Seaborn, for visualization
- Pandas, to work with timeseries and high dimensional data
- Jupyter, to benefits from the power of Jupyter notebooks
- scikit-learn, to have metrics from scikit-learn
- Hyperopt, to optimise hyperparameters.
-
Everything is in the requirements file.

```bash
pip install -r tutorials/requirements.txt
```

### Opening the notebook

Using Jupyter is recommended to follow this tutorial. You can install it using:

```bash
pip install jupyter
```

Then, from within your virtual environment where Jupyter is installed, use:

```bash
jupyter notebook
```
at the root of the directory containing the Jupyter notebook (.ipynb file extension).
