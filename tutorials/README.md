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
feedback connections, deep ESNs, custom weight matrices... for RC enthusiasts.

3. **[General Introduction to Reservoir Computing](./3-General_Introduction_to_Reservoir_Computing.ipynb)**

An easy way to discover Reservoir Computing in general, with nice visualizations in a Jupyter notebook.

- First part gives a good overview of ReservoirPy main functionnalities, with online and offline learning rules applied to ESNs
with chaotic timeseries.

- Second part is composed of two short demos of ESNs applied to "real life" tasks, one concerning robotics and the other
concerning audio annotation for birdsongs.

4. **[Understand and optimize hyperparameters](4-Understand_and_optimize_hyperparameters.ipynb)**

A gentle introduction to some important hyperparameters defining Reservoir Computing architectures,
followed by a tutorial on how to combine *hyperopt* with ReservoirPy to find the bests parameters for a model.

An introduction to hyper-parameter search using optuna, and parallel search can be found here: **[Hyperparameter search with Optuna](./4.a-Hyperparameter%20search%20with%20Optuna/)**

5. **[Classification with Reservoir Computing](5-Classification-with-RC.ipynb)**

A simple example of classification task using Reservoir Computing: the Japanese vowels dataset.
The notebook describes two simple model (sequence-to-sequence and sequence-to-vector) able to solve
this task.

6. **[Interfacing with scikit-learn](6-Interfacing_with_scikit-learn.ipynb)**

A guide to use the ScikitLearnNode, an interface to integrate any scikit-learn models to
your ReservoirPy architecture, including classification models. It also includes an example to speed-up the optimisation of ridge parameter search with *Ridge-CV*.


## Getting started

Each tutorial may have its own dependencies, and therefore require some installation.
In general, you will need:
- ReservoirPy,
- Matplotlib for visualization
- Pandas, to work with timeseries and high dimensional data
- Jupyter, to benefits from the power of Jupyter notebooks
- scikit-learn, to have metrics from scikit-learn
- Hyperopt, to optimise hyperparameters.

Everything is in the requirements file.


**Dependencies:**
```txt
hyperopt
jupyter
matplotlib
pandas
requests
reservoirpy
scikit-learn
seaborn
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
