
# Introduction to Reservoir Computing

This example covers basic usage of Reservoir Computing with ReservoirPy for simple timeseries prédiction and generation.

## Summary

In this example, you will:

- learn how to make accurate prediction of a chaotic timeseries, the Mackey-Glass timeseries, using Echo State Networks (ESN);
- learn how to use ESN to artificially generate chaotic timeseries;
- dive into the hyperparemeters of an ESN, and understand their roles;
- apply these techniques on real data from the real world.

By the end of this short tutorial, you will master:
- the `ESN` and `mat_gen` APIs;
- the `dataset` module, containing toy chaotic timeseries.

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

You will also need Pandas, Matplotlib and requests:

```bash
pip install pandas matplotlib requests
```

Of course, you will also need to install ReservoirPy.

```bash
pip install reservoirpy==0.2.4
```
