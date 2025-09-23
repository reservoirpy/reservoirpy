:orphan:

.. _advanced_install:

===========================
Advanced installation guide
===========================

This page will guide you into installing ReservoirPy on your system.

Before any package installation, make sure that you have a compatible Python distribution already installed
on your computer. **ReservoirPy is meant to be used only with Python 3.9 and higher**.

To check the version of your Python distribution, you can run the following command in a terminal,
in Linux/MacOS/Windows :

.. code-block::

    python --version

When performing the installation of ReservoirPy and all its dependencies, we also recommend using a
virtual environment to avoid any unintended interactions with the dependencies that are already installed
on your system. To learn more about virtual environment, you can check `Python documentation on virtual
environments and packages <https://docs.python.org/3/tutorial/venv.html>`_, or the documentation of the
`conda environment manager <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
if you are using Anaconda.


Installation using `pip`
========================

ReservoirPy package is hosted by `Pypi <https://pypi.org/project/reservoirpy/>`_ and can
therefore be installed using `pip` on Linux/MacOS/Windows (in the latter case, having an
`Anaconda <https://www.anaconda.com/products/individual>`_ distribution installed
on your computer may be necessary).

To install ReservoirPy using `pip`, simply run the following command in a terminal:

.. code-block:: bash

    pip install reservoirpy


To check your installation of ReservoirPy, run one of the following:

.. code-block:: bash

    pip show reservoirpy
    python -c "from reservoirpy import __version__; print(__version__)"


Installation using the source code
==================================

You can find the source code of ReservoirPy on `GitHub (https://github.com/reservoirpy/reservoirpy)
<https://github.com/reservoirpy/reservoirpy>`_.

Download the latest version on the ``master`` branch, or any other branch you would like
to install (``dev`` branch or older versions branches). You can also fork the project from
GitHub.

Then, unzip the project (or clone the forked repository). You can then install ReservoirPy in
editable mode using `pip` :

.. code-block::

    pip install -e /path/to/reservoirpy


Additional dependencies and requirements
========================================

**Hyperoptimization and visualization tools**

All basic dependencies of ReservoirPy should be installed when using `pip` as package manager.

Although, to use the hyperoptimization and visualization tools from the :py:mod:`reservoirpy.hyper` module, you will need to install a few
more dependencies in your virtual environment, namely `hyperopt`, `matplotlib` and `seaborn`. You can do so using the `hyper` extra dependencies:

.. code-block::

    pip install reservoirpy[hyper]

**ScikitLearnNode**

You can use some of scikit-learn's linear models through the use of the the ScikitLearnNode.

In the same manner, you can install scikit-learn with the appropriate version using:

.. code-block::

    pip install "reservoirpy[sklearn]"

**Jax**

If you want to use the ``reservoirpy.jax`` submodule:

.. code-block::

    pip install "reservoirpy[jax]"

**Development tools**

ReservoirPy use `pytest` as test framework, and `flake8` as linter.
If you want to contribute to ReservoirPy, you should have the following
additional dependencies installed:

.. code-block::

    pip install pytest pytest-cov flake8

**All dependencies**

A summary of all dependencies and their purpose in ReservoirPy
can be found in the table below:

.. list-table:: All dependencies
    :widths: 50 25 50
    :header-rows: 1

    * - Dependency
      - Version
      - Purpose
    * - numpy
      - 1.18.1
      - build, install
    * - scipy
      - 1.4.1
      - build, install
    * - joblib
      - 0.14.1
      - build, install
    * - hyperopt
      - 0.2.5
      - reservoirpy.hyper, examples
    * - matplotlib
      - 3.3.3
      - reservoirpy.hyper, examples
    * - tqdm
      - 4.43.0
      - reservoirpy.hyper, docs
    * - pytest
      - 6.1.2
      - tests
    * - pytest-cov
      - 2.10.1
      - tests
    * - scikit-learn
      - 0.24.1
      - tests
    * - sphinx
      - 7.2.6
      - docs
    * - pydata-sphinx-theme
      - 0.13.3
      - docs
    * - sphinx-copybutton
      - 0.5.2
      - docs
    * - ipython
      - 7.31.1
      - docs
    * - nbsphinx
      - 0.8.7
      - docs
    * - sphinx-design
      - 0.5.0
      - docs
