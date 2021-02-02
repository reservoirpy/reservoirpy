===============
Getting started
===============

Let's have a look at ReservoirPy's main functionnalities and features.

Before getting started to use ReservoirPy, make sure you have correctly installed
the package and all its dependencies by looking at the :ref:`installation guide`.

ReservoirPy is an open-source project which aims to develop a simple yet performant
*reservoir computing* tool, written in scientific Python and without using any
heavy machine learning library.

To learn more about the theoretical aspects of reservoir computing, you can read the
page :ref:`whats rc`.

Back to basics: scientific Python and linear regressions
========================================================

Most of machine learning work these days is becoming more and more complicated.
To deal with this increasing complexity, a lot of open source projects have
blossomed, and allow one to go from very simple models to over-complicated
architectures.

This is not the philosophy of reservoir computing, and thus, not the philosophy
of ReservoirPy.

Because reservoir computing aims at making complexity emerge from apparent simplicity,
ReservoirPy provides its users with very simple tools that can achieve a wide range
of machine learning tasks, in particular when it comes to deal with sequential data.

These tools are all based on `NumPy <https://numpy.org/>`_ and `SciPy <https://www.scipy.org/>`_,
the Python *basic* scientific libraries, and can therefore be mastered by any Python enthusiast,
from beginners to experts.

Here is how, with only a few NumPy arrays, you can define a complete dummy Echo State Network:

.. doctest::

    >>> import numpy as np
    >>> from reservoirpy import ESN
    >>> Win = np.array([[1, -1], [-1, 1], [1, -1]])
    >>> W = np.array([[0.0, 0.1, 0.1],
    ...               [0.5, 0., 0.0 ],
    ...               [0.0, 0.2, 0.3]])
    >>> esn = ESN(lr=0.1, W=W, Win=Win)
    >>> esn
    ESN(trained=False, feedback=False, N=3, lr=0.1, input_bias=True, input_dim=3)


The ``W`` arrays is used to store the weights of the neurons of the reservoir, while the
``Win`` matrix is used to store the weights of tht connections between those neurons and
the inputs.

ESNs can then be trained on sequential data, for instance text, sound, speech, or any kind
of timeseries, especially chaotic ones. Once again, this training only requires very simple
computational steps: computing the internal states of the reservoir, and then fitting these
states to the desired outputs using a simple linear regression to build the readout matrix
``Wout``. These two steps are handled by the function :py:func:`reservoirpy.ESN.train` :

.. doctest::

    >>> from math import sin, cos
    >>> # some dummy sequential data
    >>> Xn0 = np.array([[sin(x), cos(x)] for x in range(0, 100)])
    >>> Xn1 = np.array([[sin(x), cos(x)] for x in range(1, 101)])
    >>> # learn to predict X(n+1) (Xn1) given X(n) (Xn0)
    >>> states = esn.train([Xn0], [Xn1])

That's it ! Your model is now ready to be used for prediction.