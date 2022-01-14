.. _quick:

{{ header }}

===========
Quick start
===========

Let's have a look at ReservoirPy's main functionalities and features.

Before using :mod:`reservoirpy`, make sure you have correctly installed
the package and all its dependencies by :ref:`getting_started`.

ReservoirPy is an open-source project which aims to develop a simple yet performant
*Reservoir Computing* tool, written in scientific Python and without using any
heavy machine learning library or framework.


Building your first Echo State Nertwork
=======================================

    Here is how, with only a few NumPy arrays, you can define a complete (not so) dummy Echo State Network,
    using the :py:class:`Reservoir` and :py:class:`Ridge` classes:

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
    ``Win`` matrix is used to store the weights of the connections between those neurons and
    the inputs.

    In a more realistc context, you will probably need to quickly define these matrices and to
    tune some of their parameters. You can use the :py:mod:`reservoirpy.mat_gen` module to achieve this:

    .. doctest::

        >>> from reservoirpy.mat_gen import fast_spectral_initialization
        >>> W = fast_spectral_initialization(100, sr=0.9, proba=0.2)

    ``W`` is now a sparse array of shape ``(100, 100)``, i.e. represents a reservoir network of 100 neuronal
    units, where only 20% (``proba=0.2``) of all possible connections between the neurons are non-zero,
    and with a spectral radius (``sr=0.9``) equals to :math:`0.9`.

    It is also possible to define the input matrix ``Win`` using :py:mod:`reservoirpy.mat_gen` module:

    .. doctest::

        >>> from reservoirpy.mat_gen import generate_input_weights
        >>> Win = generate_input_weights(100, 2, input_bias=False, proba=1.)

    Here, we created a dense (``proba=1.``) input matrix able to connect a 2-dimensional input to a
    reservoir composed of 100 neuronal units, without adding a constant bias.

    For the next steps of this tutorial, we will keep using the first two "dummy" matrices we
    defined in the first place.

Training your ESN
=================

    ESNs can then be trained on sequential data, for instance text, sound, speech, or any kind
    of timeseries, especially chaotic ones. Let's build some simple sinusoidal sequential data,
    for instance simple oscillations flowing through time:

    .. doctest::

        >>> from math import sin, cos
        >>> # some dummy sequential data
        >>> Xn0 = np.array([[sin(x), cos(x)] for x in np.linspace(0, 4*np.pi, 500)])
        >>> Xn1 = np.array([[sin(x), cos(x)] for x in np.linspace(np.pi/4, 4*np.pi+np.pi/4, 500)])

    The result is displayed below: two timeseries based on cosinus and sinus functions.
    The ESN will have to predict their future values :math:`\frac{\pi}{4}` timesteps towards
    their current values, simultaneously:

    .. image:: _static/img/getting_started_sinus.svg

    Training the ESN on this task only requires very simple
    computational steps: computing the internal states of the reservoir, and then fitting these
    states to the desired outputs using a simple linear regression to build the readout matrix
    ``Wout``. These two steps are handled by the function :py:func:`reservoirpy.ESN.train` :

    .. doctest::

        >>> # learn to predict X(n+1) (Xn1) given X(n) (Xn0)
        >>> states = esn.train([Xn0], [Xn1])

    That's it ! Your model has now a readout matrix ``Wout`` storing the connections weights in charge of
    computing the desired output. It is now ready for prediction.

Testing and running your ESN
============================

    First, let's add some (lot of) noise to the original input timeseries :

    .. doctest::

        >>> Xtest0 = Xn0 + np.random.normal(0, 0.5, size=Xn0.shape)

    Then, we use the :py:func:`reservoirpy.ESN.run` function to use the freshly
    trained ESN on those noisy data:

    .. doctest::

        >>> outputs, states = esn.run([Xtest0])

    .. image:: _static/img/getting_started_sinus_result.svg

    Not so bad ! Of course this example is trivial, and the ESN can be used on much more
    complicated tasks, like speech recognition or chaotic timeseries prediction. To fully
    deploy the capacities of ESNs, ReservoirPy provides you with many other simple tools
    that can handle a large variety of tasks and situations, from simple timeseries forecasting
    to sound analysis.

Going further
=============

To handle more complicated and realistic cases, you will probably need to pay a particular attention to
how the reservoir and input matrix are built, how the readout matrix is trained, and how to evaluate
your model to find the best parameters. All these aspects of reservoir computing are covered in the following tutorials:

- :ref:`rc with reservoirpy`, to go deeper into ReservoirPy API and see more realistc examples and applications
