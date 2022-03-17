.. _compat:

{{ header }}

================================
Previous versions guide (v0.2)
================================

This guide covers the usage of ReservoirPy v0.2. All tools from v0.2 are still available in the
:py:mod:`reservoirpy.compat` module.


Build a reservoir
=================

Build internal weights
----------------------

The core component of any flavor of reservoir computing is the *reservoir*, a type of recurrent artificial
neural network.

Long story short, reservoirs are recurrent neural networks typically sparsely connected, i.e. only a few (generally around
10% or 20%) of all the possible connections between the neuronal units are used. These networks are usually quite
large (a thousand units or more), and all their connections are initialized randomly, **and kept constant during
training**. Initialization of connections weights is therefore a very important step. Two major aspects of this
initialization are covered by ReservoirPy:

* the **spectral radius**, a parameter controlling the choatic dynamics emerging
  in the neuronal activities of the reservoir,
* the **probability of connection**, or *sparsity* of the reservoir, which controls how sparsely connected the neuronal
  units are inside the reservoir.

To quickly build reservoirs, ReservoirPy provides you with the :py:mod:`reservoirpy.mat_gen` module, and in particular the
:py:func:`mat_gen.generate_internal_weights` function::

    >>> from reservoirpy.mat_gen import generate_internal_weights
    >>> W = generate_internal_weights(100, proba=0.2, sr=0.9)

Here, `W` represents the connection weights between 100 neuronal units in a
square matrix of shape `(100, 100)`, with only 20% of non-zero values, and a spectral
radius of 0.9.

The weights values stored in `W` are randomly chosen following the standard normal distribution, but this can
be adjusted as well::

    >>> W = generate_internal_weights(100, dist="uniform", low=-1, high=1, sr=0.9)

The non-zero weights in `W` are now uniformly distributed between -1 and 1.
Because the probability of connection should generaly be quite low, the default `proba` parameter
is set to 0.1: only 10% of the connections are non-zero.

.. note::

    All weights are initialized using random variable models from the module
    `scipy.stat <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. Hence,
    any of the distributions described in this module can be used in ReservoirPy, along
    with their corresponding parameters. The only exception is the ``uniform`` distribution,
    whose parameters are `low` and `high` in ReservoirPy (in replacement of the Scipy
    `loc` and `scale` that can be confusing in that case).

Connect the reservoir to the inputs
-----------------------------------

The reservoir needs to be connected to the inputs with external connections randomly going from each
input dimensions to some of the reservoir units. We can define such connections using once again the
:py:mod:`reservoirpy.mat_gen` module, and the :py:func:`mat_gen.generate_input_weights` function::

    >>> from reservoirpy.mat_gen import generate_input_weights
    >>> Win = generate_input_weights(100, 10, proba=0.1, input_scaling=0.5, input_bias=True)

This function allows to tune two important parameters of the input weights:

* the **input scaling**, which rescales the input weights values.
* the **probability of connection**, as in the reservoir, which controls the proportion
  of connection between the inputs and the reservoir units.

In the last exemple, we created an input matrix `Win` of shape `(100, 11)`, meaning that this matrix
is able to operate between 10-dimensional inputs and a reservoir made of 100 neuronal units, with
an additional bias term, that can be discarded by setting the `input_bias` parameter to ``False``.
Only 10% of all the possible connections between the reservoir and the inputs are non-zero. All the other
weights are by default randomly chosen to be :math:`1` or :math:`-1`. Then, these weights are rescaled by a factor
of :math:`0.5`.

Like with the reservoir weights generation, the distribution of weights can be adjusted,
for example::

    >>> Win = generate_input_weights(100, 10, proba=0.1, input_scaling=1.0, dist="norm", scale=0.8)

will generate weights distributed following a normal distribution centered around 0 with a standard
deviation of 0.8.

.. note::

    All weights are initialized using random variable models from the module
    `scipy.stat <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. Hence,
    any of the distributions described in this module can be used in ReservoirPy, along
    with their corresponding parameters. The only exception is the ``uniform`` distribution,
    whose parameters are `low` and `high` in ReservoirPy (in replacement of the Scipy
    `loc` and `scale` that can be confusing in that case).

Work with feedback loops
------------------------

In some cases, you might want to define another set of connections between the output of the
model and the reservoir. This is called a feedback loop. We can define such connections using
once again the :py:func:`mat_gen.generate_input_weights` function::

    >>> from reservoirpy.mat_gen import generate_input_weights
    >>> Wfb = generate_input_weights(100, 5, proba=0.1, input_scaling=0.5, input_bias=False)

`Wfb` is in that case the matrix storing the weights between the output of the model and the
reservoir. These weights should also be chosen randomly, and only a few amount of the connections
should be non-zero (only 10% in the example above). We define the shape of `Wfb` in the same way
we define the shape of `Win`. However, the input dimension is now the output dimension of the model.
Here, for example, the model is supposed to predict a 5-dimensional vector of response, hence the
shape of `Wfb` is `(100, 5)` (considering we are still using a reservoir of 100 neuronal units,
and that adding a constant bias is not necessary).

Make it faster
--------------

In most cases, the reservoir should contain a high number of units, typically around
one or two thousands. ReservoirPy allows to exploit some properties of the internal
weights of the reservoir to speed up both the initialization of weights
and the computations during a task.

To speed up the initialization, we use the method of Gallichio et al. [1]_ called
*fast spectral spectral initialization*. In large matrices, computing the eigenvalues
to find the spectral radius can be long. This method allows to fix the spectral radius
value to a desired constant without having to explicitely compute the spectral radius,
saving great amount of time, using the :py:func:`mat_gen.fast_spectral_initialization` function::

    >>> from reservoirpy.mat_gen import fast_spectral_initialization
    >>> W = fast_spectral_initialization(100, proba=0.2, sr=0.9)

Additionally, all internal weights matrices are by default provided as sparse matrices, using
the `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ matrix format. This
format allows to benefit from the sparsity of the matrices defined (typically only 10%-20% of the
values are non-zero) to speed up the computations and save memory. If needed, these matrices can be
converted to Numpy array with their ``to_array()`` method. The `sparsity_type` parameter of all the
weight generation functions, if set to ``"dense"``, will override the default behavior and make
the functions build classical Numpy arrays::

    >>> # this is a sparse matrix (by default)
    >>> W = generate_interal_weights(100)
    >>> # this is dense matrix
    >>> W = generate_internal_weights(100, sparsity_type="dense")

.. [1] C. Gallicchio, A. Micheli, and L. Pedrelli,
       "Fast Spectral Radius Initialization for Recurrent
       Neural Networks", in Recent Advances in Big Data and
       Deep Learning, Cham, 2020, pp. 380–390,
       doi: 10.1007/978-3-030-16841-4_39.

Gather the components
---------------------

All weights matrices can then be packed inside a :py:class:`compat.ESN` object. The :py:class:`compat.ESN` object
is used to model an *Echo State Network*, a simple recurrent neural network only composed of an input matrix
weights `Win`, a reservoir matrix weights `W`, and optionally a feedback matrix weights `Wfb`. This
object will also provide the user with functions to run the reservoir on inputs and to learn a
response from the generated internal activations.

To define an ESN, simply build an :py:class:`compat.ESN` object with the previously
built matrices::

    >>> from reservoirpy import ESN
    >>> esn = ESN(lr=0.2, W=W, Win=Win, input_bias=True)

Notice that we have to define a last parameter to make things works: the `lr` parameter, standing
for *leaking rate*. The leaking rate controls the *memory flow* of the reservoir during the computation
of its internal states. A low leaking rate allows previous states to be *remembered more*, while
a maximum leaking rate of 0 deactivates the leaky integration.

What about feedback ? It can also be enabled by specifying some more parameters, like an output
activation function::

    >>> fb_esn = ESN(lr=0.2, W=W, Win=Win, Wfb=Wfb, input_bias=True, fbfunc=lambda x: x)

We just defined an ESN using some feedback connections specifyed in `Wfb`. The outputs will be
fed to `Wfb`, after being passed to the `fbfunc` function. Here, the output activation function
is just the identity function, but it could be, for instance, a sigmoid function if you are building
a classifier with logits outputs.

Run a reservoir
===============

.. _run_reservoir:

The :py:class:`compat.ESN` object provides you with a method to compute internal activations produced
in response to some inputs: the :py:meth:`compat.ESN.compute_all_states` method::

    >>> states = esn.compute_all_states(X)

Assuming that ``X`` is a list of Numpy arrays, you will obtain a list of internal activations
``states`` of the same length. :py:class:`compat.ESN` objects can only operate on Python list objects,
to enable **parallelization of state computation**. If ``X`` contains several independant
sequences of inputs, for instance, several sequences of text, or speech, or timeseries,
ReservoirPy will enabled parallelization of computation by default, using the
`joblib <https://joblib.readthedocs.io/en/latest/>`_ module. Each sequence will then be
treated separately and simultaneously to speed up the computation of activations.

.. note::

    If ``X`` is composed of an unique sequence (i.e. an unique Numpy array),
    it should always be passed as parameter as ``[X]``, a list containing only
    one element.


Start from a previously computed state
--------------------------------------

In some cases, you may want to initialize the current state of the resevoir with some
previously computed activations to avoid having arbitrary transient values, or to perform
the computation of next states using the memory accumulated during previous computations.
This can be achieved by passing a state vector through the `init_state` parameter::

    >>> states = esn.compute_all_states(X, init_state=previous_state)

The ``previous_state`` variable must be a Numpy array storing a N-dimensional vector, where
N is the number of units in the reservoir.

Use feedback
------------

If a feedback matrix was provided when creating the ESN,
feedback will be enabled and computing internal states will
require having the previously computed output values at disposal. If the ESN is not trained yet,
and has no way to produce other outputs that its internal states, the *teacher forcing* technique
must be used::

    >>> states = fb_esn.compute_all_states(X, forced_teachers=y)

We artificialy use some expected output values ``y`` as feedback values for the ESN. Of course,
the ``y`` values must be the outputs values expected from ``X``, hence ``X`` and ``y`` are both
sequences of Numpy arrays of same length.

.. note::

    If ``y`` is composed of an unique sequence (i.e. an unique Numpy array),
    it should always be passed as parameter as ``[y]``, a list containing only
    one element.

If the ESN is trained, and is therefore able to compute the output value from its internal states,
teacher forcing is no longer required, and the feedback loop will be automatically handled by
ReservoirPy.

Because the feedback vector will be initialized to 0 at the beginning of the run, it is also
possible to provide the function with an initial feedback vector to avoid producing
odd transient states. This can be achieved using the `init_fb` parameter::

    >>> states = fb_esn.compute_all_states(X, init_fb=last_output)

The ``last_output`` variable must be a Numpy array storing a V-dimensional vector, where
V is the output dimension of the reservoir.

Train an ESN
============

.. _train esn:

ESNs can be trained on various tasks using very simple learning rules, usually adapated from
linear regression techniques. This learning phase allows to create a new weight matrix `Wout`,
that will be used as a *readout* of the internal states: each activation vector produced
by the reservoir will be *read* using the learnt weights to produce the desired output value.

There are several ways to train an :py:class:`compat.ESN` in ReservoirPy, but the most important one
is using the :py:meth:`compat.ESN.train` method::

    >>> esn.train(X, y)

Using some target values ``y`` and the states produced using the input values ``X``, the ESN
will learn a new ``Wout`` matrix.

By default, training states are not explicitely computed and returned. You can force this behaviour
by using the ``return_states`` parameter::

    >>> states = esn.train(X, y, return_states=True)

.. note::

    Similarly than with the :py:meth:`compat.ESN.compute_all_states` method, both the input values ``X``
    and target values ``y`` must be list of Numpy array. If only one sequence is required for training,
    then ``X`` and ``y`` should be lists with only one element.

This training can be tuned in several ways: first, all the parameters presented in :ref:`run_reservoir`
can be used in the :py:meth:`compat.ESN.train` method, like `wash_nr_time_steps` or `init_state`. This
allows to define the behavior of the ESN regarding its transient states. Second, the learning rule
for ``Wout`` can be specifyed at the creation of the ESN object. By default, this learning rule
use a pseudo-inversion of the states matrix (the concatenation of all internal states vectors)
to find the solution of :math:`y = Wout \cdot X`. But many other learning rules are available for
offline learning, and an another object (:py:class:`compat.ESNOnline`) also provides online learning
rules.

Train with ridge regression
---------------------------

A common learning process for readout matrix is the `Tikhonov regression <https://fr.wikipedia.org/wiki
/R%C3%A9gularisation_de_Tikhonov>`_, using L2 regularization,
also called *ridge regression*.

Ridge regression is implemented inside ReservoirPy. You can enable ridge regression by simply
passing a ridge coefficient to the ESN constructor::

    >>> esn = ESN(lr=0.2, W=W, Win=Win, ridge=1e-6)

The `ridge` parameter allows to set this coefficient to any positive value. If this parameter is
set, ridge regression is automatically enabled during training.

Ridge regression is well-suited for most tasks, while preventing overfitting.

Train with a *scikit-learn* regression model
--------------------------------------------

If you wish to use another strategy for the learning of `Wout`, it is possible to
use any `scikit-learn <https://scikit-learn.org/stable/>`_ estimator, as soon as this
estimator can be approximated to a linear transformation, i.e. the estimator has ``coef_`` and
``intercept_`` attributes.

For instance, a `LogisticRegression <https://scikit-learn.org/stable/modules/generated/
sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_ estimator
can be pass to the ESN through the `reg_model` parameter, to use logistic regression as a
learning rule for the `Wout` matrix::

    >>> from sklearn.linear_model import LogisticRegression
    >>> logit = LogisticRegression()
    >>> esn = ESN(lr=0.2, W=W, Win=Win, reg_model=logit)

The `Wout` matrix will then be defined as the concatenation of the ``coef_`` matrix storing
the learnt weights in the estimator and the ``intercept_`` vector storing its learnt biases.

.. note::

    When using a scikit-learn estimator, make sure the shape and format of the outputs ``y``
    correspond to what is expected by the estimator ``fit()`` method.

Delayed training on precomputed states
--------------------------------------

If you wish to learn the readout matrix *after* having computed the states, you can use
the :py:meth:`compat.ESN.fit_readout` method::

    >>> states = esn.compute_all_states(X) # compute the states
    >>> # do some other tasks
    >>> ...
    >>> # finally learn the readout
    >>> Wout = esn.fit_readout(states, y)
    >>> # optionally store the readout in a model
    >>> esn.Wout = Wout

This method also allows to change momentarily the learning rule of an existing model
to compute a readout, using `ridge`, `reg_model` and `force_pinv` parameters::

    >>> Wout0 = esn.fit_readout(states, y, ridge=1e-5) # change the ridge value
    >>> Wout1 = esn.fit_readout(states, y, force_pinv=True) # use pseudo inverse for solving

Use the ESN for prediction
==========================

To start predicting values with an :py:class:`reservoirpy.ESN` object, you must first train it on a
task as presented in the :ref:`train esn` section.

Once the training process is done, your :py:class:`compat.ESN` object is ready to use for prediction,
using its learnt `Wout` matrix. To predict values from a sequence of inputs, you can use the
:py:meth:`compat.ESN.run` method::

    >>> outputs, states = esn.run(X)

This method always returns both the internal states and the predicted outputs for the given inputs.

Delayed predictions on precomputed states
-----------------------------------------

If you wish to learn the readout matrix *after* having computed the states, you can use
the :py:meth:`compat.ESN.compute_outputs` method::

    >>> states = esn.compute_all_states(X) # compute the states
    >>> # do some other tasks
    >>> ...
    >>> # finally compute the predicted values
    >>> outputs = esn.compute_outputs(states)

Use the ESN for generation
==========================

ReservoirPy also allows, under certain conditions, to use the :py:class:`compat.ESN` objects on
so called *generative mode*. In generative mode, the ESN is asked **to run on its own predictions**.
Its outputs become its inputs, making the ESN generating data without the need of any external
inputs.

To enable generative mode, the ESN must be trained on a regression task, where the output
space has exactly the same dimensions than the input space. Otherwise, it would be impossible
for the ESN to run on its own inputs (except using feedback loops, but this is not yet possible
in ReservoirPy).

If the ESN is trained on the right task, you can use its :py:meth:`compat.ESN.generate` method to
generate data::

    >>> outputs, states, warming_outputs, warming_states = esn.generate(100,
    ...                                                                 warming_inputs=warm_x)

This method generates ``outputs`` and ``states`` without the need on any inputs. Here, we
generate 100 time steps of ``outputs`` and ``states``.

Because generation can not begin with a default null state, it is mandatory to pass as
parameter either an initial state to begin the generation with or some inputs to warm
up the internal states of the reservoir, using the `init_state` or `warming_inputs`
parameters. In the example above, we give as warming inputs a Numpy array ``warm_x`` storing
a few time steps of inputs values. As a consequence, the method :py:meth:`compat.ESN.generate` will
also returns ``warming_outputs`` and ``warming_states``, the outputs and states computed from
the warming inputs.

Online ESN
==========

ReservoirPy also implements the FORCE [2]_ learning rule, allowing to learn the readout matrix
in a online way. This learning process is available through the :py:class:`compat.ESNOnline` object,
which works in a similar way than the :py:class:`compat.ESN` object. Some additional methods however,
like the :py:meth:`compat.ESNOnline.train_on_current_state` method, allows to train readouts
continuously.

.. [2] D. Sussillo and L. F. Abbott, ‘Generating Coherent
       Patterns of Activity from Chaotic Neural Networks’, Neuron,
       vol. 63, no. 4, pp. 544–557, Aug. 2009,
       doi: 10.1016/j.neuron.2009.07.018.
