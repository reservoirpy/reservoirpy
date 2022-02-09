.. _learning_rules:

==============
Learning rules
==============

Reservoir Computing techniques allow the use of a great variety of learning mechanisms to solve specific tasks.
In ReservoirPy, these learning rules are sorted in two categories: **offline** learning rules
and **online** learning rules.

Nodes can be equiped with such learning rules, and learning can be triggered by using their,
:py:meth:`~.Node.fit` (offline learning) and :py:meth:`~.Node.train` (online learning) methods.

Offline learning rules - Linear regression
------------------------------------------

Offline learning rules are the most common learning rules in machine learning. They include gradient descent and linear
regression amongst others. Within the Reservoir Computing field, linear regression is probably the simplest and the more
used way of training an artificial neural network.

Linear regression is said to be an *offline* learning rule because parameters of the linear regression model are learned
given all available samples of data and all available samples of target values. Once the model is learned, it can not
be updated without training the model on the whole dataset another time. Training and data gathering happen in two
separate phases.

Linear regression is implemented in ReservoirPy through the :py:class:`~.Ridge` node. Ridge node is equiped with a
regularized linear regression learning rule, of the form :eq:`ridge`:

.. math::
    :label: ridge

    W_{out} = YX^\intercal (XX^\intercal + \lambda Id)^{-1}

Where :math:`X` is a series of inputs, and :math:`Y` is a series of target values that the network must learn to
predict. :math:`\lambda` is a regularization
parameter used to avoid overfitting. In most cases, as the :py:class:`~.Ridge` node will be used within an Echo State
Network (ESN), :math:`X` wil represent the series of activations of a :py:class:`~.Reservoir` node over a timeseries.
The algorithm will therefore compute a matrix of neuronal weights :math:`W_{out}` (and a bias term)
such as predictions can be computed using equation :eq:`ridgeforward`.
:math:`W_{out}` (and bias) is stored in the node :py:attr:`Node.params` attribute.

.. math::
    :label: ridgeforward

    y[t] = W_{out}^\intercal x[t] + bias

which is the forward function of the Ridge node. :math:`y[t]` represents the state of the Ridge neurons at step
:math:`t`, and also the predicted value given the input :math:`x[t]`.


Offline learning with :py:meth:`~.Node.fit`
-------------------------------------------

Offline learning can be performed using the :py:meth:`~.Node.fit` method.
In the following example, we will use the :py:class:`~.Ridge` node.

We start by creating some input data ``X`` and some target data ``Y`` that the model has to predict.

.. ipython:: python

    X = np.arange(100)[:, np.newaxis]
    Y = np.arange(100)[:, np.newaxis]

Then, we create a :py:class:`~.Ridge` node. Notice that it is not necessary to indicate the number of neurons in that
node. ReservoirPy will infer it from the shape of the target data.

.. ipython:: python

    from reservoirpy.nodes import Ridge

    ridge = Ridge().fit(X, Y)

We can access the learned parameters looking at the ``Wout`` and ``bias`` parameter of the node.

.. ipython:: python

    print(ridge.Wout, ridge.bias)

As ``X`` and ``Y`` where the same timeseries, we can see learning was successful: the node has learned the identity
function, with a weight of 1 and a bias of 0.

Ridge regression can obviously handle much more complex tasks, such as chaotic attractor modeling or timeseries
forecasting, when coupled with a reservoir inside an ESN.

Offline learning with :py:meth:`~.Model.fit`
--------------------------------------------

Models also have a :py:meth:`~.Model.fit` method, working similarly to the one of the Node class presented above.
The :py:meth:`~.Model.fit` method can only be used if all nodes in the model are offline nodes, or are not trainable.
If all nodes are offlines, then the :py:meth:`~.Node.fit` method of all offline nodes in the model will be called
as soon as all input data is available. If input data for an offline node B comes from another offline node A,
then the model will fit A on all available data, then run it, and finally resume training B.

As an example, we will train the readout layer of an ESN using linear regression. We first create some toy dataset: the
task we need the ESN to perform is to predict the cosinus form of a wave given its sinus form.

.. ipython:: python

    X = np.sin(np.linspace(0, 20, 100))[:, np.newaxis]
    Y = np.cos(np.linspace(0, 20, 100))[:, np.newaxis]

Then, we create an ESN model by linking a :py:class:`~.Reservoir` node with a :py:class:`~.Ridge` node. The
:py:class:`~.Ridge` node will be used as readout and trained to learn a mapping between reservoir states
and targeted outputs. We will regularize its activity using a ridge parameter of :math:`10^{-3}`. We will also tune
some of the reservoir hyperparameters to obtain better results.
We can then train the model using :py:meth:`~.Model.fit`.

.. ipython:: python

    from reservoirpy.nodes import Reservoir, Ridge

    reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), Ridge(ridge=1e-3)
    esn = reservoir >> readout
    esn.fit(X, Y)

During that step, the reservoir has been run on the whole timeseries, and the resulting internal states has been
used to perform a linear regression between these states and the target values, learning the connection weights
between the reservoir and the readout.
We can then run the model to evaluate its predictions:

.. ipython:: python

    X_test = np.sin(np.linspace(20, 40, 100))[:, np.newaxis]
    predictions = esn.run(X_test)

.. plot::

    from reservoirpy.nodes import Reservoir, Ridge
    reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), Ridge(ridge=1e-3)
    esn = reservoir >> readout
    X = np.sin(np.linspace(0, 20, 100))[:, np.newaxis]
    Y = np.cos(np.linspace(0, 20, 100))[:, np.newaxis]
    esn.fit(X, Y)
    X_test = np.sin(np.linspace(20, 40, 100))[:, np.newaxis]
    Y_test = np.cos(np.linspace(20, 40, 100))[:, np.newaxis]
    S = esn.run(X_test)
    plt.plot(Y_test, label="Ground truth cosinus")
    plt.plot(S, label="Predicted cosinus")
    plt.ylabel("ESN output")
    plt.xlabel("Timestep $t$")
    plt.legend()
    plt.show()

Online learning rules
---------------------

As opposed to offline learning, online learning allows to learn a task using only **local information in time**. Example
of online learning rules are Hebbian learning rules, Least Mean Squares (LMS) algorithm or Recurrent Least Squares
(RLS) algorithm.

These rules can update the parameters of a model one sample of data at a time, or one episode at a
time to borrow vocabulary used in the Reinforcement Learning field. While most deep learning algorithms can not used
such rules to update their parameters, as gradient descent algorithms
requires several samples of data at a time to obtain
convergence, Reservoir Computing algorithms can use this kind of rules. Indeed, only readout connections need to be
trained. A single layer of neurons can be trained using only local information (no need for gradients coming from
upper layers in the models and averaged over several runs).

Online learning with :py:meth:`~.Node.train`
--------------------------------------------

Online learning can be performed using the :py:meth:`~.Node.train` method.
In the following example, we will use the :py:class:`~.FORCE` node, a single layer of neurons equiped with
an online learning rule called FORCE algorithm.

We start by creating some input data ``X`` and some target data ``Y`` that the model has to predict.

.. ipython:: python

    X = np.arange(100)[:, np.newaxis]
    Y = np.arange(100)[:, np.newaxis]

Then, we create a :py:class:`~.FORCE` node. Notice that it is not necessary to indicate the number of neurons in that
node. ReservoirPy will infer it from the shape of the target data.

.. ipython:: python

    from reservoirpy.nodes import FORCE

    force = FORCE()

The :py:meth:`~.Node.train` method can be used as the call method of a Node. Every time the method is called, it updates
the parameter of the node along with its internal state, and return the state.

.. ipython:: python

    s_t1 = force.train(X[0], Y[0])
    print("Parameters after first update:", force.Wout, force.bias)
    s_t1 = force.train(X[1], Y[1])
    print("Parameters after second update:", force.Wout, force.bias)

The :py:meth:`~.Node.train` method can also be called on a timeseries of variables and targets, in a similar way to
what can be done with the :py:meth:`~.Node.run` function. All states computed during the training will be returned
by the node.

.. ipython:: python

    force = FORCE()
    S = force.train(X, Y)

As the parameters are updated incrementaly, we can see convergence of the model throughout training, as opposed
to offline learning where parameters can only be updated once, and evaluated at the end of the training phase.
We can see that convergence is really fast. Only the first timesteps of output display visible errors:

.. plot::

    from reservoirpy.nodes import FORCE
    X = np.arange(100)[:, np.newaxis]
    Y = np.arange(100)[:, np.newaxis]
    force = FORCE()
    S = force.train(X, Y)
    plt.plot(S, label="Predicted")
    plt.plot(Y, label="Training targets")
    plt.title("Activation of FORCE readout during training")
    plt.xlabel("Timestep $t$")
    plt.legend()
    plt.show()


We can access the learned parameters looking at the ``Wout`` and ``bias`` parameter of the node.

.. ipython:: python

    print(force.Wout, force.bias)

As ``X`` and ``Y`` where the same timeseries, we can see learning was successful: the node has learned the identity
function, with a weight of 1 and a bias close to 0.

Online learning with :py:meth:`~.Model.train`
---------------------------------------------

Models also have a :py:meth:`~.Model.train` method, working similarly to the one of the Node class presented above.
The :py:meth:`~.Model.train` method can only be used if all nodes in the model are online nodes, or are not trainable.
If all nodes are online, then the :py:meth:`~.Node.train` methods of all online nodes in the model will be called in the
topological order of the graph defined by the model. At each timesteps, onlines nodes are trained, called, and their
updated states are given to the next nodes in the graph.

As an example, we will train the readout layer of an ESN using FORCE learning. We first create some toy dataset: the
task we need the ESN to perform is to predict the cosinus form of a wave given its sinus form.

.. ipython:: python

    X = np.sin(np.linspace(0, 20, 100))[:, np.newaxis]
    Y = np.cos(np.linspace(0, 20, 100))[:, np.newaxis]

Then, we create an ESN model by linking a :py:class:`~.Reservoir` node with a :py:class:`~.FORCE` node. The
:py:class:`~.FORCE` node will be used as readout and trained to learn a mapping between reservoir states
and targeted outputs. We will tune some of the reservoir hyperparameters to obtain better results.
We can then train the model using :py:meth:`~.Model.train`.

.. ipython:: python

    from reservoirpy.nodes import Reservoir, FORCE

    reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), FORCE()
    esn = reservoir >> readout
    predictions = esn.train(X, Y)

During that step, the reservoir has been trained on the whole timeseries using online learning. We can have a look at
the outputs produced by the model during training to evaluate convergence:

.. plot::

    X = np.sin(np.linspace(0, 20, 100))[:, np.newaxis]
    Y = np.cos(np.linspace(0, 20, 100))[:, np.newaxis]
    from reservoirpy.nodes import Reservoir, FORCE
    reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), FORCE()
    esn = reservoir >> readout
    S = esn.train(X, Y)
    plt.plot(S, label="Predicted")
    plt.plot(Y, label="Training targets")
    plt.title("Activation of FORCE readout during training")
    plt.xlabel("Timestep $t$")
    plt.legend()
    plt.show()

We can then run the model to evaluate its predictions:

.. ipython:: python

    X_test = np.sin(np.linspace(20, 40, 100))[:, np.newaxis]
    predictions = esn.run(X_test)

.. plot::

    from reservoirpy.nodes import Reservoir, FORCE
    reservoir, readout = Reservoir(100, lr=0.2, sr=1.0), FORCE()
    esn = reservoir >> readout
    X = np.sin(np.linspace(0, 20, 100))[:, np.newaxis]
    Y = np.cos(np.linspace(0, 20, 100))[:, np.newaxis]
    esn.train(X, Y)
    X_test = np.sin(np.linspace(20, 40, 100))[:, np.newaxis]
    Y_test = np.cos(np.linspace(20, 40, 100))[:, np.newaxis]
    S = esn.run(X_test)
    plt.plot(Y_test, label="Ground truth cosinus")
    plt.plot(S, label="Predicted cosinus")
    plt.ylabel("ESN output")
    plt.xlabel("Timestep $t$")
    plt.legend()
    plt.show()
