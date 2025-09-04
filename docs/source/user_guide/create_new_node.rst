.. _create_new_node:

.. py:currentmodule:: reservoirpy.node

==================================
Create your own :py:class:`~.Node`
==================================


Subclassing the :py:class:`~.Node` to create a custom operator takes only a
few steps to be done and operational. Subclasses of :py:class:`~.Node` can
then be used as any other node instances. Before implementing your Node, you have to know
which type of Node you want to implement.


The Node abstract base classes
==============================

Here is an overview of the different abstract base classes for the :py:class:`~.Node` and their inheritance
relationships:

.. code::

    Node
    ╠══ TrainableNode
    ║   ╠══ ParallelNode
    ║   ╚══ OnlineNode


- | A :py:class:`~.Node` is the most basic class. It implements the :py:meth:`~.Node.step` and
    :py:meth:`~.Node.run` methods, as well as a few useful other.
  | :py:class:`reservoirpy.nodes.Reservoir`, :py:class:`reservoirpy.nodes.ES2N` or :py:class:`reservoirpy.nodes.Tanh`
    are some examples of node classes that inherit the ``Node`` class but not the others.

- | A :py:class:`~.TrainableNode` is a node that can be trained. It implements the ``.fit`` method. This method
    can update attributes of the instance (for example, ``Wout``, ``bias``, ...).
  | :py:class:`reservoirpy.nodes.LocalPlasticityReservoir`, :py:class:`reservoirpy.nodes.ScikitLearnNode` or
    :py:class:`reservoirpy.nodes.IPReservoir` are some examples of node classes that inherit the ``TrainableNode`` class
    but not the others.

- | A :py:class:`~.ParallelNode` is a trainable node that can be fit in parallel on multiple timeseries. Parallel
    learning can be done using the ``workers`` argument of the :py:meth:`~.ParallelNode.fit` method.
    Under the hood, those nodes implements two methods, :py:meth:`~.ParallelNode.worker` and
    :py:meth:`~.ParallelNode.master`.
  | :py:class:`reservoirpy.nodes.Ridge` is an example of a ``ParallelNode``.

- | An :py:class:`~.OnlineNode` is a trainable node that can be trained incrementally, and returns a prediction
    at each timestep. Online learning can be done using the :py:meth:`~.OnlineNode.partial_fit`.
  | :py:class:`reservoirpy.nodes.RLS` and :py:class:`reservoirpy.nodes.LMS` are some examples of an ``OnlineNode``.



Create your own :py:class:`~.Node`
==================================

All ReservoirPy Nodes inherits the :py:class:`~.Node` class. It is the simplest kind of node that you can create.
Here is the general outline of a new node:


.. code-block:: python

    from reservoirpy import Node


    class NodeName(Node):
        def __init__(self, params):
            ...
            # set attributes from params

        def initialize(self, x: Timestep | Timeseries | MultiTimeseries):
            # set input_dim & output_dim
            # initialize the node state dict
            self.initialized = True

        def _step(self, state: State, x: Timestep) -> State:
            # compute the new_state
            y = ...
            return {"out": y}

:py:meth:`~.Node.__init__`
~~~~~~~~~~~~~~~~~~~~~~~~~~

This method simply takes the node arguments and set attributes from them. It is recommended to support the following
parameters: ``input_dim``, ``output_dim``, ``dtype``, ``seed`` and ``name``.

:py:meth:`~.Node.initialize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usually, the ``input_dim`` and ``output_dim`` attributes can be inferred from the data the node receives. This makes
those arguments optional at node creation, and alleviates the need to specify each input and output dimensions for each
node of a model. For this to work, ReservoirPy relies on a delayed initialization mechanism:

Whenever a node is called on data (:py:meth:`~.Node.step`, :py:meth:`~.Node.run`, :py:meth:`~.TrainableNode.fit`,
:py:meth:`~.OnlineNode.partial_fit`, ...), if the Node has not been initialized, the :py:meth:`~.Node.initialize` method is
called before everything else. The ``initialize`` method takes the input (``x``) and optionally the training data (``y``),
and from those values, define the input and output dimensions of the node, and, if needed, use them to generate additional
values (e.g. ``W``, ``Win``).

This method must also initialize the node state. It is a dictionary that must at least have the ``"out"`` key, with an
array of shape ``(output_dim,)``. This array is the default output of the node.

:py:meth:`~.Node._step`
~~~~~~~~~~~~~~~~~~~~~~~

Once everything has been initialized, our node can be used! The fundamental operation of the node is in :py:meth:`_step`.
This method takes the current state of the node and a timestep, and returns the new state of the node.

Any node can be run in parallel. But for this to work the `_step` method must be purely functional. This means:
- No mutation of the object. You can retrieve variables from ``self``, but not modify them. If your node is supposed to
evolve over time, then it should probably be in the ``state``, or be part of the training phase.
- No external calls or side effects: printing, reading/writing files, and such are prohibited, as their usage can lead
to unexpected consequences.

(optional) :py:meth:`~.Node._run`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to the :py:meth:`_step` method, :py:meth:`_run` is a purely functional method that defines the behavior of the
node when ran on a timeseries. This should be strictly equivalent to calling :py:meth:`_step` in a loop, collecting the
output, and returning the last state. This is what it does by default, so it is not necessary to redefine it.

However, in some cases, it may be beneficial to reimplement it, as some nodes can benefit from vectorization.

A simple example
~~~~~~~~~~~~~~~~

Let's illustrate what we have seen with a simple example. We will define a :py:class:`Node` that simply adds a floating
number ``a`` to its input. This Node has the same input and output dimension. And this node can be vectorized, so we
override the default implementation of ``_run``.

.. code-block:: python

    import numpy as np
    from reservoirpy import Node


    class MyNode(Node):
        def __init__(self, a, name=None):
            self.a = a
            self.name = name

        def initialize(self, x):
            # set input_dim & output_dim
            self._set_input_dim(x)
            self.output_dim = self.input_dim
            # define the state
            self.state = {"out": np.zeros((self.output_dim,))}
            # switch the initialized parameter to True
            self.initialized = True

        def _step(self, state: State, x: Timestep) -> State:
            output_value = x + a
            return {"out": output_value}

        def _run(self, state: State, x: Timeseries) -> tuple[State, Timeseries]:
            output_series = x + a  # vectorized on the whole series
            return {"out": output_series[-1]}, output_series


    my_node = MyNode(a=10)




Create your own :py:class:`~.TrainableNode`
===========================================

What we have created so far is a simple node that cannot be trained, like the reservoir. But some nodes need to be
trained, such as the readout layer in the regular reservoir computing paradigm. A :py:class:`~.TrainableNode` is a
:py:class:`~.Node` that implements the :py:meth:`~.TrainableNode.fit` method.


:py:meth:`~.TrainableNode.initialize`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a trainable node, :py:meth:`initialize` takes an additional ``y`` parameter, that can be ``None`` in case of an
unsupervised Node.

:py:meth:`~.TrainableNode.fit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a stateful (modifies the node attributes, and can have side effects) method that takes input data (a timeseries,
or multiple timeseries in the form of a 3D array or a list of timeseries) ``x`` and an optional teacher data ``y`` in
the same format. The fit method also takes a ``warmup`` parameter.

For the user convenience, this method returns its instance.


A simple example
~~~~~~~~~~~~~~~~

In this scenario, our training phase consists in computing the mean difference between ``x`` and ``y`` for each
dimension. In the predicting phase, we will just add this value to the input.

.. code-block:: python

    import numpy as np
    from reservoirpy import Node


    class MyNode(TrainableNode):
        mean_diff: np.ndarray = None

        def __init__(self, name=None):
            self.name = name

        def initialize(self, x, y):
            self._set_input_dim(x)
            self._set_output_dim(y)
            assert self.input_dim == self.output_dim
            self.state = {"out": np.zeros((self.output_dim,))}
            self.initialized = True

        def _step(self, state: State, x: Timestep) -> State:
            output_value = x + self.mean_diff
            return {"out": output_value}

        def fit(self, x, y, warmup):
            if not self.initialized:
                self.initialize(x, y)

            if isinstance(x, Sequence):
                x = np.concatenate([x_[warmup:] for x_ in x])
                y = np.concatenate([y_[warmup:] for y_ in y])
            else:
                x, y = x[..., warmup:, :], y[..., warmup:, :]

            self.mean_diff = np.mean((y - x).reshape(-1, x.shape[-1]), axis=0)
            return self


    my_node = MyNode(a=10)




Create your own :py:class:`~.OnlineNode`
========================================

If you want to implement an online learning rule, your node inherits the :py:class:`~.OnlineNode` class. This class
implements the :py:meth:`~.OnlineNode.partial_fit` method.


:py:meth:`~.OnlineNode.partial_fit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an online node, :py:meth:`partial_fit` takes an optional ``y`` parameter, that can be ``None`` in case of an
unsupervised Node. Note that this method does not allow for multiple timeseries, it only takes timeseries as inputs.



Create your own :py:class:`~.ParallelNode`
==========================================

In some cases, a model can be adapted to be fit in parallel on multiple timeseries. We support this in ReservoirPy via
the :py:class:`~.ParallelNode` . A ParallelNode implements two methods that are automatically called when fitting: the
:py:meth:`~.ParallelNode.worker` and :py:meth:`~.ParallelNode.master`. A worker processes one timeseries and return an
intermediate result. The master takes an iterable of those intermediate results and processes them to update the node's
attributes accordingly.


:py:meth:`~.ParallelNode.worker`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:meth:`worker` takes an input timeseries `x` and an optional teacher timeseries `y`. It may return anything, and
its output will be received by the ``master``.


:py:meth:`~.ParallelNode.master`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:meth:`master` takes an iterable (usually a generator) of whatever the workers returns, and doesn't return anything.
Its role is to iterate on the worker's output, process the output, and update the node.