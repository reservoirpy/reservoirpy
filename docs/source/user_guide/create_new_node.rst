.. _create_new_node:

================================
Create your own :py:class:`Node`
================================

Subclassing the :py:class:`Node` to create a custom operator takes only a
few steps to be done and operational. Sublasses of :py:class:`Node` can
then be used as any other node instances.

Create a ``forward`` function
-----------------------------

First, one needs to create the `forward` function that will be applied
by the new node class:

.. ipython::

    In [1]: import numpy as np

    In [2]: from reservoirpy import Node

    In [3]: def forward(node: Node, x: np.ndarray) -> np.ndarray:
       ...:     '''Does something to the current state of the node, the input
       ...:     data and some feedback.'''
       ...:     state = node.state()  # get current node state
       ...:     if node.has_feedback:
       ...:         feedback = node.feedback()  # call state of some distant node
       ...:     some_param = node.const1
       ...:     some_other_param = node.const2
       ...:     return x + some_param * state + some_other_param * feedback


This function **must** take as parameter a vector `x` of shape
``(1, dim x)`` (one timestep of data) and the node instance itself. You can
access any parameter stored in the node through this instance.

Create an ``initialize`` function
---------------------------------

Then, one needs to create the `initialize` function that will be used at
runtime to infer the input and output dimensions of the node, and optionaly
initialize some parameters (some neuronal weights, for instance):

.. ipython::

    In [4]: def initialize(node: Node, x: np.ndarray = None):
       ...:      '''This function receives a data point x at runtime and uses it to
       ...:      infer input and output dimensions.'''
       ...:      if x is not None:
       ...:          node.set_input_dim(x.shape[1])
       ...:          node.set_output_dim(x.shape[1])
       ...:          # you can initialize parameters here
       ...:          node.set_param("const1", 1)

Initialize feedback connections
-------------------------------

Additionaly, another function can be created to initialize feedback signal
dimension, if the node requires feedback:

.. ipython:: python

    def initialize_fb(node: Node, feedback=None):
        """This function is called at runtime and
        infer feedback dimensions.
        """
        if node.has_feedback:
            # in our case, feedback dimension is just the dimension of the
            # feedback vector.
            if feedback is not None:
                node.set_feedback_dim(feedback.shape[1])

Finally, you can add some other functions to train the parameter of your
node. See .. TODO: add link to train page
for more information.

Instanciate a new :py:class:`Node`
----------------------------------

That's it! You can now create a new :py:class:`Node` instance
parametrized with the functions you have just written:

.. ipython:: python

    node = Node(
        forward=forward,
        initializer=initialize,
        fb_initializer=initialize_fb,
        params={"const1": None},
        hypers={"const2": -1},
        name="custom_node",
    )

.. note::
    Do not forget to declare the mutable parameters `params` and immutable
    hyperparameters `hypers` as dictionnaries. `params` should store all
    parameters that need to be initialized and that will evolve during the
    life cycle of the node (for example, neuronal weights whom value will
    change during training). `hypers` should store parameters used to
    define the architecture or the behavior of the node instance, and that
    will not change through learning mechanisms.

Subclassing :py:class:`Node`
----------------------------

You can also create a new subclass of :py:class:`Node` in a similar way:

.. ipython:: python

    class CustomNode(Node):
        def __init__(self, const2=-1, name=None):
            super().__init__(
                forward=forward,
                initializer=initialize,
                fb_initializer=initialize_fb,
                params={"const1": None},
                hypers={"const2": const2},
                name=name,
            )


    node = CustomNode(const2=-1, name="custom_node")

This allow more flexibility, as you can redefine the complete behavior of
the node in the subclass. Be careful to expose the `name` parameter in the
subclass ``__init__``, and to pass it to the base class as parameter.
It is a good practice to find meaningful names for your node instances.

.. warning::
    All Node instances names must be unique !
    ReservoirPy will raise an exception if it is not the case.
    All node classes generate their own unique default names though.
