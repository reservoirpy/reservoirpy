import os
import tempfile

import reservoirpy as rpy


def test_save_load():
    tmpdir = tempfile.gettempdir()
    from reservoirpy.nodes import Reservoir, Ridge

    node = Reservoir(10)
    model = node >> Ridge()

    rpy.save(node, os.path.join(tmpdir, "my_node"))
    node2 = rpy.load(os.path.join(tmpdir, "my_node.rpy"))
    assert node.units == node2.units

    rpy.save(model, os.path.join(tmpdir, "my_model.rpy"))
    model2 = rpy.load(os.path.join(tmpdir, "my_model"))
    assert model.nodes[0].units == model2.nodes[0].units
