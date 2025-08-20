import numpy as np
import pytest

from reservoirpy.utils.data_validation import (
    check_model_input,
    check_model_timestep,
    check_multiseries,
    check_node_input,
    check_timeseries,
    check_timestep,
    is_model_input,
    is_model_timestep,
    is_multiseries,
    is_node_input,
    is_timeseries,
    is_timestep,
)


def test_check_timestep():
    x = np.array([1, 2, 3])
    check_timestep(x)

    x = np.array([1, 2, 3])
    check_timestep(x, expected_dim=3)

    with pytest.raises(TypeError):
        check_timestep([1, 2, 3])

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        check_timestep(x)

    x = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_timestep(x, expected_dim=5)


def test_is_timestep():
    x = np.array([1, 2, 3])
    assert is_timestep(x) is True

    x = np.array([[1, 2], [3, 4]])
    assert is_timestep(x) is False


def test_check_timeseries():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    check_timeseries(x)  # Should not raise

    x = np.array([[1, 2], [3, 4], [5, 6]])
    check_timeseries(x, expected_length=3, expected_dim=2)

    with pytest.raises(TypeError):
        check_timeseries([[1, 2], [3, 4]])

    x = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_timeseries(x)

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        check_timeseries(x, expected_length=5)

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        check_timeseries(x, expected_dim=5)


def test_is_timeseries_valid():
    x = np.array([[1, 2], [3, 4]])
    assert is_timeseries(x) == True

    x = np.array([1, 2, 3])
    assert is_timeseries(x) == False


def test_check_multiseries():
    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    check_multiseries(x)  # Should not raise

    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    check_multiseries(x, expected_length=2, expected_dim=2)

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_multiseries(x)  # Should not raise

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_multiseries(x, expected_length=2, expected_dim=2)

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValueError):
        check_multiseries(x, expected_length=5)

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    with pytest.raises(ValueError):
        check_multiseries(x, expected_dim=5)

    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        check_multiseries(x)

    with pytest.raises(TypeError):
        check_multiseries(2504)

    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    check_multiseries(x)

    # Test that it infers expected_dim from first series
    ts3 = np.array([[9, 10, 11], [12, 13, 14]])  # 3 features - should fail
    x_inconsistent = [ts1, ts3]
    with pytest.raises(ValueError):
        check_multiseries(x_inconsistent)

    x_invalid_type = (
        "According to all known laws of aviation, there is no way a bee should"
        "be able to fly. Its wings are too small to get its fat little body off"
        "the ground. The bee, of course, flies anyway. Because bees don't care"
        "what humans think is impossible. Yellow, black. Yellow, black."
    )
    with pytest.raises(TypeError):
        check_multiseries({x_invalid_type})


def test_is_multiseries():
    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    assert is_multiseries(x) == True

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert is_multiseries(x) == True

    x = np.array([[1, 2], [3, 4]])
    assert is_multiseries(x) == False


def test_check_node_input():
    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    check_node_input(x)

    x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    check_node_input(x)

    x = np.array([[1, 2], [3, 4]])
    check_node_input(x)

    x = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        check_node_input(x)

    with pytest.raises(TypeError):
        check_node_input(12)

    ts1 = np.array([[1, 2], [3, 4]])  # 2 features
    ts2 = np.array([[5, 6], [7, 8]])  # 2 features
    x = [ts1, ts2]
    check_node_input(x)

    # Test dimension consistency enforcement
    ts3 = np.array([[9, 10, 11], [12, 13, 14]])  # 3 features
    x_inconsistent = [ts1, ts3]
    with pytest.raises(ValueError):
        check_node_input(x_inconsistent)


def test_is_node_input():
    x = np.array([[1, 2], [3, 4]])
    assert is_node_input(x) == True

    x = np.array([1, 2, 3])
    assert is_node_input(x) == False


def test_check_model_timestep():
    """Test check_model_timestep with mapping input."""
    x1 = np.array([1, 2])
    x2 = np.array([3, 4])
    x = {"node1": x1, "node2": x2}
    check_model_timestep(x)

    x1 = np.array([1, 2])
    x2 = np.array([3, 4])
    x = {"node1": x1, "node2": x2}
    check_model_timestep(x, expected_inputs=["node1", "node2"])

    x1 = np.array([1, 2])
    x = {"node1": x1}
    with pytest.raises(ValueError):
        check_model_timestep(x, expected_inputs=["node2"])

    x1 = np.array([1, 2])
    x2 = np.array([3, 4, 5])
    x = {"node1": x1, "node2": x2}
    expected_dim = {"node1": 2, "node2": 3}
    check_model_timestep(x, expected_dim=expected_dim)

    x1 = np.array([1, 2])
    x2 = np.array([3, 4])
    x = {"node1": x1, "node2": x2}
    check_model_timestep(x, expected_dim=2)

    x = np.array([1, 2, 3])
    check_model_timestep(x)

    x = np.array([1, 2])
    expected_dim = {"node1": 2}
    check_model_timestep(x, expected_dim=expected_dim)

    x = np.array([1, 2])
    check_model_timestep(x, expected_dim=2)

    with pytest.raises(TypeError):
        check_model_timestep(12)


def test_is_model_timestep():
    x = np.array([1, 2, 3])
    assert is_model_timestep(x) == True

    assert is_model_timestep(12) == False


def test_check_model_input():
    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = {"node1": ts1, "node2": ts2}
    check_model_input(x)

    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6, 7], [8, 9, 10]])
    x = {"node1": ts1, "node2": ts2}
    expected_dim = {"node1": 2, "node2": 3}
    check_model_input(x, expected_dim=expected_dim)

    x = np.array([[1, 2], [3, 4]])
    check_model_input(x)

    ts1 = np.array([[1, 2], [3, 4]])
    ts2 = np.array([[5, 6], [7, 8]])
    x = [ts1, ts2]
    check_model_input(x)

    x = np.array([[1, 2], [3, 4]])
    expected_dim = {"node1": 2}
    check_model_input(x, expected_dim=expected_dim)

    x = np.array([[1, 2], [3, 4]])
    expected_dim = {"node1": 2, "node2": 3}
    with pytest.raises(TypeError):
        check_model_input(x, expected_dim=expected_dim)

    with pytest.raises(TypeError):
        check_model_input(12)


def test_is_model_input():
    x = np.array([[1, 2], [3, 4]])
    assert is_model_input(x) == True

    assert is_model_input(12) == False
