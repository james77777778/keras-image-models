import os

import pytest
from keras import backend


def pytest_addoption(parser):
    parser.addoption(
        "--run_serialization",
        action="store_true",
        default=False,
        help="run serialization tests",
    )


def pytest_configure(config):
    import tensorflow as tf

    # disable tensorflow gpu memory preallocation
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # disable jax gpu memory preallocation
    # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    config.addinivalue_line(
        "markers", "serialization: mark test as a serialization test"
    )
    config.addinivalue_line(
        "markers",
        "requires_trainable_backend: mark test for trainable backend only",
    )


def pytest_collection_modifyitems(config, items):
    run_serialization_tests = config.getoption("--run_serialization")
    skip_serialization = pytest.mark.skipif(
        not run_serialization_tests,
        reason="need --run_serialization option to run",
    )
    requires_trainable_backend = pytest.mark.skipif(
        backend.backend() == "numpy", reason="require trainable backend"
    )
    for item in items:
        if "requires_trainable_backend" in item.keywords:
            item.add_marker(requires_trainable_backend)
        if "serialization" in item.name:
            item.add_marker(skip_serialization)
