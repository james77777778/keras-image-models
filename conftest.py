import os


def pytest_configure():
    import tensorflow as tf

    # disable tensorflow gpu memory preallocation
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # disable jax gpu memory preallocation
    # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
