import os


def pytest_configure():
    # disable jax gpu memory preallocation
    # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
