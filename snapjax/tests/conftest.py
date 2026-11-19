"""Configure the numerical test environment before importing test modules."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
