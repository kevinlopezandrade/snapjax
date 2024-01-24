import jax
import jax.numpy as jnp
from jaxtyping import Array


def l2(y: Array, y_hat: Array):
    return jnp.sum((y - y_hat) ** 2)


def cross_entropy(y: Array, y_hat: Array):
    logits = jax.nn.log_softmax(y_hat)
    likelihood = jnp.sum(logits * y)
    return -1 * likelihood
