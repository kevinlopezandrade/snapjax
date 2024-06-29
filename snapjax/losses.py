import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree
from optax.tree_utils import tree_sum


def l2(y: Array, y_hat: Array, mask: float):
    return jnp.sum((y - y_hat) ** 2)


def cross_entropy(y: Array, y_hat: Array, mask: float):
    logits = jax.nn.log_softmax(y_hat)
    likelihood = jnp.sum(logits * y)
    return -1 * likelihood


def masked_quadratic(y: Array, y_hat: Array, mask: float):
    loss = mask * jnp.mean(0.5 * (y - y_hat) ** 2)
    return loss


def masked_l2(y: Array, y_hat: Array, mask: float):
    return mask * jnp.sum((y - y_hat) ** 2)


@jax.jit
def mean_squared_loss(target: Array, pred: Array, mask: float):
    return mask * 1 / 2 * jnp.mean((target - pred) ** 2)


@jax.jit
def l_infinity(target: Array, pred: Array, mask: float):
    return mask * jnp.max(jnp.abs(target - pred))


# Implementation from Optax.
def tree_l1_norm(tree):
    """Compute the l1 norm of a pytree.

    Args:
      tree: pytree.

    Returns:
      a scalar value.
    """
    abs_tree = jtu.tree_map(jnp.abs, tree)
    return tree_sum(abs_tree)


def get_l1_regularizer(penalty: float):
    @jax.jit
    def _regularizer(model: PyTree, model_prev: PyTree):
        diff = jtu.tree_map(lambda a, b: a - b, model, model_prev)
        norm = tree_l1_norm(diff)
        return penalty * norm

    return _regularizer
