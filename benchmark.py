import timeit

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.nn import make_with_state

from algos import bptt, rtrl, sparse_multiplication
from rnn import StackedRNN

T = 100
num_layers = 4
hidden_size = 100
input_size = 100
key = jax.random.PRNGKey(7)

model, inmediate_jacobians_state = make_with_state(StackedRNN)(
    key,
    num_layers=num_layers,
    hidden_size=hidden_size,
    input_size=input_size,
    use_bias=False,
)

inputs = jnp.ones(shape=(T, input_size), dtype=jnp.float32)
outputs = jnp.zeros(shape=(T, input_size), dtype=jnp.float32)


print("Test call to compile jit functions.")
rtrl(
    model,
    inputs,
    inmediate_jacobians_state,
    outputs,
    matrix_product=sparse_multiplication,
    use_snap_1=False,
)

print("Starting benchmark")
repeats = 1
print("Running no optimization")
res_1 = timeit.timeit(
    lambda: jax.block_until_ready(
        rtrl(model, inputs, inmediate_jacobians_state, outputs, use_snap_1=True)
    ),
    number=repeats,
)

print("Running with optimization")
res_2 = timeit.timeit(
    lambda: jax.block_until_ready(
        rtrl(
            model,
            inputs,
            inmediate_jacobians_state,
            outputs,
            matrix_product=sparse_multiplication,
            use_snap_1=True,
        )
    ),
    number=repeats,
)

print(res_1)
print(res_2)
