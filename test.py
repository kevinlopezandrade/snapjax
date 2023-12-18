import timeit

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from algos import bptt, rtrl, sparse_multiplication
from rnn import StackedRNN

T = 10
num_layers = 4
hidden_size = 10
input_size = 10
key = jax.random.PRNGKey(7)

model = StackedRNN(
    key,
    num_layers=num_layers,
    hidden_size=hidden_size,
    input_size=input_size,
    use_bias=False,
)

inputs = jnp.ones(shape=(T, input_size), dtype=jnp.float32)
targets = jnp.zeros(shape=(T, input_size), dtype=jnp.float32)

loss, acc_grads, _ = rtrl(model, inputs, targets, use_snap_1=False)
loss_bptt, acc_grads_bptt = bptt(model, inputs, targets)

print("Comparing BPTT and RTRL")
print(loss, loss_bptt)

for (key_a, leaf_a), (key_b, leaf_b) in zip(
    jtu.tree_leaves_with_path(acc_grads), jtu.tree_leaves_with_path(acc_grads_bptt)
):
    if jnp.allclose(leaf_a, leaf_b):
        print("Match", jtu.keystr(key_a))
    else:
        print("Don't Match", jtu.keystr(key_a))
        print("Max difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))


print("Comparing SNAP1 Optimized and not Optimized")
loss_optmized, acc_grads_optmized, _ = rtrl(
    model, inputs, targets, matrix_product=sparse_multiplication, use_snap_1=True
)
loss_not_opmized, acc_grads_not_optmized, _ = rtrl(
    model, inputs, targets, use_snap_1=True
)

for (key_a, leaf_a), (key_b, leaf_b) in zip(
    jtu.tree_leaves_with_path(acc_grads_optmized),
    jtu.tree_leaves_with_path(acc_grads_not_optmized),
):
    if jnp.allclose(leaf_a, leaf_b):
        print("Match", jtu.keystr(key_a))
    else:
        print("Don't Match", jtu.keystr(key_a))
        print("Max difference: ", jnp.max(jnp.abs(leaf_a - leaf_b)))
