import timeit

import jax
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.tests.utils import get_random_batch, get_stacked_rnn

T = 1000
B = 8
L = 4
H = 256

model = get_stacked_rnn(L, H, H)
print("Computing the sparse tree")
sp_projection_tree = model.get_sp_projection_tree()
inp = get_random_batch(B, T, model)
out = get_random_batch(B, T, model)

rtrl_f = jtu.Partial(
    rtrl,
    use_snap_1=True,
    use_scan=True,
)


batched_rtrl = jax.jit(jax.vmap(rtrl_f, in_axes=(None, 0, 0, None)))
print("Compiling")
batched_rtrl(model, inp, out, sp_projection_tree)
print("Finished compiling")

res = timeit.timeit(
    lambda: jax.block_until_ready(batched_rtrl(model, inp, out, sp_projection_tree)),
    number=1,
)
print(res)
