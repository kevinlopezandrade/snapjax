import timeit

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from snapjax.algos import rtrl
from snapjax.sp_jacrev import make_jacobian_projection
from snapjax.tests.utils import get_random_batch, get_stacked_rnn

T = 1000
B = 8
L = 4
H = 256

model = get_stacked_rnn(L, H, H)
print("Computing the sparse tree")
jacobian_mask = model.get_snap_n_mask(1)
jacobian_projection = make_jacobian_projection(jacobian_mask)
print("Done")

inp = get_random_batch(B, T, model)
out = get_random_batch(B, T, model)
mask = jnp.ones(out.shape)

rtrl_f = jtu.Partial(
    rtrl,
    use_scan=True,
)


batched_rtrl = jax.jit(jax.vmap(rtrl_f, in_axes=(None, 0, 0, 0, None, None)))
print("Compiling the function.")
batched_rtrl(model, inp, out, mask, jacobian_mask, jacobian_projection)
print("Finished compiling")

res = timeit.timeit(
    lambda: jax.block_until_ready(
        batched_rtrl(model, inp, out, mask, jacobian_mask, jacobian_projection)
    ),
    number=1,
)
print(res) # 1.26 approx in GTX 3090
