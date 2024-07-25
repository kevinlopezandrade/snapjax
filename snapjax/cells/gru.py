import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import State
from snapjax.cells.rnn import RNNStandard


class GRU(RNNStandard):
    """
    Gated recurrent unit from Engel[10] which allows
    I_t being sparse.
    """

    W_iz: nn.Linear
    W_hz: nn.Linear
    W_hr: nn.Linear
    W_ir: nn.Linear
    W_ia: nn.Linear
    W_ha: nn.Linear
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jrandom.split(key, 6)
        self.W_iz = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[0])
        self.W_hz = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[1])
        self.W_ir = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[2])
        self.W_hr = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[3])
        self.W_ia = nn.Linear(input_size, hidden_size, use_bias=False, key=keys[4])
        self.W_ha = nn.Linear(hidden_size, hidden_size, use_bias=True, key=keys[5])

        self.input_size = input_size
        self.hidden_size = hidden_size

    def f(self, state: State, input: Array) -> State:
        h = state
        x = input

        z = jax.nn.sigmoid(self.W_iz(x) + self.W_hz(h))
        r = jax.nn.sigmoid(self.W_ir(x) + self.W_hr(h))
        a = jax.nn.tanh(self.W_ia(x) + r * self.W_ha(h))

        h_new = (1 - z) * h + z * a

        return h_new
