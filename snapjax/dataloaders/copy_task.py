from collections.abc import Iterator
from enum import Enum
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, PRNGKeyArray


class CopyState(Enum):
    EmitToken = 0
    EmitBlank = 1
    EmitStart = 2
    EmitRepeat = 3
    EmitEnd = 4


def token_to_str(token: Array):
    K = token.shape[0] - 2
    k = jnp.argwhere(token, size=1)[0][0]

    if k == K:
        return "[B]"
    if k == K + 1:
        return "[S]"

    return f"[{k}]"


def generate_single_sequence(
    K: int, S: int, T: int, key: PRNGKeyArray
) -> Iterator[Tuple[Array, Array]]:
    """
    Its basically a finite state machine that yields input, output
    pairs. According to the classical description of the copy task.
    Tokens are one hot encoded.
    """
    blank = K
    start = K + 1

    keys = jrandom.split(key, num=int(S))
    state = CopyState.EmitToken
    s = 0
    t = 0

    # Preallocate the choices.
    positions = [jrandom.choice(key, K) for key in keys]

    # Matrix of possible one-hot encodings.
    tokens = np.eye(K + 2)

    while state != CopyState.EmitEnd:
        # Emit sequence.
        if state == CopyState.EmitToken:
            pos = positions[s]
            yield tokens[pos], tokens[blank]
            s += 1
            if not (s < S):
                state = CopyState.EmitBlank
                s = 0

        # Emit blank tokens.
        if state == CopyState.EmitBlank:
            if not (t < T - 1):
                state = CopyState.EmitStart
                t = 0
            else:
                t += 1
                yield tokens[blank], tokens[blank]

        # Emit START token.
        if state == CopyState.EmitStart:
            yield tokens[start], tokens[blank]
            state = CopyState.EmitRepeat

        # Emit Repeated Tokens
        if state == CopyState.EmitRepeat:
            pos = positions[s]
            yield tokens[blank], tokens[pos]
            s += 1
            if not (s < S):
                state = CopyState.EmitEnd


def copy_seq_size(S: int, T: int):
    return 2 * S + T


def gen_copy_seqs(
    N: int, K: int, S: int, T: int, key: PRNGKeyArray
) -> Iterator[Tuple[bool, Array, Array]]:
    """
    Args:
        N: Number of sequences to generate.
        K: Size of the vocabulary from [0, K-1]
        S: Size of the sequence to remember.
        T: Size of the blanks + start token.
        key: Random key to generate the sequences.

    Returns:
        Iterator that goes through the N sequences, returning
        a tuple where the first element indicates if its the
        end of a sequence or not, the second element is the
        input and the third one is the target.

    """
    with jax.default_device(jax.devices("cpu")[0]):
        keys = jrandom.split(key, N)
        size = copy_seq_size(S, T)
        for key in keys:
            for i, (inp, target) in enumerate(
                generate_single_sequence(K=K, S=S, T=T, key=key), 1
            ):
                if i < size:
                    yield False, inp, target
                else:
                    yield True, inp, target


def gen_rand_delay_copy_seqs(
    N: int, K: int, S: int, min_T: int, max_T: int, key: PRNGKeyArray
):
    keys = jrandom.split(key, N)
    for key in keys:
        T_key, seq_key = jrandom.split(key, 2)
        T = min_T + jrandom.choice(T_key, (max_T - min_T) + 1)
        size = copy_seq_size(S, T)
        for i, (inp, target) in enumerate(
            generate_single_sequence(K=K, S=S, T=T, key=seq_key), 1
        ):
            if i < size:
                yield False, inp, target
            else:
                yield True, inp, target


def gen_rand_copy_seqs(
    N: int, K: int, min_S: int, max_S: int, min_T: int, max_T: int, key: PRNGKeyArray
):
    keys = jrandom.split(key, N)
    for key in keys:
        T_key, S_key, seq_key = jrandom.split(key, 3)
        S = min_S + jrandom.choice(S_key, (max_S - min_S) + 1)
        T = min_T + jrandom.choice(T_key, (max_T - min_T) + 1)
        size = copy_seq_size(S, T)
        for i, (inp, target) in enumerate(
            generate_single_sequence(K=K, S=S, T=T, key=seq_key), 1
        ):
            if i < size:
                yield False, inp, target
            else:
                yield True, inp, target


def reshape_to_batch(tokens: Array, bs: int, S: int, T: int):
    return tokens.reshape(bs, copy_seq_size(S, T), -1)


def gen_batch_copy_seqs(N: int, bs: int, K: int, S: int, T: int, key: PRNGKeyArray):
    keys = jrandom.split(key, N)
    for key in keys:
        tokens = list(gen_copy_seqs(bs, K=K, S=S, T=T, key=key))
        inp_batch = np.stack([token[1] for token in tokens])
        inp_batch = reshape_to_batch(inp_batch, bs=bs, S=S, T=T)

        out_batch = np.stack([token[2] for token in tokens])
        out_batch = reshape_to_batch(out_batch, bs=bs, S=S, T=T)

        yield inp_batch, out_batch
