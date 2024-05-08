from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from snapjax.cells.base import Jacobians, RTRLCell, RTRLLayer, State

"""
This implementation is taken from:
https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py

An only adapated to work with this repo.
"""


def matrix_init(key, shape, normalization: float = 1):
    return jax.random.normal(key=key, shape=shape) / normalization


def nu_init(key, shape, r_min, r_max):
    u = jax.random.uniform(key=key, shape=shape)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase):
    u = jax.random.uniform(key, shape=shape)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class Traces(eqx.Module):
    gamma_trace: Array
    lambda_trace: Array
    B_trace: Array


class LRU(RTRLCell):
    theta_log: Array
    nu_log: Array
    gamma_log: Array
    B_re: Array
    B_im: Array
    hidden_size: int = eqx.field(static=True)  # hidden state dimension
    input_size: int = eqx.field(static=True)  # input and output dimensions
    r_min: float = eqx.field(static=True)
    r_max: float = eqx.field(static=True)
    max_phase: float = eqx.field(static=True)
    complex_hidden_state: bool = eqx.field(static=True)
    custom_trace_update: bool = eqx.field(static=True)
    custom_grad_update: bool = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        *,
        key: PRNGKeyArray
    ):
        self.custom_grad_update = True
        self.custom_trace_update = True
        self.complex_hidden_state = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_min = r_min
        self.r_max = r_max
        self.max_phase = max_phase

        theta_key, nu_key, gamma_key, b_key = jrandom.split(key, 4)
        b_re_key, b_im_key = jrandom.split(b_key)

        self.theta_log = theta_init(
            theta_key, shape=(self.hidden_size,), max_phase=self.max_phase
        )
        self.nu_log = nu_init(
            nu_key, shape=(self.hidden_size,), r_min=self.r_min, r_max=self.r_max
        )
        self.gamma_log = gamma_log_init(gamma_key, (self.nu_log, self.theta_log))

        self.B_re = matrix_init(
            b_re_key,
            shape=(self.hidden_size, self.input_size),
            normalization=jnp.sqrt(2 * self.input_size),
        )

        self.B_im = matrix_init(
            b_im_key,
            shape=(self.hidden_size, self.input_size),
            normalization=jnp.sqrt(2 * self.input_size),
        )

    @staticmethod
    def get_diag_lambda(nu_log, theta_log):
        nu = jnp.exp(nu_log)
        theta = jnp.exp(theta_log)

        return jnp.exp(-nu + 1j * theta)

    def f(self, state: State, input: Array) -> State:
        diag_lambda = self.get_diag_lambda(self.nu_log, self.theta_log)

        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )

        h_out = diag_lambda * state + B_norm @ input

        return h_out

    def state_and_aux(self, state: State, input: Array):
        """
        Returns the value and the data necessary for every param to be updated later
        according to NicolasZucchet paper.
        """
        new_state = self.f(state, input)
        return new_state, {"h_prev": state, "input": input}

    def init_state(self) -> State:
        return 1j * jnp.zeros(shape=(self.hidden_size,))

    def make_zero_traces(self) -> Traces:
        """
        The traces for gamma, lambda, B
        """
        gamma_trace = jnp.zeros(shape=(self.hidden_size,)) * 1j
        lambda_trace = jnp.zeros(shape=(self.hidden_size,)) * 1j
        b_trace = jnp.zeros(shape=(self.hidden_size, self.input_size)) * 1j

        return Traces(
            gamma_trace=gamma_trace, lambda_trace=lambda_trace, B_trace=b_trace
        )

    def update_traces(self, prev_trace: Traces, aux: Dict[str, Any]):
        h_prev = aux["h_prev"]
        input = aux["input"]

        B = self.B_re + 1j * self.B_im
        gamma = jnp.exp(self.gamma_log)
        lambda_diag = self.get_diag_lambda(self.nu_log, self.theta_log)

        lambda_trace = lambda_diag * prev_trace.lambda_trace + h_prev
        gamma_trace = lambda_diag * prev_trace.gamma_trace + B @ input
        B_trace = jnp.diag(lambda_diag) @ prev_trace.B_trace + jnp.outer(gamma, input)

        return Traces(
            gamma_trace=gamma_trace, lambda_trace=lambda_trace, B_trace=B_trace
        )

    def update_grads(self, hidden_state_grad: Array, traces: Traces):
        """
        For every param of this cell updates its gradients.
        """
        delta_lambda = hidden_state_grad * traces.lambda_trace
        delta_gamma = (hidden_state_grad * traces.gamma_trace).real
        grad_B = jnp.diag(hidden_state_grad) @ traces.B_trace

        _, dl = jax.vjp(
            lambda nu_log, theta_log: self.get_diag_lambda(nu_log, theta_log),
            self.nu_log,
            self.theta_log,
        )
        grad_nu_log, grad_theta_log = dl(delta_lambda)
        grad_gamma_log = delta_gamma * jnp.exp(self.gamma_log)
        grad_B_re = grad_B.real
        grad_B_im = -grad_B.imag

        grads = eqx.tree_at(
            lambda model: (  # type: ignore
                model.nu_log,
                model.theta_log,
                model.gamma_log,
                model.B_re,
                model.B_im,
            ),
            self,
            (grad_nu_log, grad_theta_log, grad_gamma_log, grad_B_re, grad_B_im),
        )

        return grads


class LRULayer(RTRLLayer):
    cell: LRU
    C_re: Array
    C_im: Array
    D: Array

    d_inp: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)

    def __init__(self, cell: LRU, d_out: int, *, key: PRNGKeyArray):
        c_re_key, c_im_key, d_key = jrandom.split(key, 3)
        self.cell = cell
        self.d_inp = self.cell.input_size
        self.d_out = d_out

        self.C_re = matrix_init(
            c_re_key,
            shape=(self.d_out, self.cell.hidden_size),
            normalization=jnp.sqrt(self.cell.hidden_size),
        )

        self.C_im = matrix_init(
            c_im_key,
            shape=(self.d_out, self.cell.hidden_size),
            normalization=jnp.sqrt(self.cell.hidden_size),
        )
        self.D = matrix_init(
            d_key,
            shape=(self.d_out, self.d_inp),
            normalization=jnp.sqrt(self.d_inp),
        )

    def f(
        self,
        state: State,
        input: Array,
        perturbation: Array,
        sp_projection_cell: RTRLCell | None = None,
    ) -> Tuple[State, Jacobians, Array]:
        C = self.C_re + 1j * self.C_im

        h_out, aux = self.cell.state_and_aux(state, input)
        h_out = h_out + perturbation

        y = (C @ h_out).real + self.D @ input

        return h_out, aux, y

    def f_bptt(
        self,
        state: State,
        input: Array,
    ) -> Tuple[State, Array]:
        C = self.C_re + 1j * self.C_im

        h_out = self.cell.f(state, input)
        y = (C @ h_out).real + self.D @ input

        return h_out, y
