from abc import abstractmethod
from functools import partial
from typing import Protocol, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from jaxtyping import Array, PRNGKeyArray


class OnlineDataSet(Protocol):
    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Tuple[Array, Array, Array]:
        pass


class LinearSystem:
    def __init__(
        self,
        key: PRNGKeyArray,
        d_input: int = 1,
        d_hidden: int = 10,
        d_output: int = 1,
        T: int = 100,
        parametrization: str = "diagonal_complex",
        input_type: str = "gaussian",
        input_mean: float = 0.0,
        input_std: float = 1.0,
        noise_std: float = 0.0,
        normalization: str = "none",
        **kwargs
    ):
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.input_mean = input_mean
        self.input_std = input_std
        self.noise_std = noise_std
        self.input_type = input_type
        self.T = T

        if parametrization == "diagonal_complex":
            # Diagonal complex parametrization of the state space model
            # Enables controlling min and max norm of the eigenvalues of A
            key_nu, key_theta, key_B, key_C, key_D = jax.random.split(key, 5)
            nu = jax.random.uniform(
                key_nu,
                shape=(d_hidden,),
                minval=kwargs["min_nu"],
                maxval=kwargs["max_nu"],
            )
            theta = jax.random.uniform(
                key_theta,
                shape=(d_hidden,),
                minval=-kwargs["max_phase"],
                maxval=kwargs["max_phase"],
            )
            self.A = jnp.eye(d_hidden) * jnp.diag(nu * jnp.exp(1j * theta))
            self.B = jax.random.normal(
                key_B, shape=(d_hidden, d_input), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.C = jax.random.normal(
                key_C, shape=(d_output, d_hidden), dtype=jnp.complex64
            ) / jnp.sqrt(2 * d_hidden)
            self.D = jax.random.normal(
                key_D, shape=(d_output, d_input), dtype=jnp.float32
            ) / jnp.sqrt(d_output)
        elif parametrization == "diagonal_real":
            # Same as diagonal complex, but restrict the eigenvalues to be symmetric
            raise NotImplementedError
        elif parametrization == "controllable":
            # Controllable canonical form of the state space model
            # https://en.wikipedia.org/wiki/State-space_representation#Canonical_realizations
            assert d_input == 1
            assert d_output == 1
            raise NotImplementedError
        elif parametrization == "dense":
            # Vanilla parametrization of the state space model
            # All parameters are sampled from a normal distribution
            key_A, key_B, key_C, key_D = jax.random.split(key, 4)
            self.A = jax.random.normal(key_A, shape=(d_hidden, d_hidden)) / jnp.sqrt(
                d_hidden
            )
            with jax.default_device(
                jax.devices("cpu")[0]
            ):  # eigval decomposition has to be on CPU
                l, V = jnp.linalg.eig(self.A)
                l = (
                    kwargs["min_nu"]
                    + (kwargs["max_nu"] - kwargs["min_nu"]) * jax.nn.tanh(jnp.abs(l))
                ) * jnp.exp(1.0j * jnp.angle(l) * kwargs["max_phase"] / jnp.pi)
                self.A = (V @ jnp.diag(l) @ jnp.linalg.inv(V)).real
            self.B = jax.random.normal(key_B, shape=(d_hidden, d_input)) / jnp.sqrt(
                d_input
            )
            self.C = jax.random.normal(key_C, shape=(d_output, d_hidden)) / jnp.sqrt(
                d_hidden
            )
            self.D = jax.random.normal(key_D, shape=(d_output, d_input)) / jnp.sqrt(
                d_input
            )

            # If needed, normalize B to make sure no eigenvalues dominate the output
            if normalization == "L2":
                self.B = (
                    V
                    @ jnp.diag(jnp.sqrt(1 - jnp.abs(l) ** 2))
                    * jnp.linalg.inv(V)
                    @ self.B
                )
                self.B = self.B.real
            elif normalization == "L1":
                self.B = V @ jnp.diag(1 - jnp.abs(l)) * jnp.linalg.inv(V) @ self.B
                self.B = self.B.real
        else:
            raise ValueError

        # If we don't use B, C, D, we set them to Id, Id, 0
        if not kwargs["use_B_C_D"]:
            assert d_input == d_hidden
            assert d_hidden == d_output
            self.B = jnp.eye(d_hidden)
            self.C = jnp.eye(d_output)
            self.D = jnp.zeros_like(self.D)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key):
        key_inputs, key_noise = jax.random.split(key, 2)
        inputs = (
            self.input_mean
            + jax.random.normal(key_inputs, shape=(self.T, self.d_input))
            * self.input_std
        )
        if self.input_type == "gaussian":
            inputs = inputs
        elif self.input_type == "constant":
            inputs = jnp.ones_like(inputs) * inputs[0]
        noise = self.noise_std * jax.random.normal(
            key_noise, shape=(self.T, self.d_output)
        )

        def _step(h, x):
            new_h = self.A @ h + self.B @ x
            return new_h, new_h

        init_hiddens = jnp.zeros((self.d_hidden,), dtype=self.A.dtype)
        hiddens = jax.lax.scan(_step, init_hiddens, inputs)[1]
        outputs = jax.vmap(lambda h, x, n: (self.C @ h).real + self.D @ x + n)(
            hiddens, inputs, noise
        )
        masks = jnp.ones(outputs.shape[:-1])
        return inputs, outputs, masks

    def stats_lambdas(self):
        with jax.default_device(jax.devices("cpu")[0]):
            eigvals = jnp.linalg.eigvals(self.A)
            return jnp.abs(eigvals).min(), jnp.abs(eigvals).max()


class FlipFlop:
    def __init__(
        self,
        dt: float = 0.5,
        t_max: float = 50,
        fixation_duration: float = 1,
        stimulus_duration: float = 1,
        decision_delay_duration: float = 5,
        stim_delay_duration_min: float = 5,
        stim_delay_duration_max: float = 25,
        input_amp: float = 1.0,
        target_amp: float = 0.5,
        fixate: float = False,
    ):
        self.dt = dt
        self.t_max = t_max
        self.fixation_duration = fixation_duration
        self.stimulus_duration = stimulus_duration
        self.decision_delay_duration = decision_delay_duration
        self.stim_delay_duration_min = stim_delay_duration_min
        self.stim_delay_duration_max = stim_delay_duration_max
        self.input_amp = input_amp
        self.target_amp = target_amp
        self.fixate = fixate

        self.inp_dim = 2
        self.out_dim = 2

    def sample(self, key: PRNGKeyArray):
        fixation_duration_discrete = int(self.fixation_duration / self.dt)
        stimulus_duration_discrete = int(self.stimulus_duration / self.dt)
        decision_delay_duration_discrete = int(self.decision_delay_duration / self.dt)
        n_t_max = int(self.t_max / self.dt)
        choices = np.array([0, 1])
        signs = np.array([-1, 1])

        input_samp = np.zeros((n_t_max, 2))
        target_samp = np.zeros((n_t_max, 2))
        mask_samp = np.zeros((n_t_max,), dtype=np.uint8)

        idx_t = fixation_duration_discrete

        if self.fixate:
            # Mask
            mask_samp[:idx_t] = 1

        while True:
            # Interval until next pulse.
            key, interval_key, channel_key, sign_key = jrandom.split(key, 4)
            interval = jrandom.uniform(
                key=interval_key,
                minval=self.stim_delay_duration_min,
                maxval=self.stim_delay_duration_max,
            )
            interval += self.decision_delay_duration

            # Next pulse start index.
            n_t_interval = int(interval / self.dt)
            idx_tp1 = idx_t + n_t_interval

            channel = jrandom.choice(channel_key, choices).item()
            sign = jrandom.choice(sign_key, signs).item()

            # Input
            input_samp[idx_t : idx_t + stimulus_duration_discrete, channel] = (
                sign * self.input_amp
            )
            # Target
            target_samp[idx_t + decision_delay_duration_discrete : idx_tp1, channel] = (
                sign * self.target_amp
            )
            # Mask
            mask_samp[idx_t + decision_delay_duration_discrete : idx_tp1] = 1
            # Update
            idx_t = idx_tp1

            if idx_t > n_t_max:
                break

        return jnp.array(input_samp), jnp.array(target_samp), jnp.array(mask_samp)


class Romo:
    def __init__(
        self,
        dt: float,
        fixation_duration: float = 3,
        stimulus_duration: float = 1,
        decision_delay_duration: float = 5,
        decision_duration: float = 10,
        stim_delay_duration_min: float = 2,
        stim_delay_duration_max: float = 8,
        input_amp_min: float = 0.5,
        input_amp_max: float = 1.5,
        min_input_diff: float = 0.2,
        target_amp: float = 0.5,
        fixate: bool = True,
        return_ts: bool = False,
        test: bool = False,
        original_variant: bool = False,
    ):
        self.dt = dt
        self.fixation_duration = fixation_duration
        self.stimulus_duration = stimulus_duration
        self.decision_delay_duration = decision_delay_duration
        self.decision_duration = decision_duration
        self.stim_delay_duration_min = stim_delay_duration_min
        self.stim_delay_duration_max = stim_delay_duration_max
        self.input_amp_min = input_amp_min
        self.input_amp_max = input_amp_max
        self.min_input_diff = min_input_diff
        self.target_amp = target_amp
        self.fixate = fixate
        self.return_ts = return_ts
        self.test = test
        self.original_variant = original_variant

    def sample(self, key: PRNGKeyArray):
        if self.original_variant:
            dim_out = 1
        else:
            dim_out = 2

        # Task times
        fixation_duration_discrete = int(self.fixation_duration / self.dt)
        stimulus_duration_discrete = int(self.stimulus_duration / self.dt)
        decision_delay_duration_discrete = int(self.decision_delay_duration / self.dt)
        decision_duration_discrete = int(self.decision_duration / self.dt)
        stim_0_begin = fixation_duration_discrete
        stim_0_end = stim_0_begin + stimulus_duration_discrete
        # After stim_1, there is a random delay...
        t_max = (
            self.fixation_duration
            + 2 * self.stimulus_duration
            + self.stim_delay_duration_max
            + self.decision_delay_duration
            + self.decision_duration
        )
        n_t_max = int(t_max / self.dt)

        # Input and target sequences
        input_samp = np.zeros((n_t_max, 1))
        target_samp = np.zeros((n_t_max, dim_out))
        mask_samp = np.zeros((n_t_max,), dtype=np.uint8)

        key, stim_delay_key = jrandom.split(key)
        stim_delay = jrandom.uniform(
            stim_delay_key,
            minval=self.stim_delay_duration_min,
            maxval=self.stim_delay_duration_max,
        )
        # Set indices
        stim_delay_discrete = int(stim_delay / self.dt)
        stim_1_begin = stim_0_end + stim_delay_discrete
        stim_1_end = stim_1_begin + stimulus_duration_discrete
        response_begin = stim_1_end + decision_delay_duration_discrete
        response_end = response_begin + decision_duration_discrete

        while True:
            input_amps = jrandom.uniform(
                key, minval=self.input_amp_min, maxval=self.input_amp_max, shape=(2,)
            )
            input_diff = jnp.abs(input_amps[0] - input_amps[1])
            if input_diff >= self.min_input_diff:
                break
            else:
                key, _ = jrandom.split(key)

        larger_input = np.argmax(input_amps)

        # Set input, target
        input_samp[stim_0_begin:stim_0_end] = input_amps[0]
        input_samp[stim_1_begin:stim_1_end] = input_amps[1]
        if self.original_variant:
            target_sign = (-1) ** larger_input
            target_samp[response_begin:response_end] = target_sign * self.target_amp
        else:
            target_samp[response_begin:response_end, larger_input] = self.target_amp
        # Mask
        mask_samp[response_begin:response_end] = 1

        if self.fixate:
            # Set target output to zero until the decision delay
            mask_samp[:stim_1_end] = 1

        return jnp.array(input_samp), jnp.array(target_samp), jnp.array(mask_samp)


class OnlineRegressionDataLoader:
    def __init__(
        self,
        dataset: OnlineDataSet,
        key: PRNGKeyArray,
        size: int,
        copy_mask: bool = False,
        unstack: bool = False,
    ):
        self.dataset = dataset
        self.key = key
        self.size = size
        self.unstack = unstack
        self.copy_mask = copy_mask

    def __iter__(self):
        # Sample a new random seed everytime we initialize an iterator
        self.key, _ = jax.random.split(self.key)
        for key in jax.random.split(self.key, self.size):
            inputs, targets, masks = self.dataset.sample(key)
            if self.unstack:
                if self.copy_mask:
                    yield inputs._unstack(), targets._unstack(), masks._unstack(), jax.device_get(
                        masks
                    )
                else:
                    yield inputs._unstack(), targets._unstack(), masks._unstack()
            else:
                yield inputs, targets, masks

    def __len__(self):
        return self.size
