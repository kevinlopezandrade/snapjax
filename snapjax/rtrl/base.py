from functools import partial
from typing import Callable, List, Sequence, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from chex import Scalar
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from snapjax.algos import make_init_state, make_perturbations, make_zeros_grads
from snapjax.cells.base import (
    ArrayTree,
    RTRLLayer,
    RTRLStacked,
    Stacked,
    State,
    Traces,
    is_rtrl_cell,
)
from snapjax.losses import l2
from snapjax.sp_jacrev import DenseProjection, SparseProjection, standard_jacobian


@partial(jax.jit, static_argnums=(9, 10))
def step_loss(
    model_spatial: RTRLStacked,
    perturbations: Stacked[Array],
    model_rtrl: RTRLStacked,
    h_prev: Array,
    x_t: Array,
    y_t: Array,
    mask: float,
    jacobian_projection: RTRLStacked | None = None,
    model_prev: RTRLStacked | None = None,
    loss_func: Callable[[Array, Array, float], Array] = l2,
    regularizer: Callable[[RTRLStacked, RTRLStacked], float] | None = None,
):
    model = eqx.combine(model_spatial, model_rtrl)
    h_t, inmediate_jacobians, y_hat = model.f_rtrl(
        h_prev, x_t, perturbations, jacobian_projection
    )

    res = loss_func(y_t, y_hat, mask)
    if regularizer and model_prev:
        res = res + regularizer(model, model_prev)

    return res, (h_t, y_hat, inmediate_jacobians)


def _sparse_jacobian(leaf: SparseProjection | DenseProjection):
    # Jacobians are saved as the tranpose jacobians.
    if isinstance(leaf, DenseProjection):
        return standard_jacobian(jnp.zeros(leaf.jacobian_shape))

    zeros = jnp.zeros(leaf.sparse_def.nse)
    indices = leaf.sparse_def.indices_csc[:, ::-1]  # For the transpose.
    structure = (zeros, indices)
    return BCOO(
        structure,
        shape=leaf.sparse_def.jacobian_shape[::-1],  # For the transpose.
        # This is guaranteed since csc is sorted by colum and we transpose.
        indices_sorted=True,
        unique_indices=True,
    )


class RTRL(eqx.Module):
    loss_func: Callable = eqx.field(static=True)
    use_scan: bool = eqx.field(static=True)
    regularizer: Callable = eqx.field(static=True)
    return_hidden_state_grads: bool = eqx.field(static=True)

    def __init__(
        self,
        loss_func: Callable[[Array, Array, Array], Scalar],
        use_scan: bool = True,
        regularizer: Callable | None = None,
        return_hidden_state_grads: bool = False,
    ):
        self.loss_func = loss_func
        self.use_scan = use_scan
        self.regularizer = regularizer
        self.return_hidden_state_grads = return_hidden_state_grads

    @jax.jit
    def step(
        self,
        model: RTRLStacked,
        jacobians_prev: RTRLStacked,
        h_prev: Stacked[State],
        input: Array,
        target: Array,
        mask: float = 1.0,
        jacobian_mask: RTRLStacked | None = None,
        jacobian_projection: RTRLStacked | None = None,
        model_prev: RTRLStacked | None = None,
    ):
        theta_rtrl, theta_spatial = eqx.partition(
            model,
            lambda leaf: is_rtrl_cell(leaf),
            is_leaf=is_rtrl_cell,
        )
        step_loss_and_grad = jax.value_and_grad(
            jtu.Partial(
                step_loss, loss_func=self.loss_func, regularizer=self.regularizer
            ),
            argnums=(0, 1),
            has_aux=True,
        )
        perturbations = make_perturbations(theta_rtrl)

        (loss_t, aux), (grads) = step_loss_and_grad(
            theta_spatial,
            perturbations,
            theta_rtrl,
            h_prev,
            input,
            target,
            mask,
            jacobian_projection=jacobian_projection,
            model_prev=model_prev,
        )

        h_t, y_hat, inmediate_jacobians = aux
        spatial_grads, hidden_states_grads = grads

        jacobians = self.update_traces(
            jacobians_prev,
            inmediate_jacobians,
            theta_rtrl,
            jacobian_mask=jacobian_mask,
        )

        grads = self.update_grads(
            spatial_grads, hidden_states_grads, jacobians, theta_rtrl
        )

        # Reshape the flattened gradients of the RTRL cells.
        grads = jtu.tree_map(
            lambda grad, mat: (
                grad if isinstance(mat, BCOO) else grad.reshape(mat.shape)
            ),
            grads,
            model,
            is_leaf=lambda node: isinstance(node, BCOO),
        )

        h_t = cast(Stacked[State], h_t)
        grads = cast(RTRLStacked, grads)
        jacobians = cast(RTRLStacked, jacobians)
        loss_t = cast(float, loss_t)
        y_hat = cast(Array, y_hat)

        if self.return_hidden_state_grads:
            return h_t, grads, jacobians, loss_t, y_hat, hidden_states_grads
        else:
            return h_t, grads, jacobians, loss_t, y_hat

    def rtrl(
        self,
        model: RTRLStacked,
        inputs: Array,
        targets: Array,
        mask: Array,
        jacobian_mask: RTRLStacked | None = None,
        jacobian_projection: RTRLStacked | None = None,
    ):
        def forward_repack(carry, data):
            input, target, mask = data
            h_prev, acc_grads, jacobians_prev, acc_loss = carry

            out = self.step(
                model,
                jacobians_prev,
                h_prev,
                input,
                target,
                mask,
                jacobian_mask=jacobian_mask,
                jacobian_projection=jacobian_projection,
            )
            h_t, grads, jacobians_t, loss_t, y_hat = out
            acc_loss = acc_loss + loss_t
            acc_grads = jtu.tree_map(
                lambda acc_grads, grads: acc_grads + grads, acc_grads, grads
            )

            return (h_t, acc_grads, jacobians_t, acc_loss), y_hat

        h_init: Sequence[State] = make_init_state(model)
        acc_grads: RTRLStacked = make_zeros_grads(model)

        zero_jacobians = self.init_traces(model, jacobian_projection)

        acc_loss = 0.0

        if self.use_scan:
            carry_T, y_hats = jax.lax.scan(
                forward_repack,
                init=(h_init, acc_grads, zero_jacobians, acc_loss),
                xs=(inputs, targets, mask),
            )
        else:
            carry = (h_init, acc_grads, zero_jacobians, acc_loss)
            T = inputs.shape[0]
            y_hats = [None] * T
            for i in range(inputs.shape[0]):
                carry, y_hat = forward_repack(carry, (inputs[i], targets[i], mask[i]))
                y_hats[i] = y_hat

            y_hats = jnp.stack(y_hats)
            carry_T = carry

        h_T, acc_grads, jacobians_T, acc_loss = carry_T

        acc_loss = cast(float, acc_loss)
        y_hats = cast(Array, y_hats)
        return acc_loss, acc_grads, y_hats

    @staticmethod
    def init_traces(model: RTRLStacked, jacobian_projection: RTRLStacked): ...

    @staticmethod
    def update_traces(
        prev_traces: RTRLStacked,
        payloads: List[ArrayTree],
        theta_rtrl: RTRLStacked,
        jacobian_mask: RTRLStacked | None = None,
    ): ...

    @staticmethod
    def update_grads(
        grads: RTRLStacked,
        hidden_states_grads: List[ArrayTree],
        jacobians: RTRLStacked | Traces,
        theta_rtrl: RTRLStacked,
    ): ...


class RTRLApprox(RTRL):
    @staticmethod
    @jax.jit  # type: ignore
    def init_traces(model: RTRLStacked, jacobian_projection: RTRLStacked | None = None):

        cells_traces = eqx.filter(
            model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell
        )

        layer_index = 0
        n_layers = model.num_rtrl_layers()
        for i, layer in enumerate(model.layers):
            if isinstance(layer, RTRLLayer):
                cell = layer.cell
                all_traces = []
                for _ in range(layer_index, layer_index + 1):
                    if not jacobian_projection:
                        traces = cell.init_traces()
                    else:
                        traces = jtu.tree_map(
                            lambda leaf: _sparse_jacobian(leaf),
                            jacobian_projection.layers[i].cell,
                            is_leaf=lambda node: isinstance(
                                node, (SparseProjection, DenseProjection)
                            ),
                        )

                    all_traces.append(traces)

                cells_traces = eqx.tree_at(
                    lambda cells: cells.layers[i].cell,
                    cells_traces,
                    tuple(all_traces),
                )
                layer_index += 1

        return cells_traces

    @staticmethod
    def update_traces(
        prev_traces: RTRLStacked,  # Also known as the trace
        payloads: List[ArrayTree],
        theta_rtrl: RTRLStacked,
        jacobian_mask: RTRLStacked | None = None,
    ):
        # Jax will do loop unrolling here, but number of layers is not that big
        # so it will be fine.
        traces = prev_traces
        num_layers = len(theta_rtrl.layers)
        cell_index = theta_rtrl.num_rtrl_layers() - 1
        for i in range(num_layers - 1, -1, -1):
            layer = theta_rtrl.layers[i]
            if isinstance(layer, RTRLLayer):
                cell = layer.cell

                if jacobian_mask:
                    jacobian_cell_mask = jacobian_mask.layers[i].cell
                else:
                    jacobian_cell_mask = None

                cell_traces = cell.update_traces(
                    traces.layers[i].cell,
                    payloads[cell_index : cell_index + 1],
                    jacobian_cell_mask,
                )

                traces = eqx.tree_at(
                    lambda traces: traces.layers[i].cell, traces, cell_traces
                )

                cell_index -= 1

        return traces

    @staticmethod
    def update_grads(
        grads: RTRLStacked,
        hidden_states_grads: List[ArrayTree],
        jacobians: Traces | RTRLStacked,
        theta_rtrl: RTRLStacked,
    ):

        cell_index = 0
        for i, layer in enumerate(grads.layers):
            if isinstance(layer, RTRLLayer):
                ht_grad = hidden_states_grads[cell_index]
                rtrl_cell_jac = jacobians.layers[i].cell

                cell_grads = theta_rtrl.layers[i].cell.compute_grads(
                    ht_grad, rtrl_cell_jac[-1]
                )

                grads = eqx.tree_at(
                    lambda grads: grads.layers[i].cell,
                    grads,
                    cell_grads,
                    is_leaf=lambda x: x is None,
                )

                cell_index += 1

        return grads


class RTRLExact(RTRL):
    @staticmethod
    @jax.jit  # type: ignore
    def init_traces(model: RTRLStacked, jacobian_projection: RTRLStacked | None = None):

        cells_traces = eqx.filter(
            model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell
        )

        layer_index = 0
        n_layers = model.num_rtrl_layers()
        for i, layer in enumerate(model.layers):
            if isinstance(layer, RTRLLayer):
                cell = layer.cell
                all_traces = []
                for _ in range(layer_index, n_layers):
                    if not jacobian_projection:
                        traces = cell.init_traces()
                    else:
                        traces = jtu.tree_map(
                            lambda leaf: _sparse_jacobian(leaf),
                            jacobian_projection.layers[i].cell,
                            is_leaf=lambda node: isinstance(
                                node, (SparseProjection, DenseProjection)
                            ),
                        )

                    all_traces.append(traces)

                cells_traces = eqx.tree_at(
                    lambda cells: cells.layers[i].cell,
                    cells_traces,
                    tuple(all_traces),
                )
                layer_index += 1

        return cells_traces

    @staticmethod
    def update_traces(
        prev_traces: RTRLStacked,  # Also known as the trace
        payloads: List[ArrayTree],
        theta_rtrl: RTRLStacked,
        jacobian_mask: RTRLStacked | None = None,
    ):
        # Jax will do loop unrolling here, but number of layers is not that big
        # so it will be fine.
        traces = prev_traces
        num_layers = len(theta_rtrl.layers)
        cell_index = theta_rtrl.num_rtrl_layers() - 1
        for i in range(num_layers - 1, -1, -1):
            layer = theta_rtrl.layers[i]
            if isinstance(layer, RTRLLayer):
                cell = layer.cell

                if jacobian_mask:
                    jacobian_cell_mask = jacobian_mask.layers[i].cell
                else:
                    jacobian_cell_mask = None

                cell_traces = cell.update_traces(
                    traces.layers[i].cell,
                    payloads[cell_index:],
                    jacobian_cell_mask,
                )

                traces = eqx.tree_at(
                    lambda traces: traces.layers[i].cell, traces, cell_traces
                )

                cell_index -= 1

        return traces

    @staticmethod
    def update_grads(
        grads: RTRLStacked,
        hidden_states_grads: List[ArrayTree],
        jacobians: Traces | RTRLStacked,
        theta_rtrl: RTRLStacked,
    ):

        cell_index = 0
        for i, layer in enumerate(grads.layers):
            if isinstance(layer, RTRLLayer):
                ht_grad = hidden_states_grads[-1]
                rtrl_cell_jac = jacobians.layers[i].cell

                cell_grads = theta_rtrl.layers[i].cell.compute_grads(
                    ht_grad, rtrl_cell_jac[-1]
                )

                grads = eqx.tree_at(
                    lambda grads: grads.layers[i].cell,
                    grads,
                    cell_grads,
                    is_leaf=lambda x: x is None,
                )

                cell_index += 1

        return grads
