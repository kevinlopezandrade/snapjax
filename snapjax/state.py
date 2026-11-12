"""Shared initialization helpers, independent of algorithm entry points."""

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO

from snapjax.cells.base import RTRLCell, RTRLLayer, RTRLStacked, Stacked, State, is_rtrl_cell
from snapjax.sp_jacrev import DenseProjection, SparseProjection, standard_jacobian


@jax.jit
def make_zeros_jacobians_sp(jacobian_projection: RTRLStacked):
    # Jacobians are saved as the tranpose jacobians.
    def _sparse_jacobian(leaf: SparseProjection | DenseProjection):
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

    zero_jacobians = jtu.tree_map(
        lambda leaf: _sparse_jacobian(leaf),
        jacobian_projection,
        is_leaf=lambda node: isinstance(node, (SparseProjection, DenseProjection)),
    )

    return zero_jacobians


@jax.jit
def make_zeros_jacobians(model: RTRLStacked):
    cells = eqx.filter(model, lambda leaf: is_rtrl_cell(leaf), is_leaf=is_rtrl_cell)

    def _cell_zero_jacobian(cell: RTRLCell):
        if cell.custom_trace_update:
            return cell.make_zero_traces()
        else:
            return jtu.tree_map(
                lambda leaf: standard_jacobian(leaf), cell.make_zero_jacobians()
            )

    zero_jacobians = jtu.tree_map(
        lambda cell: _cell_zero_jacobian(cell),
        cells,
        is_leaf=lambda leaf: is_rtrl_cell(leaf),
    )

    return zero_jacobians


@jax.jit
def make_zeros_grads(model: RTRLStacked):
    def zeros_in_leaf(leaf):
        if isinstance(leaf, BCOO):
            return jnp.zeros(leaf.nse)
        elif eqx.is_array(leaf):
            return jnp.zeros(shape=leaf.shape)

    zero_grads = jtu.tree_map(
        zeros_in_leaf, model, is_leaf=lambda node: isinstance(node, BCOO)
    )

    return zero_grads


def make_perturbations(model: RTRLStacked):
    perturbations = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            zeros = cell.init_perturbation()
            perturbations.append(zeros)

    return tuple(perturbations)


@jax.jit
def make_init_state(model: RTRLStacked) -> Stacked[State]:
    states: Sequence[State] = []
    for layer in model.layers:
        if isinstance(layer, RTRLLayer):
            cell = layer.cell
            states.append(cell.init_state())

    return tuple(states)
