# The Sparse n-Step Approximation in JAX

## Install
Use [uv](https://docs.astral.sh/uv/) to install Python 3.11 and the locked
dependencies, including an editable installation of `snapjax`:

```bash
uv sync --locked
```

This baseline supports CPU execution. It requires no CUDA toolkit, GPU, or
prebuilt CUDA extension. Select the CPU backend before JAX initializes:

```bash
JAX_PLATFORMS=cpu uv run python your_script.py
JAX_PLATFORMS=cpu uv run pytest
```

The JAX, jaxlib, Equinox, and Numba versions are constrained deliberately while
the project uses JAX's custom-call and internal sparse differentiation APIs.
`uv.lock` records the complete tested environment; upgrading the JAX stack
requires compatibility and numerical validation.

## Sparse CPU primitive

The sparse-preserving CSR–dense multiplication uses a Numba-compiled CPU kernel
inside JAX's compiled execution. It computes only entries in the requested
output pattern, without constructing a dense product. Numba compiles the kernel
on the host; no separate C/C++ build step is needed.

Historical CUDA sources remain in `snapjax/spp_primitives/gpu` for reference.
They and the old platform-specific binary are excluded from distributions and
are not part of the supported CPU runtime.

The primitive supports matching float32/float64 operands and int32 CSR/pattern
indices. Enable `jax_enable_x64` before creating float64 arrays. Its input must
be valid CSR; it supports JIT and vmap, but has no autodiff rule. The legacy
`mask_mat_mul` and `mlops.set_cuda` helpers explicitly report that CUDA execution
is unsupported.

## Validation

```bash
# Numerical regressions, including small training cases.
uv run --locked pytest -m "not slow"

# Original 256-unit RTRL/BPTT and online-training cases (more CPU time/memory).
uv run --locked pytest -m slow

# Build a wheel and source distribution without CUDA artifacts.
uv build
```

Tests select the CPU backend and enable float64 before collection.
Shared initialization helpers live in
`snapjax.state`, and Jacobian updates in `snapjax.jacobians`; their existing
imports through `snapjax.algos` remain available.
