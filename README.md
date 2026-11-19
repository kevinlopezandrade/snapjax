# The Sparse n-Step Approximation in JAX

## Install
Use a current release of [uv](https://docs.astral.sh/uv/) to install Python
3.13 and the locked dependencies, including an editable installation of `snapjax`:

```bash
uv sync --locked
```

The default installation runs on CPU and requires no CUDA toolkit or GPU.
Select the CPU backend before JAX initializes:

```bash
JAX_PLATFORMS=cpu uv run python your_script.py
JAX_PLATFORMS=cpu uv run pytest
```

The stack uses JAX/jaxlib 0.12.0.1, Equinox 0.13.8, Numba 0.68.1,
NumPy 2.5.4, and pytest 9.2.0. `uv.lock` records the complete tested environment.
The native-kernel stack is pinned because JAX upgrades require compatibility
and numerical validation. Sparse differentiation uses the public `jax.vjp`
API with fixed sparse indices.

## Sparse native primitive

The sparse-preserving CSR–dense multiplication uses a Numba-compiled CPU kernel
inside JAX's compiled execution. It computes only entries in the requested
output pattern, without constructing a dense product. Numba compiles the kernel
on the host; no separate C/C++ build step is needed. Registration and lowering
use `jax.ffi` with XLA's unified, status-returning custom-call ABI.

An optional CUDA extension runs the same sparse-preserving operation on GPU.
It assigns one thread to each requested output entry and sums the corresponding
CSR row products. The extension uses JAX's typed FFI and the supplied CUDA stream.
Native builds target H100 (SM90) and B200 (SM100), with PTX for forward compatibility.

The primitive supports matching float32/float64 operands and int32 CSR/pattern
indices. Enable `jax_enable_x64` before creating float64 arrays. Its input must
be valid CSR; it supports JIT and vmap, but has no autodiff rule.
`mlops.set_cuda()` selects CUDA before
backend initialization; CPU remains available for constructing sparse masks.

### CUDA installation and validation

On Linux with a CUDA 13 toolkit, compatible NVIDIA driver, and H100 or B200:

```bash
uv sync --locked --extra cuda13
uv pip install ./snapjax/spp_primitives/gpu
JAX_PLATFORMS=cuda,cpu uv run --no-sync pytest -q snapjax/tests/primitive_test.py snapjax/tests/bptt_test.py::test_bptt_snap_diagonal_sparse
```

The separate `snapjax-cuda` package builds the `snapjax_cuda` extension. The base
SnapJax wheel stays platform independent; the source distribution includes the
CUDA build project. The checks compare sparse/dense masked-trace updates and
verify that the compiled program actually calls the native primitive.

## Validation

```bash
# Numerical regressions, including small training cases.
uv run --locked pytest -m "not slow"

# Original 256-unit RTRL/BPTT and online-training cases (more CPU time/memory).
uv run --locked pytest -m slow

# Build the pure-Python wheel and source distribution (including CUDA sources).
uv build
```

Tests default to CPU and enable float64 before collection. Set
`JAX_PLATFORMS=cuda,cpu` to run them on CUDA after installing the extension.
Shared initialization helpers live in
`snapjax.state`, and Jacobian updates in `snapjax.jacobians`; their existing
imports through `snapjax.algos` remain available.

## Experiment tracking

Experiments use [Minfx](https://minfx.ai/) project
`kevinlopezandrade@gmail.com/rtrl`. Install the tracking and plotting dependencies:

```bash
uv sync --locked --group experiments
```

Copy `.env.example` to `.env` and set your API token locally. `.env` is ignored
by Git. This project uses `MINFX_PROJECT`, `MINFX_API_TOKEN`, and `MINFX_MODE`.
Load them with uv when launching an experiment script:

```bash
uv run --locked --group experiments --env-file .env python path/to/experiment.py
```

Minfx 0.7.5 does not automatically read these names, so pass them explicitly
when initializing one run in the experiment entrypoint:

```python
import os
import minfx

run = minfx.init_run(
    project=os.environ["MINFX_PROJECT"],
    api_token=os.environ["MINFX_API_TOKEN"],
    mode=os.environ.get("MINFX_MODE", "async"),
    name="experiment-name",
    tags=["jax", "cpu"],
)
try:
    # Train and log metrics here.
    pass
finally:
    run.stop()
```

Log configuration under `params`, append scalar metrics such as `train/loss`
with an explicit `step`, and store final values under `summary`. Metric series provide learning
curves in Minfx; upload Matplotlib figures saved under `runs/` to fields such as
`plots/learning_curve` using `run["plots/learning_curve"].upload(plot_path)`.

Keep logging outside JAX-transformed functions (`jit`, `vmap`, and `lax.scan`).
Wrap the tracked work in `try/finally` and call `run.stop()` in `finally` so that
the run closes on both success and failure.

For local checks, set `MINFX_MODE=debug`; for deferred uploads, use
`MINFX_MODE=offline`. The SDK's local `.neptune/` cache is ignored by Git.
The SDK's sync command still reads Neptune-compatible names; map them only
for that process when uploading offline runs:

```bash
uv run --locked --group experiments --env-file .env sh -c '
  export NEPTUNE_API_TOKEN="$MINFX_API_TOKEN"
  export NEPTUNE_PROJECT="$MINFX_PROJECT"
  exec minfx sync --project "$MINFX_PROJECT" --offline-only
'
```
