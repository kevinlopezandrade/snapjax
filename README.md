# The Sparse n-Step Approximation in JAX.

## Install
In a new conda environment with python 3.11.
```bash
$ conda create -n NAME python=3.11
```
run the following commands to have the modules installed
as a python package.
```bash
$ pip install poetry
$ poetry update
$ pip install -e .
```
## CUDA Primitive
The cuda kernel in *snapjax/spp_primitives/*, was compiled
against cuda 12.2. If recompilation is necessary
for your needs, modify the CMakeLists.txt file in
*snapjax/spp_primitives/gpu*.
