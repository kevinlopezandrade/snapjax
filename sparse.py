import timeit

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array
from numba import njit
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array

from algos import sparse_multiplication


def create_test_jacobian(n: int, key):
    matrices = []
    for i in range(n):
        x = jax.random.normal(key, shape=(n,))
        matrix = jnp.zeros(shape=(n, n))
        matrix = matrix.at[i, :].set(x)
        matrices.append(matrix)

    return jnp.stack(matrices)


@jax.jit
def sparse_preserving_product(A: Array, B: BCOO):
    print("Compiling sparse_preserving_product_new")
    indices = B.indices

    rows_indices = indices[:, 0]
    rows = A[rows_indices]

    col_indices = indices[:, 1]
    cols = B[:, col_indices].todense()  # Fuck this might be slow.

    data = jax.vmap(jnp.matmul, in_axes=(0, 1))(rows, cols)

    return BCOO((data, indices), shape=B.shape)


@njit
def spp_csr_matmul(
    data: NDArray[np.float32],
    cols: NDArray[np.int32],
    indptr: NDArray[np.int32],
    B: NDArray[np.float32],
    sp: NDArray[np.int32],
) -> NDArray[np.float32]:
    """
    Performs a sparse preserving matrix multiplciation
    between A(data, cols, indptr) @ B, where only
    the entries of the sparse pattern are computed.

    Args:
        data: The data array of the csr format for matrix A.
        cols: The cols array of the csr format for matrix A.
        indptr: The row pointers of the csr format for matrix A.
        B: The dense matrix B
        sp: A 2D array where each row contains the indeces of the
            non-zero entries of the sparse pattern.

    Returns:
        The data array to be used in the construction of a COO
        matrix.
    """
    N = sp.shape[0]
    res = np.zeros(shape=(N,), dtype=np.float32)

    for n in range(N):
        i, j = sp[n]
        row_start = indptr[i]
        row_end = indptr[i + 1]
        if row_end - row_end == 0:
            res[n] = 0
        else:
            row = data[row_start:row_end]
            local_cols_indices = cols[row_start:row_end]
            res[n] = np.dot(row, B[:, j][local_cols_indices])

    return res


def dense_coo_product(D: NDArray[np.float32], J: coo_array):
    """
    Its goint to compute D & J keeping the same sparsity pattern of J. The
    returned value is also a COO array. Only works for rhs shape invariant
    products. i.e (m, m)x(m, L) = (m, L)

    The computation is done by actually tranposing the product, D & J = C =>
    J.T & D.T = C.T and then we undo the transpose. For this to be efficient we
    need J.T to be in CSR format.
    """
    J = J.T
    sp = np.stack((J.row, J.col), axis=1)

    # To CSR
    J = J.tocsr()
    D = D.T

    data = spp_csr_matmul(J.data, J.indices, J.indptr, D, sp)
    res = coo_array((data, (sp[:, 0], sp[:, 1])), shape=J.shape)

    return res.T


if __name__ == "__main__":
    N = 256
    J = create_test_jacobian(N, jax.random.PRNGKey(7))

    J_expanded = J.reshape(N, N * N)
    D = jnp.arange(N * N).reshape(N, N).astype(jnp.float32)
    print(J_expanded.nbytes * 10e-6)

    print("Starting benchmark")
    number = 1
    res_1 = timeit.timeit(lambda: (D @ J_expanded).block_until_ready(), number=number)
    print(res_1)

    J_np = np.array(J_expanded)
    J_np = coo_array(J_np)
    D_np = np.array(D, dtype=np.float32)

    print("Test call to numba")
    dense_coo_product(D_np, J_np)

    res_3 = timeit.timeit(
        lambda: dense_coo_product(D_np, J_np), number=number
    )
    print(res_3)
