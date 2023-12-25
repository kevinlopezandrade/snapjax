import timeit

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array
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


def sparse_preserving_product_scipy(
    A: np.ndarray, B: csr_array, rows_indices: np.ndarray, col_indices: np.ndarray
):
    """
    Its goint to compute B[rows_indices, :] @ A[:, col_indices], and return the data
    as data for a COO array.
    """
    N = rows_indices.shape[0]
    res = np.empty(shape=(N,), dtype=np.float32)

    for i in range(N):
        row = rows_indices[i]
        col = col_indices[i]
        res[i] = B.getrow(row) @ A[:, col]

    return res


def new_sparse_preserving_product(
    A: np.ndarray, B: csr_array, rows_indices: np.ndarray, col_indices: np.ndarray
):
    N = rows_indices.shape[0]
    data = B.data
    cols = B.indices
    indptr = B.indptr

    print((data.nbytes + cols.nbytes + indptr.nbytes) * 10e-6)

    res = np.empty(shape=(N,), dtype=np.float32)

    for n, (i, j) in enumerate(zip(rows_indices, col_indices)):
        row_start = indptr[i]
        row_end = indptr[i + 1]
        row = data[row_start:row_end]
        local_cols_indices = cols[row_start:row_end]
        res[n] = np.dot(row, A[:, j][local_cols_indices])

    return res


def dense_coo_product(D: np.ndarray, J: coo_array, new_algo: bool = False):
    """
    Its goint to compute D & J keeping the same
    sparsity pattern of J. The returned value is also
    a COO array.

    Only works for rhs shape invariant products.
    i.e (m, m)x(m, L) = (m, L)

    The computation is done by actually tranposing the
    product i.e:
    D & J = C
    J.T & D.T = C.T
    and then we undo the transpose.
    For this to be efficient we need J.T to be in
    CSR format.
    """
    J = J.transpose()
    rows = J.row
    cols = J.col

    J = J.tocsr()
    D = D.T

    if new_algo:
        data = new_sparse_preserving_product(D, J, rows, cols)
    else:
        data = sparse_preserving_product_scipy(D, J, rows, cols)
    res = coo_array((data, (rows, cols)), shape=J.shape)

    return res.transpose()


if __name__ == "__main__":
    N = 1500
    J = create_test_jacobian(N, jax.random.PRNGKey(7))

    J_expanded = J.reshape(N, N * N)
    D = jnp.arange(N * N).reshape(N, N)
    print(J_expanded.nbytes * 10e-6)

    print("Starting benchmark")
    number = 1
    res_1 = timeit.timeit(lambda: (D @ J_expanded).block_until_ready(), number=number)
    print(res_1)

    J_np = np.array(J_expanded)
    J_np = coo_array(J_np)
    D_np = np.array(D)

    res_3 = timeit.timeit(
        lambda: dense_coo_product(D_np, J_np, new_algo=True), number=number
    )
    print(res_3)
