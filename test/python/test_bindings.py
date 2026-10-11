import numpy as np
import pytest
import scipy.sparse as sp

from hpdlp import PDLP
from hpdlp._core import solve_once


def test_direct_core_param_value_validation(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={'verbose': 1})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={'iteration_limit': False})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={'eps_optimal_relative': 0.0})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={'time_sec_limit': float('nan')})
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, params={'optimality_norm': 'l1'})
    with pytest.raises(ValueError, match='Unknown parameter'):
        solve_once(A, c, None, lb, ub, l, u, params={'eps_optimal_relatve': 1e-06})
    with pytest.raises(ValueError, match='must be a number'):
        solve_once(A, c, None, lb, ub, l, u, params={'time_sec_limit': '60'})
    with pytest.raises(ValueError, match='int32'):
        solve_once(A, c, None, lb, ub, l, u, params={'iteration_limit': 2 ** 40})


def test_direct_core_model_data_validation(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    with pytest.raises(ValueError):
        solve_once(A, [np.nan, 1.0], None, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once(np.array([[np.nan, 1.0], [0.0, 1.0], [3.0, 2.0]]), c, None, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, np.nan, lb, ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, [0.0, np.nan], ub, l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, [2.0, 0.0], [1.0, 1.0], l, u)
    with pytest.raises(ValueError):
        solve_once(A, c, None, lb, ub, l, u, primal_start=[np.nan, 0.0])


@pytest.mark.parametrize('defect', ['start', 'end', 'order', 'index', 'overflow'])
def test_direct_malformed_sparse_rejected(defect, base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    matrix = sp.csr_matrix(A)
    if defect == 'start':
        matrix.indptr[0] = 1
    elif defect == 'end':
        matrix.indptr[-1] -= 1
    elif defect == 'order':
        matrix.indptr[1] = matrix.indptr[2] + 1
    elif defect == 'index':
        matrix.indices[0] = matrix.shape[1]
    else:
        matrix.indices = matrix.indices.astype(np.int64)
        matrix.indices[0] = 2 ** 32
    with pytest.raises((ValueError, OverflowError)):
        solve_once(matrix, c, 0.0, lb, ub, l, u)


@pytest.mark.parametrize('position,value', [(0, np.inf), (1, -np.inf), (2, np.inf), (3, -np.inf)])
def test_direct_invalid_infinite_bound_rejected(position, value):
    bounds = [None] * 4
    bounds[position] = np.array([value])
    with pytest.raises(ValueError):
        solve_once(np.array([[1.0]]), np.array([1.0]), 0.0, *bounds)


@pytest.mark.gpu
@pytest.mark.parametrize("format", [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_direct_sparse_solve(base_lp_data, format, index_dtype):
    c, a, lower, upper, lb, ub = base_lp_data
    matrix = format(a)
    if matrix.format == "coo":
        if hasattr(matrix, "coords"):
            matrix.coords = tuple(index.astype(index_dtype) for index in matrix.coords)
        else:
            matrix.row = matrix.row.astype(index_dtype)
            matrix.col = matrix.col.astype(index_dtype)
    else:
        matrix.indices = matrix.indices.astype(index_dtype)
        matrix.indptr = matrix.indptr.astype(index_dtype)
    info = solve_once(matrix, c, 4., lb, ub, lower, upper,
                      params={"verbose": False, "presolve": False,
                              "eps_optimal_relative": 1e-7, "eps_feasible_relative": 1e-7})
    assert info["StatusCode"] == PDLP.OPTIMAL
    np.testing.assert_allclose(info["X"], [1., 2.], atol=1e-5, rtol=0)
    assert info["PrimalObj"] == pytest.approx(c @ info["X"] + 4., abs=1e-10, rel=0)
    np.testing.assert_allclose(c - a.T @ info["Pi"] - info["RC"], 0, atol=1e-6, rtol=0)


@pytest.mark.gpu
def test_direct_dense_layout_and_objective_sense(base_lp_data):
    c, a, lower, upper, lb, ub = base_lp_data
    matrix = np.asfortranarray(a, dtype=np.float32)
    info = solve_once(matrix, c.astype(np.float32), 2., lb, ub, lower, upper,
                      params={"verbose": False, "presolve": False}, minimize=False)
    assert info["StatusCode"] == PDLP.OPTIMAL
    np.testing.assert_allclose(info["X"], [1.5, 1.75], atol=1e-3, rtol=0)
    assert info["PrimalObj"] == pytest.approx(5.25, abs=1e-3, rel=0)
