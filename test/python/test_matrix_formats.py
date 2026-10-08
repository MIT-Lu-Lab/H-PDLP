import numpy as np
import pytest
import scipy.sparse as sp


from hpdlp import Model


def test_sparse_model_arrays_are_read_only(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, sp.csr_matrix(A), l, u, lb, ub)
    with pytest.raises(ValueError):
        model.A.data[0] = 99.0
    with pytest.raises(ValueError):
        model.A.indices[0] = 99
    with pytest.raises(ValueError):
        model.A.indptr[0] = 99


def test_non_contiguous_and_typed_inputs(base_lp_data):
    _, _, l, u, _, _ = base_lp_data
    c_nc = np.ones((2, 4))[:, 1]
    assert not c_nc.flags['C_CONTIGUOUS']
    A = sp.csr_matrix(np.array([[1, 2], [0, 1], [3, 2]], dtype=np.int64))
    model = Model(c_nc, A, l, u)
    assert model.c.dtype == np.float64
    assert model.A.dtype == np.float64
    assert model.A.indices.dtype == np.int32
    np.testing.assert_array_equal(model.c, [1., 1.])
    np.testing.assert_array_equal(model.A.toarray(), [[1., 2.], [0., 1.], [3., 2.]])


def test_int64_index_downcast(base_lp_data):
    _, _, l, u, _, _ = base_lp_data
    c = np.array([1.0, 1.0])
    A = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 2.0]]))
    A.indices = A.indices.astype(np.int64)
    A.indptr = A.indptr.astype(np.int64)
    model = Model(c, A, l, u)
    assert model.A.indices.dtype == np.int32
    assert model.A.indptr.dtype == np.int32
    assert A.indices.dtype == np.int64
    assert A.indptr.dtype == np.int64
    np.testing.assert_array_equal(model.A.toarray(), A.toarray())


@pytest.mark.parametrize('fmt,index', [('csr', -4294967296), ('csc', -1), ('coo', 1), ('lil', -1)])
def test_invalid_sparse_indices_rejected_before_conversion(fmt, index):
    matrix = getattr(sp, fmt + '_matrix')([[1.0]])
    if fmt == 'lil':
        matrix.rows[0] = [index]
    elif fmt == 'coo':
        if hasattr(matrix, 'coords'):
            matrix.coords = (matrix.row, np.array([index], dtype=np.int64))
        else:
            matrix.col = np.array([index], dtype=np.int64)
    else:
        matrix.indices = np.array([index], dtype=np.int64)
    with pytest.raises(ValueError):
        Model([1.0], matrix, [1.0], None)


@pytest.mark.parametrize('indptr', [[0, 4, 2], [1, 1, 2], [0, -1, 2]])
def test_invalid_sparse_structure_rejected(indptr):
    matrix = sp.eye(2, format='csr')
    matrix.indptr = np.array(indptr, dtype=np.int64)
    with pytest.raises(ValueError):
        Model([1.0, 1.0], matrix, [1.0, 1.0], None)


@pytest.mark.parametrize("format", [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                                   sp.csr_array, sp.lil_matrix])
def test_sparse_inputs_preserve_values_and_caller_ownership(format):
    source = sp.csr_matrix((np.array([3., 2., -3., 4., -1.]),
                           np.array([2, 0, 2, 1, 1]), np.array([0, 3, 3, 5])), shape=(3, 3))
    source = format(source)
    original = source.toarray().copy()
    model = Model(np.ones(3), source, None, None)
    np.testing.assert_array_equal(model.A.toarray(), [[2., 0., 0.], [0., 0., 0.], [0., 3., 0.]])
    np.testing.assert_array_equal(source.toarray(), original)
    assert model.A.has_canonical_format
    if source.format == "coo":
        source.data[:] = 99
    else:
        source[0, 0] = 99
    np.testing.assert_array_equal(model.A.toarray(), original)


@pytest.mark.gpu
@pytest.mark.parametrize("format", [np.asarray, sp.csr_matrix, sp.csc_matrix, sp.coo_matrix])
def test_model_solves_each_matrix_format(base_lp_data, format):
    from checks import assert_optimal

    c, a, lower, upper, lb, ub = base_lp_data
    model = Model(c, format(a), lower, upper, lb, ub)
    model.setParams(OutputFlag=False, Presolve=False, OptimalityTol=1e-7, FeasibilityTol=1e-7)
    model.optimize()
    assert_optimal(model, 3)
