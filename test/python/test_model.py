import numpy as np
import pytest
import scipy.sparse as sp

import hpdlp
from hpdlp import Model, PDLP, read


def test_public_exports_include_documented_api():
    assert hpdlp.PDLP is PDLP
    assert hpdlp.Model is Model
    assert hpdlp.read is read
    assert isinstance(hpdlp.__version__, str)
    assert set(hpdlp.__all__) == {'Model', 'PDLP', 'read', '__version__'}


def test_data_property_reads(model):
    assert model.c.shape == (2,)
    assert model.c0 == 0.0
    assert model.A.shape == (3, 2)
    assert model.constr_lb.shape == (3,)
    assert model.constr_ub.shape == (3,)
    assert model.lb is None
    assert model.ub is None
    assert model.ModelSense == PDLP.MINIMIZE


def test_data_property_writes(model):
    model.c = [2.0, 3.0]
    assert np.allclose(model.c, [2.0, 3.0])
    model.c0 = 4.0
    assert model.c0 == 4.0
    model.lb = [0.0, 0.0]
    assert np.allclose(model.lb, [0.0, 0.0])
    model.ub = [10.0, 10.0]
    assert np.allclose(model.ub, [10.0, 10.0])
    model.constr_lb = [1.0, 2.0, 3.0]
    assert np.allclose(model.constr_lb, [1.0, 2.0, 3.0])
    model.constr_ub = [4.0, 5.0, 6.0]
    assert np.allclose(model.constr_ub, [4.0, 5.0, 6.0])
    model.A = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    assert model.A.shape == (3, 2)
    model.ModelSense = PDLP.MAXIMIZE
    assert model.ModelSense == PDLP.MAXIMIZE


def test_bounds_none_clears(model):
    model.lb = [0.0, 0.0]
    model.lb = None
    assert model.lb is None
    model.ub = None
    assert model.ub is None
    model.constr_lb = None
    assert model.constr_lb is None
    model.constr_ub = None
    assert model.constr_ub is None


def test_property_validation_errors(model):
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        model.c = [[1.0, 2.0]]
    with pytest.raises(TypeError):
        model.A = 'not a matrix'
    with pytest.raises(ValueError):
        model.A = np.ones(3)
    with pytest.raises(ValueError):
        model.A = np.ones((3, 5))
    with pytest.raises(ValueError):
        model.lb = [0.0]
    with pytest.raises(ValueError):
        model.ub = [0.0]
    with pytest.raises(ValueError):
        model.constr_lb = [0.0, 0.0]
    with pytest.raises(ValueError):
        model.constr_ub = [0.0, 0.0]
    with pytest.raises(ValueError):
        model.ModelSense = 99


def test_bound_nan_validation(model):
    with pytest.raises(ValueError):
        model.lb = [0.0, np.nan]
    with pytest.raises(ValueError):
        model.ub = [np.nan, 1.0]
    with pytest.raises(ValueError):
        model.constr_lb = [0.0, np.nan, -np.inf]
    with pytest.raises(ValueError):
        model.constr_ub = [0.0, 1.0, np.nan]
    model.lb = [0.0, -np.inf]
    model.ub = [np.inf, 1.0]
    model.constr_lb = [5.0, -np.inf, -np.inf]
    model.constr_ub = [5.0, np.inf, 8.0]


def test_invalid_objective_assignment_preserves_previous_value(model):
    original = model.c.copy()
    with pytest.raises(ValueError):
        model.c = [1.0, 2.0, 3.0]
    assert np.allclose(model.c, original)


def test_dense_inputs_are_copied(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    c = c.copy()
    A = A.copy()
    l = l.copy()
    u = u.copy()
    model = Model(c, A, l, u, lb, ub)
    c[0] = 99.0
    A[0, 0] = 99.0
    l[0] = 99.0
    u[0] = 99.0
    assert np.allclose(model.c, [1.0, 1.0])
    assert model.A[0, 0] == 1.0
    assert model.constr_lb[0] == 5.0
    assert model.constr_ub[0] == 5.0


def test_model_arrays_are_read_only(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    with pytest.raises(ValueError):
        model.c[0] = 99.0
    with pytest.raises(ValueError):
        model.A[0, 0] = 99.0
    with pytest.raises(ValueError):
        model.constr_lb[0] = 99.0
    with pytest.raises(ValueError):
        model.constr_ub[0] = 99.0
    model.c = [2.0, 3.0]
    assert np.allclose(model.c, [2.0, 3.0])


def test_bound_order_validation(model):
    model.lb = [2.0, 0.0]
    model.ub = [1.0, 1.0]
    with pytest.raises(ValueError):
        model.optimize()
    with pytest.raises(ValueError):
        Model([1.0], np.array([[1.0]]), constraint_lower_bound=[2.0], constraint_upper_bound=[1.0])


def test_matrix_row_change_conflicts_with_bounds(model):
    two_row = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        model.A = two_row
    model.constr_lb = None
    with pytest.raises(ValueError):
        model.A = two_row
    model.constr_ub = None
    model.A = two_row
    model.constr_lb = [0.0, 0.0]
    model.constr_ub = [1.0, 1.0]
    assert model.num_constrs == 2


def test_init_requires_2d_matrix():
    with pytest.raises(ValueError):
        Model(np.ones(2), np.ones(3), None, None)


@pytest.mark.parametrize('attribute,value', [('lb', np.inf), ('ub', -np.inf), ('constr_lb', np.inf), ('constr_ub', -np.inf)])
def test_invalid_infinite_bound_preserves_model(attribute, value):
    model = Model([1.0], np.array([[1.0]]), [0.0], [2.0], [0.0], [2.0])
    previous = getattr(model, attribute).copy()
    with pytest.raises(ValueError):
        setattr(model, attribute, [value])
    np.testing.assert_array_equal(getattr(model, attribute), previous)


def test_omitted_variable_bounds_leave_variables_free():
    model = Model([1.], np.array([[1.]]), [-2.], [-2.])
    assert model.lb is None and model.ub is None


def test_optimize_forwards_model_and_parameters(model, monkeypatch):
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr("hpdlp.model.solve_once", capture)
    model.c0 = 4.
    model.ModelSense = PDLP.MAXIMIZE
    model.setParams(TimeLimit=12., OptimalityNorm="linf")
    assert model.optimize() is model
    args, parameters = calls[0]
    for actual, expected in zip(args, (model.A, model.c, 4., model.lb, model.ub,
                                      model.constr_lb, model.constr_ub)):
        np.testing.assert_array_equal(actual, expected)
    assert parameters["minimize"] is False
    assert parameters["params"]["time_sec_limit"] == 12.
    assert parameters["params"]["optimality_norm"] == "linf"
