import numpy as np
import pytest

from hopdlp import PDLP


def _bounds(lower, upper, size):
    return (np.full(size, -np.inf) if lower is None else np.asarray(lower),
            np.full(size, np.inf) if upper is None else np.asarray(upper))


def _norm(vector, kind):
    return np.linalg.norm(vector, ord=np.inf if kind == "linf" else 2) if vector.size else 0.


def _support(multiplier, lower, upper):
    positive, negative = multiplier > 0, multiplier < 0
    assert np.all(np.isfinite(lower[positive])), "positive multiplier has no lower bound"
    assert np.all(np.isfinite(upper[negative])), "negative multiplier has no upper bound"
    return multiplier[positive] @ lower[positive] + multiplier[negative] @ upper[negative]


def assert_optimal(model, expected_objective):
    assert model.Status == PDLP.OPTIMAL
    assert model.StatusName == "OPTIMAL"
    x, y, z = model.X, model.Pi, model.RC
    assert x.shape == (model.num_vars,) and y.shape == (model.num_constrs,)
    assert z.shape == (model.num_vars,)
    assert all(np.all(np.isfinite(vector)) for vector in (x, y, z))
    lower, upper = _bounds(model.lb, model.ub, model.num_vars)
    row_lower, row_upper = _bounds(model.constr_lb, model.constr_ub, model.num_constrs)
    activity = np.asarray(model.A @ x).ravel()
    violation = np.concatenate((x - np.clip(x, lower, upper),
                                activity - np.clip(activity, row_lower, row_upper)))
    kind = model.getParam("OptimalityNorm")
    finite_bounds = np.concatenate((row_lower[np.isfinite(row_lower) & (row_lower != row_upper)],
                                    row_upper[np.isfinite(row_upper)]))
    tolerance = model.getParam("FeasibilityTol")
    assert _norm(violation, kind) <= tolerance * (1 + _norm(finite_bounds, kind)) + 1e-12
    stationarity = model.c - np.asarray(model.A.T @ y).ravel() - z
    assert _norm(stationarity, kind) <= tolerance * (1 + _norm(model.c, kind)) + 1e-12
    sign = 1 if model.ModelSense == PDLP.MINIMIZE else -1
    objective = model.c @ x + model.c0
    dual_objective = sign * (_support(sign * y, row_lower, row_upper) +
                             _support(sign * z, lower, upper)) + model.c0
    relative_gap = abs(objective - dual_objective) / (1 + abs(objective) + abs(dual_objective))
    assert relative_gap <= model.getParam("OptimalityTol") + 1e-12
    assert objective == pytest.approx(expected_objective, abs=1e-5, rel=0)
    assert model.ObjVal == pytest.approx(objective, abs=1e-10, rel=1e-12)
    assert model.DualObj == pytest.approx(dual_objective, abs=1e-10, rel=1e-12)
    assert model.Gap == pytest.approx(abs(objective - dual_objective), abs=1e-10, rel=1e-12)
    assert model.RelGap == pytest.approx(relative_gap, abs=1e-10, rel=1e-12)


def assert_primal_certificate(model):
    assert model.Status == PDLP.PRIMAL_INFEASIBLE and model.X is None
    y, z = model.Pi, model.RC
    assert y is not None and z is not None
    assert y.shape == (model.num_constrs,) and z.shape == (model.num_vars,)
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(z))
    lower, upper = _bounds(model.lb, model.ub, model.num_vars)
    row_lower, row_upper = _bounds(model.constr_lb, model.constr_ub, model.num_constrs)
    assert _norm(np.asarray(model.A.T @ y).ravel() + z, "linf") <= 1e-6
    assert _support(y, row_lower, row_upper) + _support(z, lower, upper) > 0


def assert_dual_certificate(model):
    assert model.Status == PDLP.DUAL_INFEASIBLE
    assert model.Pi is None and model.RC is None
    direction = model.X
    assert direction is not None and direction.shape == (model.num_vars,)
    assert np.all(np.isfinite(direction))
    lower, upper = _bounds(model.lb, model.ub, model.num_vars)
    row_lower, row_upper = _bounds(model.constr_lb, model.constr_ub, model.num_constrs)
    activity = np.asarray(model.A @ direction).ravel()
    for vector, lb, ub in ((direction, lower, upper), (activity, row_lower, row_upper)):
        assert np.all(vector[np.isfinite(lb)] >= -1e-6)
        assert np.all(vector[np.isfinite(ub)] <= 1e-6)
    sign = 1 if model.ModelSense == PDLP.MINIMIZE else -1
    assert sign * model.c @ direction < 0
