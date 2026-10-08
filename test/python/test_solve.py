from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cases import lp_case
from checks import assert_dual_certificate, assert_optimal, assert_primal_certificate
from hpdlp import Model, PDLP

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("name,parameters", [
    ("minimize", {}), ("minimize", {"Presolve": True}),
    ("minimize", {"ActiveSetBoost": False}), ("maximize", {}),
    ("maximize", {"GeoMeanIters": 0, "RuizIters": 0, "UsePCAlpha": False, "BoundObjRescaling": False}),
    ("box", {"OptimalityNorm": "linf", "BoundObjRescaling": False}),
    ("fixed_and_free", {"Presolve": True}), ("degenerate", {}),
    ("zero_columns", {}), ("empty", {}),
])
def test_optimal_models(name, parameters):
    case = lp_case(name)
    model = case.build_model(**parameters)
    assert model.optimize() is model
    assert_optimal(model, case.expected_objective)


@pytest.mark.parametrize("presolve", [False, True])
def test_primal_infeasible_result(presolve):
    model = Model([1.], np.ones((2, 1)), [2., -np.inf], [np.inf, 1.])
    model.setParams(OutputFlag=False, Presolve=presolve)
    model.optimize()
    assert model.Status == PDLP.PRIMAL_INFEASIBLE and model.X is None
    if not presolve:
        assert_primal_certificate(model)
    elif model.Pi is not None and model.RC is not None:
        assert_primal_certificate(model)
    else:
        assert model.Pi is None and model.RC is None


@pytest.mark.parametrize("presolve", [False, True])
def test_dual_infeasible_result(presolve):
    model = Model([-1.], np.ones((1, 1)), [0.], None)
    model.setParams(OutputFlag=False, Presolve=presolve)
    model.optimize()
    assert model.Status in (PDLP.DUAL_INFEASIBLE, PDLP.INFEASIBLE_OR_UNBOUNDED)
    assert model.Pi is None and model.RC is None
    if not presolve or model.X is not None:
        assert_dual_certificate(model)


@pytest.mark.parametrize("parameter,value,status", [
    ("IterationLimit", 1, PDLP.ITERATION_LIMIT),
    ("TimeLimit", 1e-12, PDLP.TIME_LIMIT),
])
def test_stopping_limits_return_candidates(model, parameter, value, status):
    model.setParam(parameter, value)
    model.optimize()
    assert model.Status == status
    assert model.IterCount >= 0
    if status == PDLP.ITERATION_LIMIT:
        assert model.IterCount >= 1
    assert model.X.shape == (2,) and model.Pi.shape == (3,) and model.RC.shape == (2,)
    assert np.all(np.isfinite(model.X))
    assert model.ObjVal == pytest.approx(model.c @ model.X + model.c0, abs=1e-10, rel=0)


def test_independent_models_called_from_threads():
    names = ["minimize", "maximize", "box"]

    def solve(name):
        case = lp_case(name)
        model = case.build_model().optimize()
        assert_optimal(model, case.expected_objective)
        return model.ObjVal

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(solve, names))
    np.testing.assert_allclose(results, [5, 5.25, -11], atol=1e-5, rtol=0)
