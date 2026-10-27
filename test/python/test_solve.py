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


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("dummy_equality", [False, True])
def test_huge_objective_does_not_certify_infeasibility(presolve, dummy_equality):
    if dummy_equality:
        c = np.array([-1e10, -1.001e10, 0.])
        matrix = np.array([[1., 1., 0.], [0., 0., 1e10]])
        lower, upper = [-np.inf, 0.], [2., 0.]
    else:
        c = np.array([-1e10, -1.001e10])
        matrix = np.array([[1., 1.], [-1e10, 1e10]])
        lower, upper = [-np.inf, 0.], [2., np.inf]
    model = Model(c, matrix, lower, upper, np.zeros(c.size), None)
    model.setParams(OutputFlag=False, Presolve=presolve,
                    OptimalityTol=1e-7, FeasibilityTol=1e-7,
                    IterationLimit=20000, TimeLimit=10)
    model.optimize()
    assert model.Status == PDLP.OPTIMAL
    assert model.ObjVal == pytest.approx(-2.002e10, rel=1e-7)
    np.testing.assert_allclose(model.X[:2], [0., 2.], atol=1e-6)


@pytest.mark.parametrize("presolve", [False, True])
@pytest.mark.parametrize("objective_scale", [1e-6, 1., 1e8])
def test_infeasible_certificate_with_scaled_objective(presolve, objective_scale):
    model = Model([objective_scale], np.ones((2, 1)), [2., -np.inf], [np.inf, 1.])
    model.setParams(OutputFlag=False, Presolve=presolve, TimeLimit=10)
    model.optimize()
    assert model.Status == PDLP.PRIMAL_INFEASIBLE
    assert presolve or model.Pi is not None
    if model.Pi is not None:
        assert_primal_certificate(model)


@pytest.mark.parametrize("margin,status", [(0., PDLP.PRIMAL_INFEASIBLE), (2., PDLP.ITERATION_LIMIT)])
def test_objective_margin_controls_certificate_acceptance(margin, status):
    model = Model([1.], np.ones((2, 1)), [2., -np.inf], [np.inf, 1.])
    model.setParams(OutputFlag=False, Presolve=False, FarkasObjectiveMargin=margin,
                    IterationLimit=2000, TimeLimit=10)
    model.optimize()
    assert model.Status == status
    if status == PDLP.PRIMAL_INFEASIBLE:
        assert_primal_certificate(model)


@pytest.mark.parametrize("parameter,value,status", [
    ("IterationLimit", 1, PDLP.ITERATION_LIMIT),
    ("TimeLimit", 1e-12, PDLP.TIME_LIMIT),
])
def test_stopping_limits_return_candidates(model, parameter, value, status):
    unreachable = np.nextafter(0.0, 1.0)
    model.setParams(OptimalityTol=unreachable, FeasibilityTol=unreachable)
    model.setParam(parameter, value)
    model.optimize()
    assert model.Status == status
    assert model.IterCount == model.getParam("TermCheckFreq")
    assert model.X.shape == (2,) and model.Pi.shape == (3,) and model.RC.shape == (2,)
    assert np.all(np.isfinite(model.X))
    assert np.all(np.isfinite(model.Pi)) and np.all(np.isfinite(model.RC))
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
