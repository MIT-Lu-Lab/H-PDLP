import numpy as np
import pytest

from checks import assert_optimal


@pytest.fixture
def core_calls(monkeypatch):
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    monkeypatch.setattr("hpdlp.model.solve_once", capture)
    return calls


def test_warm_start_copies_and_forwards_inputs(model, core_calls):
    primal = np.array([1., 2.])
    dual = np.array([1., -1., 0.])
    model.setWarmStart(primal=primal, dual=dual)
    primal[:] = dual[:] = 99
    model.optimize()
    _, parameters = core_calls[-1]
    np.testing.assert_array_equal(parameters["primal_start"], [1., 2.])
    np.testing.assert_array_equal(parameters["dual_start"], [1., -1., 0.])


def test_partial_updates_and_explicit_clears(model, core_calls):
    model.setWarmStart(primal=[1., 2.], dual=[1., -1., 0.])
    model.setWarmStart(primal=[2., 1.])
    model.optimize()
    np.testing.assert_array_equal(core_calls[-1][1]["dual_start"], [1., -1., 0.])
    model.setWarmStart(dual=None)
    model.optimize()
    np.testing.assert_array_equal(core_calls[-1][1]["primal_start"], [2., 1.])
    assert core_calls[-1][1]["dual_start"] is None
    model.clearWarmStart()
    model.optimize()
    assert core_calls[-1][1]["primal_start"] is None
    assert core_calls[-1][1]["dual_start"] is None


@pytest.mark.parametrize("update", [
    {"primal": [1.]}, {"dual": [1., 2.]},
    {"primal": [np.nan, 0.]}, {"dual": [0., np.inf, 0.]},
    {"primal": [2., 1.], "dual": [1., 2.]},
])
def test_rejected_update_keeps_previous_starts(model, core_calls, update):
    model.setWarmStart(primal=[1., 2.], dual=[1., -1., 0.])
    with pytest.raises(ValueError):
        model.setWarmStart(**update)
    model.optimize()
    np.testing.assert_array_equal(core_calls[-1][1]["primal_start"], [1., 2.])
    np.testing.assert_array_equal(core_calls[-1][1]["dual_start"], [1., -1., 0.])


@pytest.mark.gpu
@pytest.mark.parametrize("start", [
    {"primal": [1., 2.]}, {"dual": [1., -1., 0.]},
    {"primal": [1., 2.], "dual": [1., -1., 0.]},
])
def test_warm_start_solve(model, start):
    model.setWarmStart(**start)
    model.optimize()
    assert_optimal(model, 3)
