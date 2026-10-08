import gc

import numpy as np
import pytest

from checks import assert_optimal
from hpdlp import PDLP


RESULT_FIELDS = {
    "ObjVal": "PrimalObj", "DualObj": "DualObj", "Gap": "ObjectiveGap",
    "RelGap": "RelativeObjectiveGap", "IterCount": "Iterations",
    "Runtime": "RuntimeSec", "RescalingTime": "RescalingTimeSec",
    "RelPrimalResidual": "RelativePrimalResidual",
    "RelDualResidual": "RelativeDualResidual",
    "PrimalInfeas": "RelativePrimalResidual", "DualInfeas": "RelativeDualResidual",
    "FarkasMargin": "FarkasMargin", "FarkasResidual": "FarkasResidual",
    "HomogeneousScale": "HomogeneousScale", "RestartCount": "RestartCount",
    "PresolveTime": "PresolveTime", "PresolveStatus": "PresolveStatus",
    "ASBRaiseCount": "ASBRaiseCount", "ASBRevertCount": "ASBRevertCount",
    "ASBPowerIterations": "ASBPowerIterations",
    "FinalStepSizePrimal": "FinalStepSizePrimal", "FinalStepSizeDual": "FinalStepSizeDual",
    "FinalStepSizeHomogeneous": "FinalStepSizeHomogeneous",
}


@pytest.fixture
def core_result(monkeypatch):
    info = {field: index + 1 for index, field in enumerate(dict.fromkeys(RESULT_FIELDS.values()))}
    info.update(X=np.array([1., 2.]), Pi=np.array([3., 4., 5.]), RC=np.array([6., 7.]),
                Status="OPTIMAL", StatusCode=PDLP.OPTIMAL)
    monkeypatch.setattr("hpdlp.model.solve_once", lambda *args, **kwargs: info)
    return info


def test_result_properties_map_to_core_fields(model, core_result):
    for field in (*RESULT_FIELDS, "X", "Pi", "RC", "Status", "StatusName"):
        assert getattr(model, field) is None, field
    assert model.optimize() is model
    for attribute, field in RESULT_FIELDS.items():
        assert getattr(model, attribute) == core_result[field], attribute
    assert model.Status == PDLP.OPTIMAL and model.StatusName == "OPTIMAL"
    for attribute in ("X", "Pi", "RC"):
        np.testing.assert_array_equal(getattr(model, attribute), core_result[attribute])
        assert not getattr(model, attribute).flags.writeable


@pytest.mark.parametrize("attribute,value", [
    ("c", [2., 3.]), ("c0", 4.), ("A", np.array([[2., 0.], [0., 1.], [3., 2.]])),
    ("lb", [0., 0.]), ("ub", [10., 10.]),
    ("constr_lb", [5., -np.inf, -np.inf]), ("constr_ub", [5., 3., 9.]),
    ("ModelSense", PDLP.MAXIMIZE),
])
def test_model_change_clears_result_cache(model, core_result, attribute, value):
    model.optimize()
    setattr(model, attribute, value)
    for field in (*RESULT_FIELDS, "X", "Pi", "RC", "Status", "StatusName"):
        assert getattr(model, field) is None, field


@pytest.mark.parametrize("attribute,value", [("c", [1.]), ("A", np.ones((2, 2))), ("c0", np.nan)])
def test_rejected_change_preserves_result_cache(model, core_result, attribute, value):
    model.optimize()
    previous_x, previous_objective = model.X, model.ObjVal
    with pytest.raises(ValueError):
        setattr(model, attribute, value)
    assert model.X is previous_x and model.ObjVal == previous_objective


def test_failed_solve_clears_previous_result(model, core_result, monkeypatch):
    model.optimize()

    def fail(*args, **kwargs):
        raise RuntimeError("native solve failed")

    monkeypatch.setattr("hpdlp.model.solve_once", fail)
    with pytest.raises(RuntimeError, match="native solve failed"):
        model.optimize()
    assert model.X is None and model.Status is None and model.ObjVal is None


@pytest.mark.gpu
def test_result_arrays_survive_resolve_and_model_release(model):
    model.optimize()
    assert_optimal(model, 3)
    previous = {field: getattr(model, field) for field in ("X", "Pi", "RC")}
    copies = {field: value.copy() for field, value in previous.items()}
    model.c0 = 7
    assert model.X is None
    model.optimize()
    assert_optimal(model, 10)
    current = {field: getattr(model, field) for field in previous}
    current_copies = {field: value.copy() for field, value in current.items()}
    for field in previous:
        assert not np.shares_memory(previous[field], current[field])
        assert not previous[field].flags.writeable
    del model
    gc.collect()
    for field in previous:
        np.testing.assert_array_equal(previous[field], copies[field])
        np.testing.assert_array_equal(current[field], current_copies[field])
