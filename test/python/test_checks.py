import numpy as np
import pytest

from checks import assert_optimal
from hopdlp import PDLP


def optimal_result():
    return dict(X=np.array([1., 2.]), Pi=np.array([1., -1., 0.]), RC=np.zeros(2),
                Status="OPTIMAL", StatusCode=PDLP.OPTIMAL,
                PrimalObj=3., DualObj=3., ObjectiveGap=0., RelativeObjectiveGap=0.)


@pytest.mark.parametrize("defect", [None, "primal", "dual", "objective"])
def test_math_check_detects_bad_results_without_repair(model, monkeypatch, defect):
    info = optimal_result()
    if defect == "primal":
        info["X"][0] = 0
    elif defect == "dual":
        info["Pi"][:] = 0
    elif defect == "objective":
        info["PrimalObj"] = 4
    monkeypatch.setattr("hopdlp.model.solve_once", lambda *args, **kwargs: info)
    model.optimize()
    original = {field: getattr(model, field).copy() for field in ("X", "Pi", "RC")}
    if defect is None:
        assert_optimal(model, 3)
    else:
        with pytest.raises(AssertionError):
            assert_optimal(model, 3)
    for field in original:
        np.testing.assert_array_equal(getattr(model, field), original[field])
    assert model.Status == PDLP.OPTIMAL
    assert model.ObjVal == info["PrimalObj"]
