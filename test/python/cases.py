from dataclasses import dataclass

import numpy as np

from hopdlp import Model, PDLP


@dataclass
class LPCase:
    objective: np.ndarray
    matrix: np.ndarray
    row_lower: np.ndarray | None
    row_upper: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    expected_objective: float
    constant: float = 0
    sense: int = PDLP.MINIMIZE

    def build_model(self, **parameters):
        model = Model(self.objective, self.matrix, self.row_lower, self.row_upper,
                      self.lower, self.upper, objective_constant=self.constant)
        model.ModelSense = self.sense
        model.setParams(OutputFlag=False, Presolve=False,
                        OptimalityTol=1e-7, FeasibilityTol=1e-7,
                        IterationLimit=20000, TimeLimit=10)
        model.setParams(**parameters)
        return model


def lp_case(name):
    if name in ("minimize", "maximize"):
        return LPCase(np.array([1., 1.]), np.array([[1., 2.], [0., 1.], [3., 2.]]),
                      np.array([5., -np.inf, -np.inf]), np.array([5., 2., 8.]),
                      np.zeros(2), None, 5.25 if name == "maximize" else 5,
                      constant=2, sense=PDLP.MAXIMIZE if name == "maximize" else PDLP.MINIMIZE)
    if name == "box":
        return LPCase(np.array([1., -2.]), np.empty((0, 2)), None, None,
                      np.array([-3., 2.]), np.array([1., 4.]), -11)
    if name == "fixed_and_free":
        return LPCase(np.array([1., 2.]), np.array([[1., 1.]]),
                      np.array([3.]), np.array([3.]), np.array([1., -np.inf]),
                      np.array([1., np.inf]), 5)
    if name == "degenerate":
        return LPCase(np.ones(2), np.ones((1, 2)), np.array([1.]), None,
                      np.zeros(2), None, 1)
    if name == "zero_columns":
        return LPCase(np.empty(0), np.empty((1, 0)), np.array([0.]),
                      np.array([1.]), None, None, 3, constant=3)
    if name == "empty":
        return LPCase(np.empty(0), np.empty((0, 0)), None, None, None, None, 2, constant=2)
    raise KeyError(name)
