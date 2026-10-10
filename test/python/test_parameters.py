import numpy as np
import pytest


from hpdlp import Model, PDLP
from hpdlp._core import get_default_params, validate_params


def test_set_params_is_transactional(model):
    old_time_limit = model.getParam('TimeLimit')
    with pytest.raises(KeyError):
        model.setParams(TimeLimit=123.0, DefinitelyNotAParam=1)
    assert model.getParam('TimeLimit') == old_time_limit


def test_param_value_validation(model):
    model.setParam('TimeLimit', 12)
    assert model.getParam('TimeLimit') == 12.0
    model.setParam('TimeLimit', np.float64(30.0))
    assert model.getParam('TimeLimit') == 30.0
    model.setParam('OptimalityNorm', 'LINF')
    assert model.getParam('OptimalityNorm') == 'linf'
    model.setParam('OutputFlag', 1)
    assert model.getParam('OutputFlag') is True
    model.setParam('OutputFlag', 0)
    assert model.getParam('OutputFlag') is False
    model.setParam('OutputFlag', np.bool_(True))
    assert model.getParam('OutputFlag') is True
    model.setParam('IterationLimit', np.int64(1000))
    assert model.getParam('IterationLimit') == 1000 and isinstance(model.getParam('IterationLimit'), int)
    model.setParam('IterationLimit', 2000.0)
    assert model.getParam('IterationLimit') == 2000
    with pytest.raises(TypeError):
        model.setParam('IterationLimit', False)
    with pytest.raises(TypeError):
        model.setParam('IterationLimit', 1.5)
    with pytest.raises(TypeError):
        model.setParam('OutputFlag', 'yes')
    with pytest.raises(ValueError):
        model.setParam('OutputFlag', 2)
    with pytest.raises(ValueError):
        model.setParam('IterationLimit', -1)
    with pytest.raises(ValueError):
        model.setParam('TermCheckFreq', 0)
    with pytest.raises(ValueError):
        model.setParam('OptimalityTol', 0.0)
    with pytest.raises(ValueError):
        model.setParam('TimeLimit', -1.0)
    with pytest.raises(ValueError):
        model.setParam('OptimalityNorm', 'l1')


def test_set_params_value_validation_is_transactional(model):
    old_time_limit = model.getParam('TimeLimit')
    with pytest.raises(TypeError):
        model.setParams(TimeLimit=123.0, OutputFlag='yes')
    assert model.getParam('TimeLimit') == old_time_limit


def test_reset_params(base_lp_data):
    c, A, l, u, lb, ub = base_lp_data
    model = Model(c, A, l, u, lb, ub)
    default_time_limit = model.getParam('TimeLimit')
    default_output_flag = model.getParam('OutputFlag')
    model.setParams(TimeLimit=12.0, OutputFlag=not default_output_flag)
    model.resetParams()
    assert model.getParam('TimeLimit') == default_time_limit
    assert model.getParam('OutputFlag') is default_output_flag


def test_params_view(model):
    model.Params.TimeLimit = 123.0
    assert model.Params.TimeLimit == 123.0
    model.Params['OptimalityTol'] = 1e-05
    assert model.Params['OptimalityTol'] == 1e-05
    assert model.getParam('TimeLimit') == 123.0
    assert 'TimeLimit' in model.Params
    assert 'time_sec_limit' in model.Params
    assert 'DefinitelyNotAParam' not in model.Params
    assert len(list(model.Params.keys())) > 0
    assert len(list(model.Params.values())) == len(list(model.Params.keys()))
    assert dict(model.Params.items())['time_sec_limit'] == 123.0
    with pytest.raises(AttributeError):
        _ = model.Params.NoSuchParameter


def _changed_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        if not value or not np.isfinite(value):
            return 1.0
        return value * 2 if value > 1 else value / 2
    return 'linf' if value == 'l2' else 'l2'


def test_param_aliases_resolve_consistently(model):
    for alias, key in PDLP._PARAM_ALIAS.items():
        before = dict(model.Params.items())
        value = _changed_value(before[key])
        model.setParam(alias, value)
        after = dict(model.Params.items())
        assert {name for name in after if after[name] != before[name]} == {key}
        assert model.getParam(alias) == model.getParam(key) == value
        assert getattr(model.Params, alias) == model.Params[alias] == value


def test_param_name_validation(model):
    with pytest.raises(KeyError):
        model.setParam('DefinitelyNotAParam', 1)
    with pytest.raises(KeyError):
        model.getParam('DefinitelyNotAParam')


def test_specific_parameters_and_transactional_validation(base_lp_data):
    model = Model(*base_lp_data)
    model.setParams(FarkasHomogeneousTol=1e-08, FarkasDisplacementTol=1e-09, ConeNewtonTol=1e-08, ConeNewtonMaxIter=150, StepSizeFactor=0.9, Ki=0.1, Kd=0.01, ISmooth=0.2, PrimalWeightSafeguard=False, Diagnose=True)
    assert model.Params.eps_farkas_homogeneous == 1e-08
    assert model.Params.eps_farkas_displacement == 1e-09
    assert model.Params.cone_newton_max_iter == 150
    assert model.Params.k_i == 0.1
    before = dict(model.Params.items())
    with pytest.raises(ValueError):
        model.setParams(TimeLimit=1, StepSizeFactor=1.0)
    assert dict(model.Params.items()) == before
    for name in ('InfeasibleTol', 'FeasibilityPolishing', 'ASBActivationTol'):
        with pytest.raises(KeyError):
            model.setParam(name, 1)


@pytest.mark.parametrize('params', [{'eps_farkas_homogeneous': 0}, {'eps_farkas_displacement': -1}, {'cone_newton_max_iter': 0}, {'cone_newton_tol': float('nan')}, {'step_size_factor': 1}, {'i_smooth': 1}, {'k_i': float('inf')}, {'asb_relative_tol': -1}, {'optimality_norm': 'l1'}, {'sufficient_reduction_for_restart': 0.8, 'necessary_reduction_for_restart': 0.4}])
def test_core_parameter_ranges(params):
    with pytest.raises(ValueError):
        validate_params(params)


def test_defaults_are_usable_without_gpu_work(base_lp_data):
    defaults = get_default_params()
    validate_params(defaults)
    model = Model(*base_lp_data)
    assert dict(model.Params.items()) == defaults


@pytest.mark.parametrize("alias,canonical,value", [
    ("TimeLimit", "time_sec_limit", 12.),
    ("OptimalityTol", "eps_optimal_relative", 1e-5),
    ("FeasibilityTol", "eps_feasible_relative", 1e-5),
    ("Presolve", "presolve", False),
    ("ActiveSetBoost", "active_set_boost", False),
    ("OptimalityNorm", "optimality_norm", "linf"),
])
def test_documented_aliases(model, alias, canonical, value):
    model.setParam(alias, value)
    assert model.getParam(canonical) == value
    model.setParam(canonical, value)
    assert model.getParam(alias) == value
