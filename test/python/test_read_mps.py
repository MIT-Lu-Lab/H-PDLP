import gzip


import numpy as np


import pytest


import scipy.sparse as sp


import hopdlp


from hopdlp import Model, PDLP


MPS_MIN = """NAME          TESTLP
ROWS
 N  COST
 E  R1
 L  R2
 L  R3
COLUMNS
    X1        COST      1.0   R1        1.0
    X1        R3        3.0
    X2        COST      1.0   R1        2.0
    X2        R2        1.0   R3        2.0
RHS
    RHS       R1        5.0   R2        2.0
    RHS       R3        8.0
ENDATA
"""


MPS_MAX = MPS_MIN.replace(
    "ROWS\n",
    "OBJSENSE\n    MAXIMIZE\nROWS\n",
)


MPS_NAMELESS = """NAME          TESTLP
ROWS
 N  COST
 E  R1
 L  R2
 L  R3
COLUMNS
    X1        COST      1.0   R1        1.0
    X1        R3        3.0
    X2        COST      1.0   R1        2.0
    X2        R2        1.0   R3        2.0
RHS
              R1        5.0   R2        2.0
              R3        8.0   COST     -1.5
RANGES
              R3        6.0
BOUNDS
 UP           X1        4.0
 LO           X2        0.5
 MI           X1
ENDATA
"""


@pytest.fixture
def mps_min_file(tmp_path):
    path = tmp_path / 'test_min.mps'
    path.write_text(MPS_MIN)
    return path


@pytest.fixture
def mps_max_file(tmp_path):
    path = tmp_path / 'test_max.mps'
    path.write_text(MPS_MAX)
    return path


@pytest.fixture
def mps_gz_file(tmp_path):
    path = tmp_path / 'test_min.mps.gz'
    with gzip.open(path, 'wt') as f:
        f.write(MPS_MIN)
    return path


def _check_min_model_data(model):
    assert isinstance(model, Model)
    assert model.num_vars == 2
    assert model.num_constrs == 3
    assert np.allclose(model.c, [1.0, 1.0])
    assert model.c0 == 0.0
    A = model.A.toarray() if sp.issparse(model.A) else np.asarray(model.A)
    assert np.allclose(A, [[1.0, 2.0], [0.0, 1.0], [3.0, 2.0]])
    assert np.allclose(model.constr_lb, [5.0, -np.inf, -np.inf])
    assert np.allclose(model.constr_ub, [5.0, 2.0, 8.0])
    assert np.allclose(model.lb, [0.0, 0.0])
    assert np.allclose(model.ub, [np.inf, np.inf])


def test_read_builds_model(mps_min_file):
    model = hopdlp.read(str(mps_min_file))
    _check_min_model_data(model)
    assert model.ModelSense == PDLP.MINIMIZE


def test_read_accepts_pathlike(mps_min_file):
    model = hopdlp.read(mps_min_file)
    _check_min_model_data(model)


def test_read_gzip(mps_gz_file):
    model = hopdlp.read(str(mps_gz_file))
    _check_min_model_data(model)


def test_read_objsense_maximize(mps_max_file):
    model = hopdlp.read(str(mps_max_file))
    _check_min_model_data(model)
    assert model.ModelSense == PDLP.MAXIMIZE


def test_read_sections_without_set_name(tmp_path):
    path = tmp_path / 'nameless.mps'
    path.write_text(MPS_NAMELESS)
    model = hopdlp.read(str(path))
    assert model.num_vars == 2
    assert model.num_constrs == 3
    assert np.allclose(model.c, [1.0, 1.0])
    assert np.isclose(model.c0, 1.5)
    assert np.allclose(model.constr_lb, [5.0, -np.inf, 2.0])
    assert np.allclose(model.constr_ub, [5.0, 2.0, 8.0])
    assert np.allclose(model.lb, [-np.inf, 0.5])
    assert np.allclose(model.ub, [4.0, np.inf])


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hopdlp.read(str(tmp_path / 'no_such_file.mps'))


MPS_BOUNDS = """NAME          BOUNDS
ROWS
 N  COST
 L  R1
COLUMNS
    C0        COST      1.0   R1        1.0
    MARKER                 'MARKER'                 'INTORG'
    I1        COST      1.0   R1        1.0
    I2        COST      1.0   R1        1.0
    I3        COST      1.0   R1        1.0
    I4        COST      1.0   R1        1.0
    I5        COST      1.0   R1        1.0
    MARKER                 'MARKER'                 'INTEND'
    C6        COST      1.0   R1        1.0
    C7        COST      1.0   R1        1.0
    C8        COST      1.0   R1        1.0
RHS
    RHS       R1        5.0
BOUNDS
 UP BND       I2        5.0
 LO BND       I3        2.0
 MI BND       I4
 UP BND       I5        -3.0
 UP BND       C6        -3.0
 LO BND       C7        -10.0
 UP BND       C7        -3.0
 LO BND       C8        -1e30
 UP BND       C8        1e20
ENDATA
"""


def _read_text(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return hopdlp.read(path)


def test_read_bounds_conventions(tmp_path):
    model = _read_text(tmp_path, 'bounds.mps', MPS_BOUNDS)
    inf = np.inf
    expected_lb = [0.0, 0.0, 0.0, 2.0, -inf, -inf, -inf, -10.0, -1e+30]
    expected_ub = [inf, 1.0, 5.0, inf, inf, -3.0, -3.0, -3.0, 1e+20]
    assert np.array_equal(model.lb, expected_lb), model.lb
    assert np.array_equal(model.ub, expected_ub), model.ub


MPS_NO_OBJ_ROW = """NAME          NOOBJ
ROWS
 E  R1
 L  R2
COLUMNS
    X1        R1        1.0   R2        1.0
    X2        R1        2.0
RHS
    RHS       R1        5.0   R2        2.0
ENDATA
"""


def test_read_without_objective_row(tmp_path):
    model = _read_text(tmp_path, 'noobj.mps', MPS_NO_OBJ_ROW)
    assert model.num_vars == 2
    assert model.num_constrs == 2
    assert np.allclose(model.c, [0.0, 0.0])
    assert np.allclose(model.constr_lb, [5.0, -np.inf])
    assert np.allclose(model.constr_ub, [5.0, 2.0])


MPS_SOS = MPS_MIN.replace(
    "ENDATA\n",
    "SOS\n S1 SOS       s1\n    X1        1\n    X2        2\nENDATA\n",
)


MPS_QUADOBJ = MPS_MIN.replace(
    "ENDATA\n",
    "QUADOBJ\n    X1        X1        2.0\nENDATA\n",
)


def test_read_sos_section_is_ignored(tmp_path):
    model = _read_text(tmp_path, 'sos.mps', MPS_SOS)
    _check_min_model_data(model)


def test_read_unsupported_section_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _read_text(tmp_path, 'quadobj.mps', MPS_QUADOBJ)


@pytest.mark.parametrize("selector,expected_c,expected_offset", [
    ("", 1., -5.), ("OBJNAME\n SECOND\n", -2., -7.),
    ("OBJNAME SECOND\n", -2., -7.),
])
def test_objective_selection(tmp_path, selector, expected_c, expected_offset):
    text = ("NAME OBJ\n" + selector + "ROWS\n N FIRST\n N SECOND\n L LIMIT\n"
            "COLUMNS\n X FIRST 1 SECOND -2\n X LIMIT 3\n"
            "RHS\n R LIMIT 4 FIRST 5\n R SECOND 7\nENDATA\n")
    model = _read_text(tmp_path, "objectives.mps", text)
    np.testing.assert_array_equal(model.c, [expected_c])
    assert model.c0 == expected_offset and model.num_constrs == 1
    np.testing.assert_array_equal(model.A.toarray(), [[3.]])


@pytest.mark.parametrize("name", ["MISSING", "LIMIT"])
def test_invalid_objective_selection(tmp_path, name):
    text = ("NAME OBJ\nOBJNAME\n " + name + "\nROWS\n N FIRST\n L LIMIT\n"
            "COLUMNS\n X FIRST 1 LIMIT 3\nENDATA\n")
    with pytest.raises(RuntimeError):
        _read_text(tmp_path, "objectives.mps", text)


@pytest.mark.gpu
@pytest.mark.parametrize("sense,expected", [(PDLP.MINIMIZE, 3.), (PDLP.MAXIMIZE, 3.25)])
def test_read_then_solve(mps_min_file, sense, expected):
    from checks import assert_optimal

    model = hopdlp.read(mps_min_file)
    model.ModelSense = sense
    model.setParams(OutputFlag=False, Presolve=False, OptimalityTol=1e-7, FeasibilityTol=1e-7)
    model.optimize()
    assert_optimal(model, expected)
