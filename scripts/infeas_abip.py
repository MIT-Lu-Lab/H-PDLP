"""ctypes binding of ABIP-LP, whose C sources build_abip.sbatch compiles with -DDLONG into abip/libabip.so."""
import ctypes
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

LIB = Path(__file__).resolve().parents[1] / "abip" / "libabip.so"

abip_int = ctypes.c_int64      # -DDLONG makes abip_int a 64-bit long
abip_float = ctypes.c_double

# Status codes of include/glbopts.h: a stopped solve is a timeout, while an inaccurate certificate keeps its name for scoring.
STATUS = {
    1: "OPTIMAL",                          # ABIP_SOLVED
    2: "TIME_LIMIT",                       # ABIP_SOLVED_INACCURATE
    0: "TIME_LIMIT",                       # ABIP_UNFINISHED
    -1: "DUAL_INFEASIBLE",                 # ABIP_UNBOUNDED
    -2: "PRIMAL_INFEASIBLE",               # ABIP_INFEASIBLE
    -3: "NUMERICAL_ERROR",                 # ABIP_INDETERMINATE
    -4: "NUMERICAL_ERROR",                 # ABIP_FAILED
    -5: "TIME_LIMIT",                      # ABIP_SIGINT
    -6: "DUAL_INFEASIBLE_INACCURATE",      # ABIP_UNBOUNDED_INACCURATE
    -7: "PRIMAL_INFEASIBLE_INACCURATE",    # ABIP_INFEASIBLE_INACCURATE
}


# The structures below mirror include/abip.h and linsys/amatrix.h field by field.
class Matrix(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(abip_float)), ("i", ctypes.POINTER(abip_int)),
                ("p", ctypes.POINTER(abip_int)), ("m", abip_int), ("n", abip_int)]


class Settings(ctypes.Structure):
    _fields_ = [
        ("normalize", abip_int), ("pfeasopt", abip_int), ("scale", abip_float),
        ("rho_y", abip_float), ("sparsity_ratio", abip_float),
        ("max_ipm_iters", abip_int), ("max_admm_iters", abip_int),
        ("max_time", abip_float), ("eps", abip_float), ("alpha", abip_float),
        ("cg_rate", abip_float), ("adaptive", abip_int), ("eps_cor", abip_float),
        ("eps_pen", abip_float), ("dynamic_sigma", abip_float),
        ("dynamic_x", abip_float), ("dynamic_eta", abip_float),
        ("restart_fre", abip_int), ("restart_thresh", abip_int),
        ("verbose", abip_int), ("warm_start", abip_int),
        ("adaptive_lookback", abip_int), ("origin_rescale", abip_int),
        ("pc_ruiz_rescale", abip_int), ("qp_rescale", abip_int),
        ("ruiz_iter", abip_int), ("hybrid_mu", abip_int),
        ("hybrid_thresh", abip_float), ("dynamic_sigma_second", abip_float),
        ("half_update", abip_int), ("avg_criterion", abip_int),
    ]


class Data(ctypes.Structure):
    _fields_ = [("m", abip_int), ("n", abip_int), ("A", ctypes.POINTER(Matrix)),
                ("b", ctypes.POINTER(abip_float)), ("c", ctypes.POINTER(abip_float)),
                ("sp", abip_float), ("stgs", ctypes.POINTER(Settings))]


class Solution(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(abip_float)), ("y", ctypes.POINTER(abip_float)),
                ("s", ctypes.POINTER(abip_float))]


class Info(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_char * 32), ("status_val", abip_int),
        ("ipm_iter", abip_int), ("admm_iter", abip_int),
        ("pobj", abip_float), ("dobj", abip_float), ("res_pri", abip_float),
        ("res_dual", abip_float), ("rel_gap", abip_float),
        ("res_infeas", abip_float), ("res_unbdd", abip_float),
        ("setup_time", abip_float), ("solve_time", abip_float),
    ]


def standard_form(data, cones):
    """(A, b, c) of ABIP's min c'x s.t. Ax = b, x >= 0: the conic form with a slack column for each nonnegative-cone row."""
    A, n_zero = data["A"], cones["z"]
    n_ineq = A.shape[0] - n_zero
    slacks = sp.vstack([sp.csc_matrix((n_zero, n_ineq)), sp.eye(n_ineq, format="csc")], format="csc")
    return sp.hstack([A, slacks], format="csc"), data["b"], np.concatenate([data["c"], np.zeros(n_ineq)])


def solve(data, cones, time_limit, eps=None):
    """(status, IPM iterations, objective, solve seconds) of ABIP on the conic form of infeas_conic.conic_form."""
    A, b, c = standard_form(data, cones)
    m, n = A.shape
    Ax = np.ascontiguousarray(A.data, dtype=np.float64)
    Ai = np.ascontiguousarray(A.indices, dtype=np.int64)
    Ap = np.ascontiguousarray(A.indptr, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    c = np.ascontiguousarray(c, dtype=np.float64)

    lib = ctypes.CDLL(str(LIB))
    lib.abip_main.restype = abip_int
    matrix = Matrix(Ax.ctypes.data_as(ctypes.POINTER(abip_float)), Ai.ctypes.data_as(ctypes.POINTER(abip_int)),
                    Ap.ctypes.data_as(ctypes.POINTER(abip_int)), m, n)
    settings = Settings()
    abip_data = Data(m, n, ctypes.pointer(matrix), b.ctypes.data_as(ctypes.POINTER(abip_float)),
                     c.ctypes.data_as(ctypes.POINTER(abip_float)), A.nnz / (m * n), ctypes.pointer(settings))
    # ABIP fills its defaults first, since the ABIP+ restart and hybrid-mu defaults are literals with no #define.
    lib.abip_set_default_settings(ctypes.byref(abip_data))
    settings.max_time = float(time_limit)
    settings.verbose = 0
    # No iteration cap: the defaults of 500 IPM and 1e6 ADMM iterations stopped runs far inside the time limit.
    settings.max_ipm_iters = settings.max_admm_iters = 2 ** 63 - 1
    if eps is not None:
        settings.eps = float(eps)
    solution, info = Solution(), Info()
    lib.abip_main(ctypes.byref(abip_data), ctypes.byref(solution), ctypes.byref(info))

    status = STATUS.get(info.status_val, f"code_{info.status_val}")
    objective = info.pobj if status == "OPTIMAL" else None
    # solve_time is in milliseconds, and a failure before solving leaves it NaN and ipm_iter at -1.
    solve_time = info.solve_time / 1e3 if math.isfinite(info.solve_time) else None
    return status, max(int(info.ipm_iter), 0), objective, solve_time
