"""Convert a HiGHS LP into the conic form min c'x s.t. Ax + s = b, s in K (zero cone, then nonnegative cone) of SCS and ABIP."""
import highspy
import numpy as np
import scipy.sparse as sp

INF = 1e30


def conic_form(highs):
    """(data, cones, offset, sign) of the loaded LP, whose objective value is sign * (conic objective + offset)."""
    lp = highs.getLp()
    sign = -1.0 if lp.sense_ == highspy.ObjSense.kMaximize else 1.0
    c = sign * np.array(lp.col_cost_, dtype=float)
    offset = sign * float(lp.offset_)
    xl = np.clip(np.array(lp.col_lower_, dtype=float), -INF, INF)
    xu = np.clip(np.array(lp.col_upper_, dtype=float), -INF, INF)
    bl = np.clip(np.array(lp.row_lower_, dtype=float), -INF, INF)
    bu = np.clip(np.array(lp.row_upper_, dtype=float), -INF, INF)
    mat = lp.a_matrix_
    A = sp.csc_matrix((np.array(mat.value_, dtype=float), np.array(mat.index_, dtype=np.int32),
                       np.array(mat.start_, dtype=np.int32)), shape=(lp.num_row_, lp.num_col_))

    # Fixed columns are substituted into the row bounds and the objective offset.
    fixed = np.abs(xu - xl) <= 1e-10 * (1 + np.abs(xu))
    fix_idx, keep_idx = np.where(fixed)[0], np.where(~fixed)[0]
    offset += float(c[fix_idx] @ xl[fix_idx])
    b_fix = A[:, fix_idx] @ xl[fix_idx]
    bl, bu = bl - b_fix, bu - b_fix
    A, c, xl, xu = A[:, keep_idx], c[keep_idx], xl[keep_idx], xu[keep_idx]

    # Each bounded column becomes a nonnegative distance from a bound: x = xl + x', or x = xu - x' with only an upper bound.
    lower, upper = xl > -INF / 2, xu < INF / 2
    reflect = ~lower & upper
    anchor = np.where(lower, xl, np.where(reflect, xu, 0.0))
    offset += float(c @ anchor)
    b_shift = A @ anchor
    bl, bu = bl - b_shift, bu - b_shift
    flip = np.where(reflect, -1.0, 1.0)
    A.data *= np.repeat(flip, np.diff(A.indptr))
    c = c * flip
    width = np.where(lower & upper, xu - xl, INF)

    # Free columns are split into x = p - q with p, q >= 0.
    free_idx = np.where(~lower & ~upper)[0]
    A = sp.hstack([A, -A[:, free_idx]], format="csc")
    c = np.concatenate([c, -c[free_idx]])
    width = np.concatenate([width, np.full(len(free_idx), INF)])
    n = A.shape[1]

    eq = np.abs(bu - bl) < 1e-8 * (1 + np.abs(bu))
    eq_idx = np.where(eq)[0]
    hi_idx = np.where(~eq & (bl <= -INF / 2) & (bu < INF / 2))[0]
    lo_idx = np.where(~eq & (bl > -INF / 2) & (bu >= INF / 2))[0]
    bt_idx = np.where(~eq & (bl > -INF / 2) & (bu < INF / 2))[0]
    ub_idx = np.where(width < INF / 2)[0]
    eye = sp.eye(n, format="csc")
    # Rows: equalities (zero cone), then Ax <= bu, -Ax <= -bl, both sides of ranged rows, x' <= width and -x <= 0.
    blocks = [(A[eq_idx, :], bu[eq_idx]), (A[hi_idx, :], bu[hi_idx]), (-A[lo_idx, :], -bl[lo_idx]),
              (A[bt_idx, :], bu[bt_idx]), (-A[bt_idx, :], -bl[bt_idx]), (eye[ub_idx, :], width[ub_idx]),
              (-eye, np.zeros(n))]
    data = {"c": c, "A": sp.vstack([a for a, _ in blocks], format="csc"), "b": np.concatenate([b for _, b in blocks])}
    return data, {"z": len(eq_idx), "l": len(data["b"]) - len(eq_idx)}, offset, sign
