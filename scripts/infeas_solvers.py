"""Baseline LP solver adapters: each solves one MPS file and returns a SolveResult."""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

# CPU baselines get 16 threads as in the cuPDLP.jl experiments, or SLURM_CPUS_PER_TASK when set.
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))

# ABIP overruns its time limit and can crash, so it runs in a subprocess killed after TL + slack for model reading.
ABIP_KILL_SLACK = 900.0


@dataclass(frozen=True)
class SolveResult:
    status: str
    iters: int
    objective: float | None = None
    solve_elapsed: float | None = None   # the optimize call alone
    elapsed: float = 0.0                 # wall clock incl. decompressing and reading, set by solve()


def solve(solver: str, mps_path: Path, time_limit: float, eps: float | None) -> SolveResult:
    """Decompress one .mps.gz and solve it with the solver's own presolve off, recording any exception as ERROR."""
    started_at = time.time()
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".mps")
        with os.fdopen(fd, "wb") as tmp, gzip.open(mps_path, "rb") as src:
            shutil.copyfileobj(src, tmp)
        result = SOLVERS[solver](tmp_path, time_limit, eps, started_at)
        return replace(result, elapsed=time.time() - started_at)
    except Exception as exc:
        return SolveResult(f"ERROR: {exc}", 0, elapsed=time.time() - started_at)
    finally:
        if tmp_path:
            os.unlink(tmp_path)


# Gurobi variants: method 1 dual simplex, 0 primal simplex, 2 barrier with (crossover -1) or without (0) crossover.
def solve_gurobi(tmp_path: str, time_limit: float, eps: float | None, started_at: float,
                 method: int, crossover: int) -> SolveResult:
    import gurobipy as gp
    from gurobipy import GRB

    status_map = {
        GRB.INFEASIBLE: "PRIMAL_INFEASIBLE",
        GRB.UNBOUNDED: "DUAL_INFEASIBLE",
        GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.ITERATION_LIMIT: "TIME_LIMIT",
    }
    params = {"OutputFlag": 0, "TimeLimit": time_limit, "Threads": THREADS, "Method": method,
              "Crossover": crossover, "Presolve": 0}
    if eps is not None:
        params.update(FeasibilityTol=eps, OptimalityTol=eps)
        if method == 2:
            params["BarConvTol"] = eps   # barrier stops on BarConvTol, not Feasibility/OptimalityTol
    with gp.Env(params=params) as env, gp.read(tmp_path, env=env) as model:
        # MIPLIB MPS files carry integer markers, and this study solves the LP relaxation.
        if model.NumIntVars:
            model.setAttr("VType", model.getVars(), ["C"] * model.NumVars)
        solve_started = time.perf_counter()
        model.optimize()
        solve_elapsed = time.perf_counter() - solve_started
        status = status_map.get(model.Status, f"code_{model.Status}")
        # IterCount counts simplex iterations only, so barrier iterations are added in.
        iters = int(model.IterCount) + int(model.BarIterCount)
        return SolveResult(status, iters, model.ObjVal if status == "OPTIMAL" else None, solve_elapsed)


def make_highs(time_limit: float, eps: float | None = None):
    import highspy

    highs = highspy.Highs()
    highs.silent()
    highs.setOptionValue("threads", THREADS)
    highs.setOptionValue("time_limit", time_limit)
    if eps is not None:
        highs.setOptionValue("primal_feasibility_tolerance", eps)
        highs.setOptionValue("dual_feasibility_tolerance", eps)
        highs.setOptionValue("ipm_optimality_tolerance", eps)
    return highs


def read_model(highs, tmp_path: str) -> None:
    """Read an MPS file into HiGHS, raising when it fails to load; a kWarning read still loads the model."""
    import highspy

    if highs.readModel(tmp_path) == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS could not parse {os.path.basename(tmp_path)}")


def solve_highs(tmp_path: str, time_limit: float, eps: float | None, started_at: float) -> SolveResult:
    import highspy

    highs = make_highs(time_limit, eps)
    highs.setOptionValue("presolve", "off")
    read_model(highs, tmp_path)
    lp = highs.getLp()
    continuous = highspy.HighsVarType.kContinuous
    # MIPLIB MPS files carry integer markers, and this study solves the LP relaxation.
    if any(kind != continuous for kind in lp.integrality_):
        highs.changeColsIntegrality(lp.num_col_, list(range(lp.num_col_)), [continuous] * lp.num_col_)
    solve_started = time.perf_counter()
    highs.run()
    solve_elapsed = time.perf_counter() - solve_started
    status = str(highs.getModelStatus()).split(".")[-1]
    info = highs.getInfo()
    # The counters are only meaningful when info is valid, and IPM iterations are added to the simplex count.
    iters = info.simplex_iteration_count + info.ipm_iteration_count if info.valid else 0
    return SolveResult(status, iters, info.objective_function_value if status == "kOptimal" else None, solve_elapsed)


def solve_mosek(tmp_path: str, time_limit: float, eps: float | None, started_at: float) -> SolveResult:
    import mosek

    with mosek.Env() as env, env.Task() as task:
        task.putintparam(mosek.iparam.log, 0)
        task.putintparam(mosek.iparam.num_threads, THREADS)
        task.putdouparam(mosek.dparam.optimizer_max_time, time_limit)
        if eps is not None:
            # intpnt_tol_* govern the linear interior-point path, intpnt_co_tol_* the conic one.
            for param in (
                mosek.dparam.intpnt_tol_pfeas,
                mosek.dparam.intpnt_tol_dfeas,
                mosek.dparam.intpnt_tol_rel_gap,
                mosek.dparam.intpnt_co_tol_pfeas,
                mosek.dparam.intpnt_co_tol_dfeas,
                mosek.dparam.intpnt_co_tol_rel_gap,
            ):
                task.putdouparam(param, eps)
        task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        task.readdata(tmp_path)
        # MIPLIB MPS files carry integer markers, and this study solves the LP relaxation.
        if task.getnumintvar():
            task.putvartypelist(list(range(task.getnumvar())), [mosek.variabletype.type_cont] * task.getnumvar())
        solve_started = time.perf_counter()
        trm = task.optimize()
        solve_elapsed = time.perf_counter() - solve_started
        status = mosek_status(task, trm)
        iters = sum(task.getintinf(item) for item in
                    (mosek.iinfitem.intpnt_iter, mosek.iinfitem.sim_dual_iter, mosek.iinfitem.sim_primal_iter))
        return SolveResult(status, iters, mosek_objective(task) if status == "OPTIMAL" else None, solve_elapsed)


def mosek_status(task, trm) -> str:
    """A certificate or optimum from the interior-point or basic solution, else why MOSEK stopped."""
    import mosek

    prosta_map = {
        mosek.prosta.prim_infeas: "PRIMAL_INFEASIBLE",
        mosek.prosta.prim_and_dual_infeas: "PRIMAL_INFEASIBLE",
        mosek.prosta.dual_infeas: "DUAL_INFEASIBLE",
        mosek.prosta.prim_infeas_or_unbounded: "INFEASIBLE_OR_UNBOUNDED",
    }
    for soltype in (mosek.soltype.itr, mosek.soltype.bas):
        if not task.solutiondef(soltype):
            continue
        if (status := prosta_map.get(task.getprosta(soltype))) is not None:
            return status
        if task.getsolsta(soltype) == mosek.solsta.optimal:
            return "OPTIMAL"
    if trm == mosek.rescode.trm_max_time:
        return "TIME_LIMIT"
    return "UNKNOWN" if trm == mosek.rescode.ok else str(trm).split(".")[-1]


def mosek_objective(task) -> float | None:
    import mosek

    for soltype in (mosek.soltype.bas, mosek.soltype.itr):
        if task.solutiondef(soltype) and task.getsolsta(soltype) == mosek.solsta.optimal:
            return float(task.getprimalobj(soltype))
    return None


def solve_scs(tmp_path: str, time_limit: float, eps: float | None, started_at: float) -> SolveResult:
    import scs
    from lp_utils import highs_to_scs

    highs = make_highs(time_limit)
    read_model(highs, tmp_path)
    data, cones = highs_to_scs(highs)
    obj_offset = data.pop("obj_offset")
    remaining = max(5.0, time_limit - (time.time() - started_at))
    tol = {} if eps is None else {"eps_abs": eps, "eps_rel": eps}
    solve_started = time.perf_counter()
    info = scs.solve(data, cones, max_iters=int(1e9), time_limit_secs=remaining, verbose=False, **tol)["info"]
    solve_elapsed = time.perf_counter() - solve_started
    status_map = {"solved": "OPTIMAL", "infeasible": "PRIMAL_INFEASIBLE", "unbounded": "DUAL_INFEASIBLE"}
    raw_status = info["status"]
    # An "(inaccurate)" certificate is kept as a detection that missed tolerance; an inaccurate solve is a timeout.
    if "inaccurate" in raw_status:
        base = status_map.get(raw_status.split(" (")[0], "")
        status = f"{base}_INACCURATE" if base in ("PRIMAL_INFEASIBLE", "DUAL_INFEASIBLE") else "TIME_LIMIT"
    else:
        status = status_map.get(raw_status, raw_status)
    objective = info["pobj"] + obj_offset if status == "OPTIMAL" else None
    return SolveResult(status, max(info["iter"], 0), objective, solve_elapsed)   # iter is -1 on failure


def solve_abip(tmp_path: str, time_limit: float, eps: float | None, started_at: float) -> SolveResult:
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "abip" / "solve_one.py"),
           tmp_path, str(time_limit)]
    if eps is not None:
        cmd.append(str(eps))
    solve_started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + ABIP_KILL_SLACK)
    except subprocess.TimeoutExpired:
        return SolveResult("TIME_LIMIT", 0, None, time.perf_counter() - solve_started)
    solve_elapsed = time.perf_counter() - solve_started
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return SolveResult(f"ERROR: {detail[:80]}", 0, None, solve_elapsed)
    out = json.loads(line[2:])
    return SolveResult(out["status"], out["iters"], out["objective"], out["solve_time"])


SOLVERS = {
    "abip": solve_abip,
    "gurobi_ds": partial(solve_gurobi, method=1, crossover=-1),
    "gurobi_br": partial(solve_gurobi, method=2, crossover=0),
    "gurobi_cr": partial(solve_gurobi, method=2, crossover=-1),
    "gurobi_ps": partial(solve_gurobi, method=0, crossover=-1),
    "highs": solve_highs,
    "mosek": solve_mosek,
    "scs": solve_scs,
}
