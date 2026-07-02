#!/usr/bin/env python3
"""Run cuHSDLP on netlib infeasible instances, with and without presolve.

Writes two CSVs into ../result/:
  infeas_results.csv          (no presolve)
  infeas_results_presolve.csv (with presolve)
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
BINARY = PROJ / "build" / "cuhsdlp"
INSTANCES_DIR = PROJ / "test" / "netlib_infeas"
RESULT_DIR = PROJ / "result"

SUMMARY_KEY_MAP = {
    "Termination Reason":        ("status",                str),
    "Presolve Time (sec)":       ("presolve_time_sec",     float),
    "Reduced Rows":              ("reduced_rows",          int),
    "Reduced Columns":           ("reduced_cols",          int),
    "Reduced Nonzeros":          ("reduced_nnz",           int),
    "Precondition time (sec)":   ("precondition_time_sec", float),
    "Runtime (sec)":             ("solve_time_sec",        float),
    "Iterations Count":          ("iterations",            int),
    "Primal Objective Value":    ("primal_obj",            float),
    "Dual Objective Value":      ("dual_obj",              float),
    "Relative Primal Residual":  ("rel_primal_res",        float),
    "Relative Dual Residual":    ("rel_dual_res",          float),
    "Restarts":                  ("restart_count",         int),
    "Rows":                      ("rows",                  int),
    "Columns":                   ("cols",                  int),
    "Nonzeros":                  ("nnz",                   int),
}

COLS_NO_PRESOLVE = [
    "problem", "status", "iterations", "solve_time_sec",
    "rows", "cols", "nnz",
    "restart_count", "precondition_time_sec",
    "rel_primal_res", "rel_dual_res",
    "primal_obj", "dual_obj",
]

COLS_PRESOLVE = [
    "problem", "status", "iterations", "solve_time_sec", "presolve_time_sec",
    "reduced_rows", "reduced_cols", "reduced_nnz",
    "rows", "cols", "nnz",
    "restart_count", "precondition_time_sec",
    "rel_primal_res", "rel_dual_res",
    "primal_obj", "dual_obj",
]


def parse_summary(path):
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in SUMMARY_KEY_MAP:
                col, typ = SUMMARY_KEY_MAP[k]
                out[col] = v if typ is str else typ(v)
    return out


def run_instance(mps, output_dir, presolve, iter_limit, time_limit):
    cmd = [str(BINARY),
           "--iter_limit", str(iter_limit),
           "--time_limit", str(time_limit)]
    if not presolve:
        cmd.append("--no_presolve")
    cmd += [str(mps), str(output_dir) + "/"]
    return subprocess.run(cmd, capture_output=True, text=True)


def run_and_parse(mps, output_dir, presolve, iter_limit, time_limit):
    """Run solver and parse summary; return a row dict.

    On any failure (non-zero exit, missing summary, parse error), record
    status='SCRIPT_ERROR' with diagnostic notes so the batch can continue."""
    name = mps.stem
    summary_path = output_dir / f"{name}_summary.txt"
    if summary_path.exists():
        summary_path.unlink()  # ensure we read fresh output, not stale
    try:
        proc = run_instance(mps, output_dir, presolve, iter_limit, time_limit)
    except Exception as e:
        return {"problem": name, "status": "SCRIPT_ERROR",
                "stderr_tail": f"subprocess raised: {e}"}
    if not summary_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-400:].replace("\n", " | ")
        return {"problem": name, "status": "SCRIPT_ERROR",
                "stderr_tail": f"exit={proc.returncode}; no summary; tail: {tail}"}
    try:
        d = parse_summary(summary_path)
    except Exception as e:
        return {"problem": name, "status": "SCRIPT_ERROR",
                "stderr_tail": f"parse error: {e}"}
    d["problem"] = name
    return d


def fmt(val):
    if isinstance(val, float):
        return f"{val:.3g}"
    return val


def write_csv(rows, cols, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter_limit", type=int, default=1_000_000)
    ap.add_argument("--time_limit", type=float, default=3600.0)
    ap.add_argument("--out_no_presolve", default="/tmp/netlib_infeas_no_presolve")
    ap.add_argument("--out_presolve",    default="/tmp/netlib_infeas_presolve")
    args = ap.parse_args()

    if not BINARY.exists():
        sys.exit(f"binary not found: {BINARY}")
    if not INSTANCES_DIR.is_dir():
        sys.exit(f"instances dir not found: {INSTANCES_DIR}")

    out_no = Path(args.out_no_presolve); out_no.mkdir(parents=True, exist_ok=True)
    out_yes = Path(args.out_presolve);   out_yes.mkdir(parents=True, exist_ok=True)

    instances = sorted(p for p in INSTANCES_DIR.glob("*.mps")
                       if p.stem and p.stat().st_size > 0)
    print(f"{len(instances)} instances", flush=True)

    rows_no, rows_yes = [], []
    for i, mps in enumerate(instances, 1):
        name = mps.stem
        print(f"[{i:2}/{len(instances)}] {name}  (no presolve)", flush=True)
        d = run_and_parse(mps, out_no, presolve=False,
                          iter_limit=args.iter_limit, time_limit=args.time_limit)
        if d.get("status") == "SCRIPT_ERROR":
            print(f"  WARNING: {d.get('stderr_tail','')}", flush=True)
        rows_no.append(d)

        print(f"[{i:2}/{len(instances)}] {name}  (presolve)", flush=True)
        d = run_and_parse(mps, out_yes, presolve=True,
                          iter_limit=args.iter_limit, time_limit=args.time_limit)
        if d.get("status") == "SCRIPT_ERROR":
            print(f"  WARNING: {d.get('stderr_tail','')}", flush=True)
        rows_yes.append(d)

    write_csv(rows_no,  COLS_NO_PRESOLVE, RESULT_DIR / "infeas_results.csv")
    write_csv(rows_yes, COLS_PRESOLVE,    RESULT_DIR / "infeas_results_presolve.csv")
    print(f"wrote {RESULT_DIR/'infeas_results.csv'}", flush=True)
    print(f"wrote {RESULT_DIR/'infeas_results_presolve.csv'}", flush=True)


if __name__ == "__main__":
    main()
