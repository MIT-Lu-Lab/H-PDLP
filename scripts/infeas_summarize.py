#!/usr/bin/env python3.12
"""Summarize an infeasible benchmark into one CSV: status, iters and time per baseline solver and arm."""
import argparse
import collections
import csv
from pathlib import Path

from common import RESULT_DIR
from infeas_datasets import DATASETS, INFEAS_DATASETS, expected_tags, is_unbounded
from infeas_run import build_config, merged_rows, result_files

# Solvers in the summary with their tolerance: None is the solver default, ABIP runs at its authors' 1e-4.
SUMMARY_SOLVERS = [("gurobi_ds", None), ("gurobi_br", None), ("gurobi_ps", None),
                   ("highs", None), ("mosek", None), ("scs", None), ("abip", 1e-4)]
ARMS = ("raw", "pslp")


def summary_csv(dataset, date):
    return RESULT_DIR / f"summary_infeas_{dataset}_{date}.csv"


def collect(dataset, date, solver, arm, eps, tags):
    """The row per expected instance that one configuration's merge would keep."""
    config = build_config(solver, dataset, date, mode=arm, eps=eps)
    rows = merged_rows(result_files(config), config, tags)
    return {tag: {"status": row[config.status_col],
                  "iters": row.get("iters") or "",
                  "time": row.get("solve_time") or row.get("elapsed") or ""}
            for tag, row in rows.items()}


def run(date, dataset="miplib", out=None):
    configs = [(f"{solver}_{arm}", solver, arm, eps) for solver, eps in SUMMARY_SOLVERS for arm in ARMS]
    keys = [key for key, _, _, _ in configs]
    arm_tags = {arm: expected_tags(DATASETS[dataset], arm) for arm in ARMS}
    tags = set().union(*arm_tags.values())
    rows = {}   # tag -> {key -> {status,iters,time}}
    for key, solver, arm, eps in configs:
        for tag, record in collect(dataset, date, solver, arm, eps, arm_tags[arm]).items():
            rows.setdefault(tag, {})[key] = record

    out = out or summary_csv(dataset, date)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "kind"] + [f"{key}_{field}" for key in keys for field in ("status", "iters", "time")])
        for tag in sorted(tags):
            line = [tag, "unbounded" if is_unbounded(tag) else "infeasible"]
            for key in keys:
                r = rows.get(tag, {}).get(key, {})
                line += [r.get("status", ""), r.get("iters", ""), r.get("time", "")]
            w.writerow(line)

    print(f"dataset={dataset}  date={date}  rows={len(tags)}  ->  {out}")
    for key in keys:
        found = [rows.get(tag, {}).get(key) for tag in tags]
        count = collections.Counter(r["status"] for r in found if r and r["status"])
        have = sum(1 for r in found if r)
        summary = "  ".join(f"{k}={v}" for k, v in count.most_common(4)) or "(no data)"
        print(f"  {key:18s}: {have}/{len(tags)} done | {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="date tag the runs were submitted under, e.g. 20260904")
    ap.add_argument("--dataset", default="miplib", choices=INFEAS_DATASETS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.date, args.dataset, args.out)


if __name__ == "__main__":
    main()
