#!/usr/bin/env python3.12
"""Run a result pipeline: bench (comparison, tables, scatter, trajectories), infeas (summary, figures) or delta."""
import argparse

import bench_compare
import bench_delta
import bench_scatter
import bench_tables
import bench_trajectories
import infeas_figures
import infeas_summarize
from infeas_datasets import INFEAS_DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="pipeline", required=True)

    bench = sub.add_parser("bench", help="cuHSDLP vs cuPDLPx comparison CSVs, tables, scatter plots and trajectories")
    bench.add_argument("--date", required=True, help="cuHSDLP result date, e.g. 20260712")
    bench.add_argument("--tol", default=bench_compare.DEFAULT_TOL, help="tolerance the cuHSDLP runs used")
    bench.add_argument("--suites", nargs="+", default=list(bench_compare.SUITE_LABEL),
                       choices=sorted(bench_compare.SUITE_LABEL))
    bench.add_argument("--infeas-date", default=None, help="date of the infeasible-suite traces, default --date")

    infeas = sub.add_parser("infeas", help="baseline infeasibility summary CSV and figures")
    infeas.add_argument("--date", required=True, help="date tag the baseline runs were submitted under")
    infeas.add_argument("--dataset", default="miplib", choices=INFEAS_DATASETS)

    delta = sub.add_parser("delta", help="per-instance diff of two cuHSDLP result dates")
    delta.add_argument("--suite", required=True)
    delta.add_argument("--new", required=True, help="newer date")
    delta.add_argument("--old", required=True, help="older date")

    args = parser.parse_args()
    if args.pipeline == "bench":
        for suite in args.suites:
            bench_compare.run(suite, args.date, args.tol)
        bench_tables.run(args.date, args.tol, args.suites)
        for suite in args.suites:
            bench_scatter.run(suite, args.date)
        bench_trajectories.run(args.date, args.infeas_date)
    elif args.pipeline == "infeas":
        infeas_summarize.run(args.date, args.dataset)
        infeas_figures.run(args.date, args.dataset)
    else:
        bench_delta.run(args.suite, args.new, args.old)


if __name__ == "__main__":
    main()
