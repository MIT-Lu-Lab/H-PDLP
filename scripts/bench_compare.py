#!/usr/bin/env python3
"""Compare cuHSDLP and cuPDLPx benchmark results in one side-by-side CSV per suite."""
import argparse
import csv
import math
import sys
from pathlib import Path

from common import RESULT_DIR, read_csv_rows, read_summary

SUITE_LABEL = {"miplib": "MIPLIB", "mittelmann": "Mittelmann"}
SOLVERS = ("HSD", "PDLPx")
DEFAULT_TOL = "1e-4"
# Achterberg shifts of the shifted geometric mean for time in seconds and for iterations.
TIME_SHIFT, ITER_SHIFT = 10, 100

# metric -> (cuHSDLP summary key, cuPDLPx CSV column)
METRICS = {
    "status":     ("Termination Reason",       "termination_reason"),
    "iters":      ("Iterations Count",         "total_count"),
    "time":       ("Runtime (sec)",            "time_sec"),
    "primal_obj": ("Primal Objective Value",   "primal_obj"),
    "dual_obj":   ("Dual Objective Value",     "dual_objective_value"),
    "rel_gap":    ("Relative Objective Gap",   "relative_objective_gap"),
    "rel_pr_res": ("Relative Primal Residual", "relative_primal_residual"),
    "rel_du_res": ("Relative Dual Residual",   "relative_dual_residual"),
}


def convert(metric, value):
    if metric == "status":
        return value
    if metric == "iters":
        return int(float(value)) if value else 0
    return float(value) if value else 0.0


def comparison_csv(suite, date):
    return RESULT_DIR / f"comparison_cuHSDLP_vs_cuPDLPx_{suite}_{date}.csv"


def load_comparison(suite, date):
    """Rows of the comparison CSV of one suite and date, with the metrics converted back to numbers."""
    path = comparison_csv(suite, date)
    if not path.is_file():
        sys.exit(f"Error: {path} not found")
    return [{"instance": row["instance"], **{f"{s}_{m}": convert(m, row[f"{s}_{m}"]) for m in METRICS for s in SOLVERS}}
            for row in read_csv_rows(path)]


# Achterberg/SCIP shifted geometric mean: exp(mean(log(x + s))) - s.
def shifted_geomean(values, shift):
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


# SGM rows are those where some solver reached OPTIMAL or hit the time limit; the cap is the largest timed-out time.
def sgm_population(rows):
    pop = [r for r in rows if {"OPTIMAL", "TIME_LIMIT"} & {r[f"{s}_status"] for s in SOLVERS}]
    timeouts = [r[f"{s}_time"] for r in pop for s in SOLVERS if r[f"{s}_status"] == "TIME_LIMIT"]
    return pop, max(timeouts, default=None)


# A timeout is charged its own recorded limit (MIPLIB limits differ by class), any other failure the cap.
def charged_time(row, solver, cap):
    return row[f"{solver}_time"] if row[f"{solver}_status"] in ("OPTIMAL", "TIME_LIMIT") or cap is None else cap


def load_hsd(suite, date):
    """cuHSDLP summaries in result/{suite}_{date} as {instance: metrics}, and whether presolve ran."""
    folder = RESULT_DIR / f"{suite}_{date}"
    if not folder.is_dir():
        sys.exit(f"Error: {folder} not found")
    results = {}
    presolved = False
    for path in sorted(folder.glob("*_summary.txt")):
        raw = read_summary(path)
        # A summary carries a presolve block only when presolve ran.
        presolved |= "Presolve Time (sec)" in raw
        results[path.name.removesuffix("_summary.txt")] = {
            m: convert(m, raw.get(key, "")) for m, (key, _) in METRICS.items()}
    return results, presolved


def load_pdlpx(suite, config):
    """cuPDLPx rows of one config as {instance: metrics}."""
    path = RESULT_DIR / f"cuPDLPx_result_{suite}.csv"
    if not path.is_file():
        sys.exit(f"Error: {path} not found")
    results = {row["instance"]: {m: convert(m, row.get(col, "")) for m, (_, col) in METRICS.items()}
               for row in read_csv_rows(path) if row["config"] == config}
    if not results:
        sys.exit(f"Error: no rows for config {config} in {path}")
    return results


def combine(hsd, pdlpx):
    """One row per instance both solvers ran, with HSD_*/PDLPx_* metrics interleaved."""
    sources = dict(zip(SOLVERS, (hsd, pdlpx)))
    return [{"instance": inst, **{f"{s}_{m}": sources[s][inst][m] for m in METRICS for s in SOLVERS}}
            for inst in sorted(set(hsd) & set(pdlpx))]


def fmt(value, metric):
    if metric in ("status", "iters"):
        return str(value)
    return f"{value:.4f}" if metric == "time" else f"{value:.6e}"


def ratio(a, b):
    return f"{a / b:.2f}" if b > 1e-12 else ""


def write_csv(rows, path):
    fields = ["instance"] + [f"{s}_{m}" for m in METRICS for s in SOLVERS] + ["iter_ratio", "time_ratio"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {"instance": r["instance"]}
            out.update({f"{s}_{m}": fmt(r[f"{s}_{m}"], m) for m in METRICS for s in SOLVERS})
            out["iter_ratio"] = ratio(r["HSD_iters"], r["PDLPx_iters"])
            out["time_ratio"] = ratio(r["HSD_time"], r["PDLPx_time"])
            w.writerow(out)


def sgm_ratio(values, shift):
    return shifted_geomean(values["HSD"], shift) / max(shifted_geomean(values["PDLPx"], shift), 1e-12)


def print_summary(rows):
    solved = {s: sum(1 for r in rows if r[f"{s}_status"] == "OPTIMAL") for s in SOLVERS}
    print(f"Shared instances: {len(rows)}")
    print(f"OPTIMAL: HSD {solved['HSD']}/{len(rows)},  PDLPx {solved['PDLPx']}/{len(rows)}")
    print()

    # The both-OPTIMAL intersection is reference only: it drops every instance one solver fails.
    both = [r for r in rows if r["HSD_status"] == "OPTIMAL" and r["PDLPx_status"] == "OPTIMAL"]
    if both:
        times = {s: [r[f"{s}_time"] for r in both] for s in SOLVERS}
        hi, pi = (sum(r[f"{s}_iters"] for r in both) for s in SOLVERS)
        ht, pt = sum(times["HSD"]), sum(times["PDLPx"])
        print(f"Both OPTIMAL ({len(both)}) [reference, intersection only]:")
        print(f"  Iters: HSD {hi:>12,}  PDLPx {pi:>12,}  ratio {hi / max(pi, 1):.2f}x")
        print(f"  Time:  HSD {ht:>11.1f}s  PDLPx {pt:>11.1f}s  ratio {ht / max(pt, 1):.2f}x")
        print(f"  SGM time  (s={TIME_SHIFT}):  ratio {sgm_ratio(times, TIME_SHIFT):.2f}x")
        print()

    pop, cap = sgm_population(rows)
    if pop:
        times = {s: [charged_time(r, s, cap) for r in pop] for s in SOLVERS}
        iters = {s: [r[f"{s}_iters"] for r in pop] for s in SOLVERS}
        others = sum(1 for r in pop for s in SOLVERS if r[f"{s}_status"] not in ("OPTIMAL", "TIME_LIMIT"))
        others_note = f", {others} non-timeout failures charged {cap:.0f}s" if others and cap else ""
        print(f"SGM over {len(pop)} instances "
              f"({len(rows) - len(pop)} without OPTIMAL or TIME_LIMIT excluded{others_note}):")
        for shift in (1, TIME_SHIFT):
            print(f"  SGM time  (s={shift:<2}): ratio {sgm_ratio(times, shift):.2f}x")
        print(f"  SGM iters (s={ITER_SHIFT}): ratio {sgm_ratio(iters, ITER_SHIFT):.2f}x")
        print()

    for winner, loser in (("HSD", "PDLPx"), ("PDLPx", "HSD")):
        wins = [r["instance"] for r in rows
                if r[f"{winner}_status"] == "OPTIMAL" and r[f"{loser}_status"] != "OPTIMAL"]
        if wins:
            print(f"{winner} solved, {loser} didn't ({len(wins)}):")
            for inst in wins:
                print(f"  {inst}")
            print()


def run(suite, date, tol=DEFAULT_TOL, output=None):
    hsd, presolved = load_hsd(suite, date)
    config = f"{suite}_{tol}" + ("_presolve" if presolved else "")
    rows = combine(hsd, load_pdlpx(suite, config))
    output = output or comparison_csv(suite, date)
    write_csv(rows, output)
    print(f"cuPDLPx config: {config}")
    print(f"Output: {output}\n")
    print_summary(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=sorted(SUITE_LABEL), default="miplib")
    p.add_argument("--date", required=True, help="cuHSDLP result folder date, e.g. 20260712")
    p.add_argument("--tol", default=DEFAULT_TOL, help="tolerance the cuHSDLP runs used, e.g. 1e-4, 1e-8")
    p.add_argument("--output", type=Path, default=None, help="output CSV path")
    args = p.parse_args()
    run(args.suite, args.date, args.tol, args.output)


if __name__ == "__main__":
    main()
