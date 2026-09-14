#!/usr/bin/env python3
"""Scatter-plot cuHSDLP against cuPDLPx on time, iterations and per-iteration time."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_compare import (ITER_SHIFT, SUITE_LABEL, TIME_SHIFT, charged_time, comparison_csv, load_comparison,
                           sgm_population, shifted_geomean)
from common import FIGURE_DIR

# category -> (color, marker, size, label), in drawing order so "both" lands on top
CATEGORIES = {
    "neither":    ("#9CA3AF", "x", 30, "Neither"),
    "hsd_only":   ("#16A34A", "v", 40, "Only HSD"),
    "pdlpx_only": ("#DC2626", "^", 40, "Only PDLPx"),
    "both":       ("#2563EB", "o", 24, "Both OPTIMAL"),
}


def classify(row):
    """Which solvers reached OPTIMAL: 'both', 'hsd_only', 'pdlpx_only' or 'neither'."""
    h = row["HSD_status"] == "OPTIMAL"
    p = row["PDLPx_status"] == "OPTIMAL"
    if h and p:
        return "both"
    if h:
        return "hsd_only"
    if p:
        return "pdlpx_only"
    return "neither"


def per_iter_ms(row, solver):
    iters = row[f"{solver}_iters"]
    return 1000.0 * row[f"{solver}_time"] / iters if iters > 0 else None


def scatter_plot(rows, pop, value, shift, xlabel, ylabel, title, out_path):
    """Log-log scatter of value(row, solver), annotated with the SGM ratio over the tables' population."""
    points = {key: ([], []) for key in CATEGORIES}
    for row in rows:
        x, y = value(row, "PDLPx"), value(row, "HSD")
        if x is not None and y is not None and x > 0 and y > 0:
            xs, ys = points[classify(row)]
            xs.append(x)
            ys.append(y)
    all_vals = [v for xs, ys in points.values() for v in xs + ys]
    if not all_vals:
        return
    lo = 10 ** (math.floor(math.log10(min(all_vals))) - 0.3)
    hi = 10 ** (math.ceil(math.log10(max(all_vals))) + 0.3)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, hi], [lo, hi], color="#D1D5DB", linewidth=1, zorder=1)
    for key, (color, marker, size, label) in CATEGORIES.items():
        xs, ys = points[key]
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, s=size, alpha=0.75, edgecolors="white",
                       linewidths=0.3, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    # Each solver's SGM is taken separately and then divided; shift 0 needs positive values.
    pairs = [(value(r, "PDLPx"), value(r, "HSD")) for r in pop]
    pairs = [(x, y) for x, y in pairs
             if x is not None and y is not None and (shift > 0 or (x > 0 and y > 0))]
    if pairs:
        xs, ys = zip(*pairs)
        sx = shifted_geomean(xs, shift)
        ratio = shifted_geomean(ys, shift) / sx if sx > 0 else float("nan")
        label = f"SGM ratio (s={shift:g}): {ratio:.2f}x" if shift else f"geo-mean ratio: {ratio:.2f}x"
        ax.text(0.97, 0.03, f"{label}\n({len(pairs)} instances)",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    for path in (out_path, out_path.with_suffix(".png")):
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  {path}")
    plt.close(fig)


def run(suite, date, output_dir=FIGURE_DIR):
    rows = load_comparison(suite, date)
    output_dir.mkdir(parents=True, exist_ok=True)
    pop, cap = sgm_population(rows)
    print(f"Generating plots from {comparison_csv(suite, date)}:")
    # (file key, value, SGM shift, axis unit, title); failed runs are charged the cap in time only.
    for key, value, shift, unit, name in (
        ("time", lambda r, s: charged_time(r, s, cap), TIME_SHIFT, "time (sec)", "Solve Time"),
        ("iters", lambda r, s: r[f"{s}_iters"], ITER_SHIFT, "iterations", "Iterations"),
        ("per_iter", per_iter_ms, 0, "per-iter time (ms)", "Per-iteration Time"),
    ):
        scatter_plot(rows, pop, value, shift,
                     xlabel=f"cuPDLPx {unit}", ylabel=f"cuHSDLP {unit}",
                     title=f"{name}: cuHSDLP vs cuPDLPx ({SUITE_LABEL[suite]})",
                     out_path=output_dir / f"scatter_{key}_{suite}.pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="miplib", choices=sorted(SUITE_LABEL))
    p.add_argument("--date", required=True, help="comparison CSV date, e.g. 20260712")
    p.add_argument("--output-dir", type=Path, default=FIGURE_DIR, help="directory for output figures")
    args = p.parse_args()
    run(args.suite, args.date, args.output_dir)


if __name__ == "__main__":
    main()
