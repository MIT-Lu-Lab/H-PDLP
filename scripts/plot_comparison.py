#!/usr/bin/env python3
"""Plot cuHSDLP vs cuPDLPx comparison from the comparison CSV."""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

# Time/iter limits for unsolved instances (placed at plot boundary)
TIME_LIMIT = 3600
ITER_LIMIT = 100_000_000


def parse_args():
    p = argparse.ArgumentParser(description="Plot cuHSDLP vs cuPDLPx comparison.")
    p.add_argument("--benchmark", default="miplib",
                   choices=["small", "miplib", "mittelmann"],
                   help="benchmark suite (selects input CSV and output filename suffix)")
    p.add_argument("--input", default=None, help="comparison CSV path (overrides --benchmark)")
    p.add_argument("--output-dir", default=FIG_DIR, help="directory for output PDFs")
    return p.parse_args()


def load_csv(path):
    """Load comparison CSV into list of dicts."""
    if not os.path.isfile(path):
        sys.exit(f"Error: {path} not found")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Shifted geometric mean (Achterberg/SCIP convention, used by the Mittelmann
# benchmarks): SGM_s(x) = exp(mean(log(xi + s))) - s.  Standard shifts are
# s = 10 for time (seconds) and s = 100 for iterations.  A shift of 0 reduces
# to the plain geometric mean (used for the rate-like per-iteration metric,
# which has no Achterberg-standard shift).
def shifted_geomean(values, shift):
    """Return the shifted geometric mean of `values` with the given shift."""
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def classify(row):
    """Return category: 'both', 'hsd_only', 'pdlpx_only', or 'neither'."""
    h = row["HSD_status"] == "OPTIMAL"
    p = row["PDLPx_status"] == "OPTIMAL"
    if h and p:
        return "both"
    if h:
        return "hsd_only"
    if p:
        return "pdlpx_only"
    return "neither"


def scatter_plot(rows, xkey, ykey, xlabel, ylabel, title, out_path, shift):
    """Draw a log-log scatter comparing two solvers on one metric.

    `shift` is the shifted-geometric-mean shift for the annotation (10 for time,
    100 for iters, 0 for the rate-like per-iter metric).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Collect data by category
    categories = {
        "both":       {"x": [], "y": [], "c": "#2563EB", "m": "o", "s": 24, "label": "Both OPTIMAL"},
        "pdlpx_only": {"x": [], "y": [], "c": "#DC2626", "m": "^", "s": 40, "label": "Only PDLPx"},
        "hsd_only":   {"x": [], "y": [], "c": "#16A34A", "m": "v", "s": 40, "label": "Only HSD"},
        "neither":    {"x": [], "y": [], "c": "#9CA3AF", "m": "x", "s": 30, "label": "Neither"},
    }

    # Axis limit for unsolved instances (per_iter is well-defined whenever iters>0,
    # so it has no clamping behavior — skip undefined values instead)
    # Failed solvers are clamped to the limit actually used in these runs, which the
    # timed-out runs themselves record (their solve time is the wall-clock cap).  Read
    # it off the data rather than hardcoding, so each suite gets its own cap; fall back
    # to the nominal constant if nothing timed out.
    if "per_iter" in xkey:
        limit = None
    else:
        fallback = TIME_LIMIT if "time" in xkey else ITER_LIMIT
        recorded = [float(row[k]) for row in rows for k, st in ((xkey, "PDLPx_status"), (ykey, "HSD_status"))
                    if row[st] == "TIME_LIMIT" and row[k]]
        limit = max(recorded) if recorded else fallback

    # SGM population (matches compare_solvers.py): every genuine optimization run --
    # at least one solver OPTIMAL, or some solver timed out.  Instances both solvers
    # agree are infeasible/unbounded carry no optimum to time, so they are excluded.
    sgm_x, sgm_y = [], []
    for row in rows:
        cat = classify(row)
        if limit is None:
            if not row[xkey] or not row[ykey]:
                continue
            xv = float(row[xkey]); yv = float(row[ykey])
        else:
            xv = float(row[xkey]) if row[xkey] else limit
            yv = float(row[ykey]) if row[ykey] else limit
            if row["PDLPx_status"] != "OPTIMAL":
                xv = max(xv, limit)
            if row["HSD_status"] != "OPTIMAL":
                yv = max(yv, limit)

        # Skip zero values (can't plot on log scale)
        if xv <= 0 or yv <= 0:
            continue

        categories[cat]["x"].append(xv)
        categories[cat]["y"].append(yv)
        if cat != "neither" or "TIME_LIMIT" in (row["HSD_status"], row["PDLPx_status"]):
            sgm_x.append(xv); sgm_y.append(yv)

    # Determine axis range
    all_vals = []
    for cat in categories.values():
        all_vals.extend(cat["x"] + cat["y"])
    if not all_vals:
        return
    lo = 10 ** (math.floor(math.log10(min(all_vals))) - 0.3)
    hi = 10 ** (math.ceil(math.log10(max(all_vals))) + 0.3)

    # Diagonal (equal performance)
    ax.plot([lo, hi], [lo, hi], color="#D1D5DB", linewidth=1, zorder=1)

    # Scatter by category (plot 'both' last so it's on top)
    for key in ["neither", "hsd_only", "pdlpx_only", "both"]:
        cat = categories[key]
        if not cat["x"]:
            continue
        ax.scatter(cat["x"], cat["y"], c=cat["c"], marker=cat["m"],
                   s=cat["s"], alpha=0.75, edgecolors="white", linewidths=0.3,
                   label=cat["label"], zorder=3)

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

    # Annotate the shifted-geometric-mean ratio over the SGM population (timeouts
    # charged the limit), matching what is plotted.  Following the Achterberg/Mittelmann
    # convention, each solver's SGM is computed separately and then divided
    # (ratio = SGM_HSD / SGM_PDLPx).
    xs = [x for x, y in zip(sgm_x, sgm_y) if x > 0 and y > 0]
    ys = [y for x, y in zip(sgm_x, sgm_y) if x > 0 and y > 0]
    if xs:
        sx = shifted_geomean(xs, shift)
        sy = shifted_geomean(ys, shift)
        ratio = sy / sx if sx > 0 else float("nan")
        label = f"SGM ratio (s={shift:g}): {ratio:.2f}x" if shift else f"geo-mean ratio: {ratio:.2f}x"
        ax.text(0.97, 0.03, f"{label}\n({len(xs)} instances)",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    png_path = os.path.splitext(out_path)[0] + ".png"
    for path in (out_path, png_path):
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")
    print(f"  {png_path}")


def main():
    args = parse_args()
    csv_path = args.input or os.path.join(RESULT_DIR, f"comparison_cuHSDLP_vs_cuPDLPx_{args.benchmark}.csv")
    rows = load_csv(csv_path)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    suffix = args.benchmark
    title_suffix = {"small": "MIPLIB (small)", "miplib": "MIPLIB", "mittelmann": "Mittelmann"}[args.benchmark]

    # Derived metric: per-iteration time in milliseconds
    for r in rows:
        for prefix in ("HSD", "PDLPx"):
            t, k = r.get(f"{prefix}_time"), r.get(f"{prefix}_iters")
            r[f"{prefix}_per_iter_ms"] = (
                f"{1000.0 * float(t) / int(k):.6f}" if t and k and int(k) > 0 else ""
            )

    print(f"Generating plots from {csv_path}:")

    scatter_plot(rows,
                 xkey="PDLPx_time", ykey="HSD_time",
                 xlabel="cuPDLPx time (sec)", ylabel="cuHSDLP time (sec)",
                 title=f"Solve Time: cuHSDLP vs cuPDLPx ({title_suffix})",
                 out_path=os.path.join(out, f"scatter_time_{suffix}.pdf"), shift=10)

    scatter_plot(rows,
                 xkey="PDLPx_iters", ykey="HSD_iters",
                 xlabel="cuPDLPx iterations", ylabel="cuHSDLP iterations",
                 title=f"Iterations: cuHSDLP vs cuPDLPx ({title_suffix})",
                 out_path=os.path.join(out, f"scatter_iters_{suffix}.pdf"), shift=100)

    scatter_plot(rows,
                 xkey="PDLPx_per_iter_ms", ykey="HSD_per_iter_ms",
                 xlabel="cuPDLPx per-iter time (ms)", ylabel="cuHSDLP per-iter time (ms)",
                 title=f"Per-iteration Time: cuHSDLP vs cuPDLPx ({title_suffix})",
                 out_path=os.path.join(out, f"scatter_per_iter_{suffix}.pdf"), shift=0)


if __name__ == "__main__":
    main()
