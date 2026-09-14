#!/usr/bin/env python3.12
"""Plot an infeasible benchmark summary: outcome donuts per arm and solver, and detection ECDFs on raw MPS."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from common import FIGURE_DIR, read_csv_rows
from infeas_datasets import INFEAS_DATASETS
from infeas_scoring import infeasible_outcome
from infeas_summarize import ARMS, SUMMARY_SOLVERS, summary_csv

RC = {
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.edgecolor":     "#aaaaaa",
    "axes.linewidth":     0.7,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    "xtick.minor.size":   2.0,
    "xtick.color":        "#555555",
    "ytick.color":        "#555555",
    "figure.facecolor":   "white",
    "savefig.facecolor":  "white",
    "text.color":         "#222222",
}

# outcome -> (color, short label, legend label); the two wrong answers are reds, the non-answers blue/grey.
OUTCOMES = {
    "detected":      ("#2ca02c", "DET", "Detected"),
    "false_optimal": ("#d62728", "OPT", "False optimal"),
    "opposite":      ("#ff9896", "OPP", "Opposite certificate"),
    "no_answer":     ("#9ecae1", "NA",  "No answer"),
    "not_run":       ("#e0e0e0", "NR",  "Not run"),
}
ARM_LABEL = {"raw": "raw MPS", "pslp": "PSLP-reduced"}
PALETTE = ["#2171B5", "#CB181D", "#e6820a", "#2ca02c",
           "#6a51a3", "#525252", "#00868B"]


def outcome_of(r, key):
    """Which of the five outcomes this solver's answer falls into."""
    status = r.get(f"{key}_status") or ""
    return infeasible_outcome(status, r["instance"]) if status else "not_run"


def num_of(r, key, field):
    v = r.get(f"{key}_{field}", "")
    try:
        return float(v) if v else None
    except ValueError:
        return None


def plot_status_grid(rows, solvers, path):
    """Outcome donuts, arm (row) x solver (col), so one arm reads left to right."""
    fig, axes = plt.subplots(len(ARMS), len(solvers),
                             figsize=(3.4 * len(solvers), 3.6 * len(ARMS)))

    global_present = set()
    for ri, arm in enumerate(ARMS):
        for ci, solver in enumerate(solvers):
            ax = axes[ri, ci]
            key = f"{solver}_{arm}"
            counts = {o: 0 for o in OUTCOMES}
            for r in rows:
                counts[outcome_of(r, key)] += 1

            present = [o for o in OUTCOMES if counts[o] > 0]
            global_present.update(present)
            ax.pie(
                [counts[o] for o in present],
                colors=[OUTCOMES[o][0] for o in present],
                startangle=90, counterclock=False,
                wedgeprops=dict(width=0.44, edgecolor="white", linewidth=2.5),
            )

            det = 100.0 * counts["detected"] / len(rows)
            ax.text(0,  0.08, f"{det:.0f}%",
                    ha="center", va="center",
                    fontsize=20, fontweight="bold", color="black")
            ax.text(0, -0.14, "detected",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color="black")

            if ri == 0:
                ax.set_title(solver, fontsize=16, fontweight="bold",
                             pad=10, color="#222222")

            # Counts are stacked vertically so they do not collide in narrow panels.
            for i, o in enumerate(present):
                color, short, _ = OUTCOMES[o]
                ax.text(0.5, -0.02 - i * 0.075, f"{short} {counts[o]}",
                        ha="center", va="top",
                        fontsize=11, fontweight="bold", color=color,
                        transform=ax.transAxes)

    for ri, arm in enumerate(ARMS):
        y = 1.0 - (ri + 0.5) / len(ARMS)
        fig.text(0.005, y, ARM_LABEL[arm],
                 va="center", ha="left", rotation=90,
                 fontsize=17, fontweight="bold", color="black")

    handles = [mpatches.Patch(color=color, label=label)
               for o, (color, _, label) in OUTCOMES.items() if o in global_present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.005),
               handlelength=1.6, handleheight=1.2, handletextpad=0.5)

    fig.tight_layout(rect=(0.035, 0.05, 1, 0.985))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def draw_ecdf(ax, rows, configs, field, floor, xlabel, title):
    n_total = len(rows)
    series = {}
    for key, _, _ in configs:
        series[key] = sorted(
            max(floor, v)
            for r in rows
            if outcome_of(r, key) == "detected"
            and (v := num_of(r, key, field)) is not None
        )
    # One shared right edge, so a solver that detects quickly does not look as if it stopped early.
    x_max = max((vals[-1] for vals in series.values() if vals), default=floor)

    for key, label, color in configs:
        vals = series[key]
        if not vals:
            continue
        n = len(vals)
        det_rate = n / n_total
        xs = [floor] + vals
        ys = [0.0] + [i / n_total for i in range(1, n + 1)]
        if det_rate < 1.0:
            # Carried to the right edge so the gap to 100% is the fraction never detected.
            xs.append(x_max)
            ys.append(det_rate)
        ax.step(xs, ys, where="post", color=color, lw=2.2, alpha=0.92, label=label)

    ax.axhline(1.0, color="#aaaaaa", lw=1.0, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(floor, x_max * 1.08)
    ax.set_ylim(-0.01, 1.06)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", labelpad=6)
    ax.set_ylabel("Fraction detected", fontsize=14, fontweight="bold", labelpad=6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="x", which="minor", length=2, color="#cccccc")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.grid(True, which="major", color="#e4e4e4", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.legend(fontsize=11, frameon=True, framealpha=0.95,
              edgecolor="#dddddd", loc="upper left", ncol=2,
              handlelength=2.0, handletextpad=0.5)


def plot_profiles(rows, solvers, path):
    """Detection ECDF against iterations and solve time on the raw MPS arm."""
    configs = [(f"{solver}_raw", solver, PALETTE[i % len(PALETTE)]) for i, solver in enumerate(solvers)]
    fig, (ax_iters, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_ecdf(ax_iters, rows, configs, "iters", 1.0, "Iterations", "(a) Detection vs iterations (raw MPS)")
    draw_ecdf(ax_time, rows, configs, "time", 1e-3, "Solve time (s)", "(b) Detection vs solve time (raw MPS)")
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def run(date, dataset="miplib", output_dir=FIGURE_DIR):
    rows = read_csv_rows(summary_csv(dataset, date))
    solvers = [solver for solver, _ in SUMMARY_SOLVERS]
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        plot_status_grid(rows, solvers, output_dir / f"viz_infeas_{dataset}_status_{date}.png")
        plot_profiles(rows, solvers, output_dir / f"viz_infeas_{dataset}_profiles_{date}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="date of the summary CSV, e.g. 20260904")
    ap.add_argument("--dataset", default="miplib", choices=INFEAS_DATASETS)
    ap.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = ap.parse_args()
    run(args.date, args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
